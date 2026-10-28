"""Per-video primitives shared by ``video_stats`` and ``ego_stats``.

Every per-video timing statistic (duration, frame rate, temporal reductions) reads the one clock
built by :func:`video_clock`. :func:`run_videos` runs one measurement per video, and :func:`walk`
feeds every frame of one video to its consumers in a single pass.
"""

__all__ = []

import sys
from collections.abc import Callable, Sequence
from functools import cached_property
from inspect import getattr_static
from itertools import islice
from typing import Any, Literal, NamedTuple, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from dataeval.config import get_max_processes
from dataeval.core._hash import _phash
from dataeval.core._temporal import DEFAULT_REDUNDANCY_RADIUS
from dataeval.protocols import AnnotatedDataset, Dataset, ProgressCallback, SharpnessMeasure, VideoStream
from dataeval.utils._array import as_numpy
from dataeval.utils._internal import PoolWrapper
from dataeval.utils.preprocessing import normalize_image_shape, to_canonical_grayscale

T = TypeVar("T")

# Readings column holding :attr:`VideoClock.order`, the key temporal reductions order by.
CLOCK = "__clock"


class VideoClock(NamedTuple):
    """Ordering key and timing of one video's frames.

    Attributes
    ----------
    order : NDArray[np.float64] or None
        One ordering key per frame, or None when time cannot order the frames.
    dt : float
        Median positive step between frames, NaN when undefined.
    duration : float
        Span of ``order`` plus ``dt``, NaN when undefined.
    """

    order: NDArray[np.float64] | None
    dt: float
    duration: float


def video_clock(time_s: Sequence[float | None]) -> VideoClock:
    """Build a video's clock from its per-frame timestamps.

    ``order`` is ``time_s`` itself when every timestamp is present and strictly increasing. Otherwise,
    when ``dt`` is defined and every timestamp is present, it is an effective clock that keeps stream
    order and advances by each positive step, or by ``dt`` where the step is not positive (as at the
    clip boundaries of stitched video). Otherwise it is None.

    Parameters
    ----------
    time_s : Sequence[float | None]
        Timestamp in seconds of each frame in stream order, None where missing.

    Returns
    -------
    VideoClock
        The ordering key, typical frame interval and duration.
    """
    t = np.array([np.nan if v is None else v for v in time_s], dtype=np.float64)
    present = bool(np.isfinite(t).all())
    steps = np.diff(t)
    positive = steps[steps > 0] if present else steps[:0]
    dt = float(np.median(positive)) if positive.size else float("nan")

    if present and bool((steps > 0).all()):
        order: NDArray[np.float64] | None = t
    elif present and not np.isnan(dt):
        order = np.concatenate(([0.0], np.cumsum(np.where(steps > 0, steps, dt))))
    else:
        order = None

    duration = float(order[-1] - order[0] + dt) if order is not None and order.size else float("nan")
    return VideoClock(order, dt, duration)


class VideoOptions(NamedTuple):
    """Per-call settings every calculator and frame view of one ``video_stats`` call shares."""

    redundancy_radius: int = DEFAULT_REDUNDANCY_RADIUS
    ego_gap: int = 1
    ego_proc_short_side: int = 360
    ego_flow: Literal["dis", "lk"] = "dis"
    ego_grid_target: int = 40
    ego_mask: NDArray[np.bool_] | None = None
    # None when no blur map is wanted, so a calculator never pays for a measure nobody reads.
    blur_measure: SharpnessMeasure | None = None
    blur_sigma: float = 6.0
    persistence_samples: int = 64


class FrameView:
    """One frame, with each derived view computed on first read and shared by every consumer.

    Laziness is what keeps a walk from paying for a view no selected statistic reads, so
    consumers declare no needs up front.
    """

    def __init__(self, frame: Any, index: int, options: VideoOptions) -> None:
        self._frame = frame
        self.index = index
        self.options = options
        # Read off the frame object, never through the pixels, so timing costs no decode.
        self.time_s: float | None = getattr(frame, "time_s", None)

    @cached_property
    def pixels(self) -> NDArray[Any]:
        """The frame as a ``(C, H, W)`` array."""
        return normalize_image_shape(as_numpy(self._frame.pixels))

    @cached_property
    def gray(self) -> NDArray[np.uint8]:
        """The frame as canonical ``(H, W)`` uint8 grayscale."""
        return to_canonical_grayscale(self.pixels)

    @cached_property
    def gray_small(self) -> NDArray[np.uint8]:
        """:attr:`gray` shrunk to ``options.ego_proc_short_side``, the frame ego flow runs on."""
        # Imported here because _ego_stats imports this module.
        from dataeval.core._ego_stats import _downscale

        return _downscale(self.gray, self.options.ego_proc_short_side)

    @cached_property
    def _hashed(self) -> tuple[str, str | None]:
        return _phash(self.gray[None])

    @property
    def phash(self) -> str:
        """The perceptual hash of :attr:`gray`, the same hash Duplicates compares."""
        return self._hashed[0]

    @property
    def phash_warning(self) -> str | None:
        """Why :attr:`phash` is empty, or None when it hashed. Left for the caller to log, once per video."""
        return self._hashed[1]


class FrameConsumer(Protocol):
    """Anything fed one video's frames in stream order, then told the video's clock."""

    def update(self, view: FrameView) -> None:
        """Take in one frame."""
        ...

    def finish(self, clock: VideoClock) -> None:
        """Close out the video, once every frame has been seen."""
        ...


def walk(
    stream: VideoStream,
    consumers: Sequence[FrameConsumer],
    options: VideoOptions,
    stop_after: int | None = None,
) -> VideoClock:
    """Feed every frame of `stream` to every consumer in one pass, then finish them with its clock.

    Each frame and its view are dropped before the next frame is pulled, so the walk holds one
    frame at a time. `stop_after` stops without pulling the frame beyond it.
    """
    times: list[float | None] = []
    for index, frame in enumerate(islice(stream, stop_after)):
        view = FrameView(frame, index, options)
        for consumer in consumers:
            consumer.update(view)
        times.append(view.time_s)
        del frame, view
    clock = video_clock(times)
    for consumer in consumers:
        consumer.finish(clock)
    return clock


_ABSENT = object()


def _videos(source: Any) -> tuple[int, Callable[[int], Any]]:
    """Count the videos in `source` and return a getter that fetches video ``i`` with one index.

    A list of frames and a list of streams are both indexable, so only the first element tells
    them apart. Its ``pixels`` is looked up statically, so a lazily decoding frame is not decoded,
    and the peeked item is reused as video 0 rather than fetched again.
    """
    if isinstance(source, AnnotatedDataset):
        return len(source), lambda i: source[i][0]
    if not isinstance(source, Dataset) or len(source) == 0:
        return 1, lambda _: source
    first = source[0]
    if getattr_static(first, "pixels", _ABSENT) is not _ABSENT:
        return 1, lambda _: source
    if isinstance(first, tuple) and len(first) == 3:
        return len(source), lambda i: first[0] if i == 0 else source[i][0]
    return len(source), lambda i: first if i == 0 else source[i]


def run_videos(  # noqa: C901
    source: Any,
    measure: Callable[[VideoStream], T],
    progress_callback: ProgressCallback | None = None,
) -> list[T]:
    """Measure every video in `source`, returning one result per video in item order.

    Runs in a thread pool when ``max_processes`` allows more than one worker and there is more
    than one video, otherwise inline. Each worker fetches and decodes its own video. Progress
    is reported once per finished video.
    """
    count, get_stream = _videos(source)
    processes = get_max_processes()
    parallel = count > 1 and processes not in (None, 1)

    def task(i: int) -> tuple[int, T]:
        return i, measure(get_stream(i))

    # Cap OpenCV's own threads while the pool runs, so the two do not oversubscribe the CPUs.
    cv2 = sys.modules.get("cv2") if parallel else None
    previous = cv2.getNumThreads() if cv2 is not None else None
    if cv2 is not None:
        cv2.setNumThreads(1)
    results: dict[int, T] = {}
    try:
        with PoolWrapper(processes if parallel else 1, context="threads") as pool:
            for done, (i, result) in enumerate(pool.imap_unordered(task, range(count)), 1):
                results[i] = result
                if progress_callback:
                    progress_callback(done, total=count)
    finally:
        if cv2 is not None:
            cv2.setNumThreads(previous)
    return [results[i] for i in range(count)]
