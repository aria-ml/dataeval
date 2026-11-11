"""Per-frame elapsed time and exact repeats."""

__all__ = []

from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

from dataeval._experimental import experimental
from dataeval.core._compute_stats import AddressedResult
from dataeval.core._video_runner import FrameView, VideoClock, VideoOptions, run_videos, walk
from dataeval.flags import FrameStats
from dataeval.protocols import MultiobjectTrackingDataset, ProgressCallback, VideoStream
from dataeval.types import Aggregator, SourceIndex

_FIRST_FRAME = "first_frame"
_MISSING_TIME = "missing_time"
_NONPOSITIVE_STEP = "nonpositive_step"
_OK = "ok"

# Share of a sequence's frames that must hold a valid step for a timing roll-up to answer. The first frame is always
# NaN and counts as a frame, so any value below 1.0 admits it. 0.25 is the threshold ego_stats trusts
# (`_TRUST_COVERAGE`), not a value tuned on data, so it is provisional.
_TIMING_COVERAGE = 0.25


def _elapsed(times: Sequence[float | None]) -> tuple[NDArray[np.float64], NDArray[np.object_]]:
    """Return each frame's seconds since the previous one, and why a value is missing.

    A step is a number only where both timestamps are present and finite and the step is positive. The status
    names the first cause that applies, in the order first frame, missing time, nonpositive step.
    """
    t = np.array([np.nan if v is None else v for v in times], dtype=np.float64)
    t[~np.isfinite(t)] = np.nan
    elapsed = np.full(t.size, np.nan)
    status = np.full(t.size, _OK, dtype=object)
    if t.size == 0:
        return elapsed, status
    step = np.diff(t)
    elapsed[1:] = np.where(step > 0, step, np.nan)
    status[1:][np.isnan(step)] = _MISSING_TIME
    status[1:][step <= 0] = _NONPOSITIVE_STEP
    status[0] = _FIRST_FRAME
    return elapsed, status


def _same_pixels(previous: NDArray[Any], current: NDArray[Any]) -> bool:
    """Whether two frames are identical in shape, dtype and values, with NaNs equal at matching positions only."""
    if previous.shape != current.shape or previous.dtype != current.dtype:
        return False
    return bool(np.array_equal(previous, current, equal_nan=current.dtype.kind in "fc"))


def _has_nan(pixels: NDArray[Any]) -> bool:
    return pixels.dtype.kind in "fc" and bool(np.isnan(pixels).any())


class _FrameRows:
    """One video's per-frame readings, fed by :func:`walk` one frame at a time."""

    def __init__(self, stats: FrameStats) -> None:
        self.stats = stats
        self.numbers: list[int] = []
        self.times: list[float | None] = []
        self.repeat: list[float] = []
        self.pair_missing: list[bool] = []
        self._previous: NDArray[Any] | None = None

    def update(self, view: FrameView) -> None:
        """Take in one frame."""
        self.numbers.append(view.number)
        self.times.append(view.time_s)
        if not self.stats & FrameStats.EXACT_REPEAT:
            return
        # ponytail: compared on `FrameView.pixels`, the (C, H, W) array every statistic reads, which keeps only the
        # first slice of a frame with more than 3 dimensions. Such frames are not a video input. Upgrade path:
        # compare the raw frame.
        current = view.pixels
        if self._previous is None:
            self.repeat.append(np.nan)
            self.pair_missing.append(False)
        else:
            same = _same_pixels(self._previous, current)
            self.repeat.append(1.0 if same else 0.0)
            self.pair_missing.append(same and _has_nan(current))
        # A copy: `view.pixels` is zero-copy, so a decoder that refills one buffer would otherwise leave the
        # previous frame equal to this one.
        self._previous = current.copy()

    def finish(self, clock: VideoClock) -> None:  # noqa: ARG002
        """Release the last frame. `run_videos` keeps every video's consumer until the call returns."""
        self._previous = None

    def elapsed(self) -> tuple[NDArray[np.float64], NDArray[np.object_]]:
        """Return the video's elapsed times and their statuses."""
        return _elapsed(self.times)


def _measure_video(stats: FrameStats, stream: VideoStream) -> _FrameRows:
    """Walk one video and return its rows."""
    rows = _FrameRows(stats)
    walk(stream, [rows], VideoOptions())
    return rows


def _aggregations(stats: FrameStats) -> tuple[Aggregator, ...]:
    """Declare the sequence roll-ups of the selected statistics only."""
    declared: list[Aggregator] = []
    if stats & FrameStats.TIMING:
        declared += [
            Aggregator(how, "unit", "sequence", ("elapsed_time_s",), min_coverage=_TIMING_COVERAGE)
            for how in ("median", "std")
        ]
    if stats & FrameStats.EXACT_REPEAT:
        # Every frame but the first has a value, so no coverage is needed beyond what is there.
        declared.append(
            Aggregator("mean", "unit", "sequence", ("repeat_frame",), min_coverage=0.0, suffix="_fraction"),
        )
    return tuple(declared)


def _join(parts: Sequence[NDArray[Any]], dtype: DTypeLike) -> NDArray[Any]:
    """Concatenate per-video arrays, which is empty when there are no videos."""
    return np.concatenate(parts) if parts else np.empty(0, dtype=dtype)


@experimental
def frame_stats(
    source: VideoStream | Sequence[VideoStream] | MultiobjectTrackingDataset,
    *,
    stats: FrameStats = FrameStats.ALL,
    progress_callback: ProgressCallback | None = None,
) -> AddressedResult[Any]:
    """Measure the time since the previous frame and whether a frame repeats it.

    A clock that skips, stalls or restarts and a frame that is the previous frame again show up in no pixel
    statistic. Each statistic compares a frame with the one before it, so the first frame of every video has no
    value.

    .. versionadded:: 1.3

    Parameters
    ----------
    source : VideoStream, Sequence[VideoStream] or MultiobjectTrackingDataset
        One video's frames, a sequence of videos, or a tracking dataset of them. Rows for every video are
        concatenated in item order.
    stats : FrameStats, default FrameStats.ALL
        The statistics to measure. The default decodes every frame; pass ``FrameStats.TIMING`` to read timestamps
        only.
    progress_callback : ProgressCallback or None, default None
        Called once per finished video.

    Returns
    -------
    AddressedResult
        One row per frame of every video, addressed by ``SourceIndex(item, unit_index, "unit")``, where
        ``unit_index`` is the frame's own number, so the rows attach to a
        :class:`~dataeval.Metadata`'s frames by :meth:`~dataeval.Metadata.add_factors`. Only the selected
        statistics are present.

        ``stats`` holds float64 columns:

        - ``elapsed_time_s``: seconds from the previous decoded frame's reported ``time_s`` to this frame's. It is the
          difference of reported times, not a measurement, and not a difference of frame numbers. NaN for the
          first frame, where either timestamp is missing or not finite (so one missing timestamp leaves two rows
          NaN), and where the step is zero or negative, as with a repeated timestamp, a decreasing clock or a clock
          that restarts. A clip that starts later is one long positive step, not NaN.
        - ``repeat_frame``: 1.0 where the frame equals the previous one in shape, dtype and every value, and 0.0
          otherwise, so a change of shape or dtype is 0.0. NaN for the first frame. Equality is exact:
          perceptual similarity and re-encoding noise read 0.0, and NaN pixels match NaN pixels only at the same
          positions. Frames are compared as the ``(channels, height, width)`` arrays every statistic reads, so a 2-D
          frame equals the same frame with one channel, and only the first slice of a frame with more than three
          axes is compared.

        ``details`` holds one value per row, and :meth:`~dataeval.Metadata.add_factors` does not attach it:

        - ``elapsed_time_status``: ``"first_frame"``, ``"missing_time"``, ``"nonpositive_step"`` or ``"ok"``, the
          reason a value is NaN.
        - ``repeat_pair_has_missing``: True where a frame repeats the one before it and holds NaN pixels, which
          sets apart a repeat that rests on matching NaN positions. False elsewhere.

        ``aggregations`` declares the sequence roll-ups of the selected statistics, which
        :meth:`~dataeval.Metadata.add_factors` applies by default: ``elapsed_time_s_median`` and
        ``elapsed_time_s_std``, which answer when at least a quarter of a sequence's frames hold a valid step, and
        ``repeat_frame_fraction``, the share of frames after the first that repeat.

    Raises
    ------
    ValueError
        If `stats` selects nothing.

    See Also
    --------
    :func:`~dataeval.core.video_stats` : One row per video, with its frame rate and share of redundant frames
    :class:`~dataeval.quality.Duplicates` : Near-duplicate frames, which exact equality does not find

    Notes
    -----
    ``TIMING`` reads timestamps and decodes no pixels. ``EXACT_REPEAT`` decodes each frame once and holds a copy
    of the previous one.

    **Parallelism.** Videos are measured concurrently, one per worker thread, when
    :func:`~dataeval.config.set_max_processes` allows more than one worker. Each worker fetches and decodes its own
    video, so a dataset whose items share one open decoder is not safe to measure this way. Set ``max_processes``
    to 1 for such a dataset.

    Examples
    --------
    Four in-memory frames half a second apart, where the second repeats the first and the fourth the third:

    >>> from types import SimpleNamespace
    >>> from dataeval.core import frame_stats
    >>> frames = [SimpleNamespace(pixels=np.full((1, 2, 2), k // 2), time_s=0.5 * k) for k in range(4)]
    >>> result = frame_stats(frames)
    >>> result["stats"]["elapsed_time_s"]
    array([nan, 0.5, 0.5, 0.5])
    >>> result["stats"]["repeat_frame"]
    array([nan,  1.,  0.,  1.])
    >>> result["details"]["elapsed_time_status"].tolist()
    ['first_frame', 'ok', 'ok', 'ok']

    Attach the rows to a tracking dataset's metadata, and the roll-ups come with them:

    >>> metadata.add_factors(frame_stats(tracking_dataset))  # doctest: +SKIP
    """
    if not stats:
        raise ValueError(
            "stats requests no statistics. Select FrameStats.TIMING, FrameStats.EXACT_REPEAT or FrameStats.ALL.",
        )
    videos = run_videos(source, partial(_measure_video, stats), progress_callback)
    source_index = [SourceIndex(item, number, "unit") for item, rows in enumerate(videos) for number in rows.numbers]
    columns: dict[str, NDArray[Any]] = {}
    details: dict[str, NDArray[Any]] = {}
    if stats & FrameStats.TIMING:
        elapsed = [rows.elapsed() for rows in videos]
        columns["elapsed_time_s"] = _join([e for e, _ in elapsed], np.float64)
        details["elapsed_time_status"] = _join([status for _, status in elapsed], object)
    if stats & FrameStats.EXACT_REPEAT:
        columns["repeat_frame"] = _join([np.array(rows.repeat, dtype=np.float64) for rows in videos], np.float64)
        details["repeat_pair_has_missing"] = _join([np.array(rows.pair_missing, dtype=bool) for rows in videos], bool)
    return AddressedResult(
        stats=columns,
        source_index=source_index,
        aggregations=_aggregations(stats),
        details=details,
    )
