"""Zero-copy proxies and helper functions for ``VideoSegments`` and ``VideoStitch``."""

__all__ = []

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Sized
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from dataeval._log import get_logger
from dataeval.protocols import (
    Dataset,
    DatasetMetadata,
    MultiobjectTrackingDatum,
    MultiobjectTrackingTarget,
    SingleFrameObjectTrackingTarget,
    VideoFrame,
    _is_protocol_instance,
)
from dataeval.types._target import detection_count, track_ids_of

_logger = get_logger(__name__)


class _Forwarding:
    """Forward undeclared public attributes to ``_wrapped``."""

    __slots__ = ("_wrapped",)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")
        return getattr(self._wrapped, name)


class _ShiftedFrame(_Forwarding):
    """A :obj:`~dataeval.protocols.VideoFrame` proxy with adjusted index and timing offsets."""

    __slots__ = ("_frame_index", "_pts_shift", "_time_shift")

    def __init__(self, frame: Any, frame_index: int, time_shift: float = 0.0, pts_shift: int = 0) -> None:
        self._wrapped = frame
        self._frame_index = int(frame_index)
        self._time_shift = float(time_shift)
        self._pts_shift = int(pts_shift)

    @property
    def pixels(self) -> Any:
        return self._wrapped.pixels

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def time_s(self) -> float | None:
        value = self._wrapped.time_s
        return None if value is None else float(value) + self._time_shift

    @property
    def pts(self) -> int | None:
        value = self._wrapped.pts
        return None if value is None else int(value) + self._pts_shift

    def __repr__(self) -> str:
        return (
            f"_ShiftedFrame(frame_index={self._frame_index}, time_shift={self._time_shift}, "
            f"pts_shift={self._pts_shift})"
        )


class _OffsetFrameTarget(_Forwarding):
    """A :obj:`~dataeval.protocols.SingleFrameObjectTrackingTarget` proxy that shifts track IDs."""

    __slots__ = ("_offset",)

    def __init__(self, target: Any, offset: int) -> None:
        self._wrapped = target
        self._offset = int(offset)

    @property
    def boxes(self) -> Any:
        return self._wrapped.boxes

    @property
    def labels(self) -> Any:
        return self._wrapped.labels

    @property
    def scores(self) -> Any:
        return self._wrapped.scores

    @property
    def track_ids(self) -> NDArray[np.intp]:
        # Standardize track IDs before applying offset; untracked (-1) IDs are preserved.
        ids = track_ids_of(self._wrapped, detection_count(self._wrapped))
        return np.where(ids >= 0, ids + self._offset, ids)

    def __repr__(self) -> str:
        return f"_OffsetFrameTarget(offset={self._offset})"


class _FrameTracksTarget:
    """A :obj:`~dataeval.protocols.MultiobjectTrackingTarget` wrapping a sequence of frame targets."""

    __slots__ = ("_frame_tracks",)

    def __init__(self, frame_tracks: Iterable[Any]) -> None:
        self._frame_tracks: tuple[Any, ...] = tuple(frame_tracks)

    @property
    def frame_tracks(self) -> Sequence[SingleFrameObjectTrackingTarget]:
        return self._frame_tracks

    def __repr__(self) -> str:
        return f"_FrameTracksTarget(n_frames={len(self._frame_tracks)})"


class _NoDetections:
    """A :obj:`~dataeval.protocols.SingleFrameObjectTrackingTarget` for a frame with nothing annotated."""

    __slots__ = ()

    @property
    def boxes(self) -> NDArray[np.float32]:
        return np.zeros((0, 4), dtype=np.float32)

    @property
    def labels(self) -> NDArray[np.intp]:
        return np.zeros(0, dtype=np.intp)

    @property
    def scores(self) -> NDArray[np.float32]:
        return np.zeros(0, dtype=np.float32)

    @property
    def track_ids(self) -> NDArray[np.intp]:
        return np.zeros(0, dtype=np.intp)

    def __repr__(self) -> str:
        return "_NoDetections()"


_NO_DETECTIONS = _NoDetections()


def _is_stream(candidate: Any) -> bool:
    """Whether `candidate` is a video stream: an iterable whose first frame has pixels.

    An array or tensor is rejected without a decode, so an image dataset costs nothing to rule out. A
    stream costs the decode of its first frame.
    """
    if isinstance(candidate, str | bytes | Mapping) or not isinstance(candidate, Iterable):
        return False
    if hasattr(candidate, "shape") or hasattr(candidate, "__array__"):
        return False
    return hasattr(next(iter(candidate), None), "pixels")


def is_pure_video(data: Any) -> bool:
    """Whether `data` is a dataset of videos with no tracking targets.

    True when the first datum is a bare stream, or a MAITE 3-tuple whose first element is a stream and whose
    second is not a :obj:`~dataeval.protocols.MultiobjectTrackingTarget`. A tracking dataset, an image
    dataset and anything else is False. A 3-tuple is always read as a MAITE datum, never as a stream of three frames.
    """
    try:
        if not isinstance(data, Dataset) or len(data) == 0:
            return False
        datum = data[0]
    except (TypeError, IndexError, KeyError):
        return False
    if isinstance(datum, tuple):
        if len(datum) != 3 or _is_protocol_instance(datum[1], MultiobjectTrackingTarget):
            return False
        datum = datum[0]
    return _is_stream(datum)


class _UntrackedVideos:
    """A tracking dataset over videos that carry no targets: every frame has no detections.

    The frame count of a video is ``len(stream)`` when the stream has one, and otherwise one walk of the
    stream, made on first use and kept.
    """

    def __init__(self, videos: Any) -> None:
        self._videos = videos
        declared = getattr(videos, "metadata", None) or {}
        self.metadata = DatasetMetadata({"id": str(declared.get("id", "dataset"))})  # type: ignore[typeddict-item]
        self._counts: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self._videos)

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        datum = self._videos[index]
        stream, metadata = (datum[0], datum[2]) if isinstance(datum, tuple) else (datum, {"id": index})
        return stream, _FrameTracksTarget([_NO_DETECTIONS] * self._frame_count(index, stream)), metadata

    def _frame_count(self, index: int, stream: Any) -> int:
        if index not in self._counts:
            if isinstance(stream, Sized):
                self._counts[index] = len(stream)
            else:
                _logger.info("Video %d has no length, so its frames are counted by decoding it once.", index)
                self._counts[index] = sum(1 for _ in stream)
        return self._counts[index]

    def __repr__(self) -> str:
        return f"_UntrackedVideos({self._videos!r})"


def as_tracking(data: Any) -> Dataset[MultiobjectTrackingDatum]:
    """Return `data` as a tracking dataset: a dataset of videos with no targets gets no detections in any frame."""
    return cast("Dataset[MultiobjectTrackingDatum]", _UntrackedVideos(data) if is_pure_video(data) else data)


class _LazyStream:
    """A re-iterable :obj:`~dataeval.protocols.VideoStream` backed by a generator factory."""

    __slots__ = ("_describe", "_walk")

    def __init__(self, walk: Callable[[], Iterator[Any]], describe: str = "") -> None:
        self._walk = walk
        self._describe = describe

    def __iter__(self) -> Iterator[VideoFrame]:
        return self._walk()

    def __repr__(self) -> str:
        return f"_LazyStream({self._describe})"


def timing_of(frame: Any, name: str) -> float | None:
    """Return a frame's ``time_s`` or ``pts`` as a float, or None if absent."""
    value = getattr(frame, name, None)
    return None if value is None else float(value)


def checked_index(index: int, count: int, view: str, noun: str) -> int:
    """Resolve a possibly negative index, raising ``IndexError`` if out of range."""
    resolved = index + count if index < 0 else index
    if not 0 <= resolved < count:
        raise IndexError(f"{view} index {index} out of range for {count} {noun}.")
    return resolved


def short_stream(view: str, source_id: Any, n_frames: int, yielded: int) -> str:
    """Format an error message when a video stream yields fewer frames than expected."""
    return f"{view}: video {source_id!r} declares {n_frames} frame target(s) but its stream yielded only {yielded}."


def long_stream(view: str, source_id: Any, n_frames: int) -> str:
    """Format an error message when a video stream yields more frames than expected."""
    return f"{view}: video {source_id!r} yields more frames than its {n_frames} frame target(s)."


def inherited_metadata(dataset: Dataset[Any], suffix: str) -> DatasetMetadata:
    """Build dataset metadata for a view, preserving ``index2label`` and suffixing ``id``.

    A dataset without a ``metadata`` attribute gives its view the id ``dataset`` and no ``index2label``.
    """
    declared = getattr(dataset, "metadata", None) or {}
    source_id = str(declared.get("id", "dataset"))
    index2label = declared.get("index2label", None)
    inherited: dict[str, Any] = {"id": f"{source_id}-{suffix}"}
    if index2label is not None:
        inherited["index2label"] = {int(key): str(value) for key, value in index2label.items()}
    return DatasetMetadata(inherited)  # type: ignore[typeddict-item]
