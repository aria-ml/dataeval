"""Zero-copy proxies and helper functions for ``VideoSegments`` and ``VideoStitch``."""

__all__ = []

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import AnnotatedDataset, DatasetMetadata, SingleFrameObjectTrackingTarget, VideoFrame
from dataeval.types._target import detection_count, track_ids_of


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


def inherited_metadata(dataset: AnnotatedDataset[Any], suffix: str) -> DatasetMetadata:
    """Build dataset metadata for a view, preserving ``index2label`` and suffixing ``id``."""
    source_id = str(dataset.metadata.get("id", "dataset"))
    index2label = dataset.metadata.get("index2label", None)
    inherited: dict[str, Any] = {"id": f"{source_id}-{suffix}"}
    if index2label is not None:
        inherited["index2label"] = {int(key): str(value) for key, value in index2label.items()}
    return DatasetMetadata(inherited)  # type: ignore[typeddict-item]
