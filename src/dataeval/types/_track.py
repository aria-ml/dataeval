"""Object track types for video datasets."""

__all__ = [
    "Track",
]

from typing import Any

from numpy.typing import NDArray
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Track:
    """All observations of a single object track within one video sequence.

    Attributes
    ----------
    track_id : int
        The unique target ID assigned by the dataset.
    boxes : NDArray[Any]
        Shape ``(T, 4)`` float32 array of bounding boxes in ``[x1, y1, x2, y2]``
        (xyxy) format, one row per observation.
    frame_indices : NDArray[Any]
        Shape ``(T,)`` int64 array of 0-based frame indices corresponding to
        each row in *boxes*.
    scores : NDArray[Any]
        Shape ``(T,)`` float32 confidence scores (1.0 for ground-truth tracks).
    labels : NDArray[Any]
        Shape ``(T,)`` int64 0-based class indices.  In practice a track
        should be single-class, but the raw per-frame labels are preserved so
        any label inconsistencies in the data remain visible.
    """

    track_id: int
    boxes: NDArray[Any]
    frame_indices: NDArray[Any]
    scores: NDArray[Any]
    labels: NDArray[Any]
