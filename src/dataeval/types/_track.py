"""Object track types for video datasets, and the frame-geometry rule they are read with."""

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


def frame_size(frame: Any) -> tuple[int | None, int | None]:
    """Pixel width and height of a decoded video frame, or ``(None, None)``.

    MAITE declares :obj:`~dataeval.protocols.VideoFrame` pixels as ``(C, H, W)``, which is
    where the trailing two axes come from. One implementation rather than two, because both
    readers of a frame's size — the structuring walk that records it as metadata and
    :func:`~dataeval.core.track_stats`, which needs it to decide whether a track touches the
    border — have to agree on which axis is which, and a disagreement would silently
    transpose every edge flag.

    A duck-typed stream is not obliged to carry pixels at all, and dispatch does not require
    the full protocol, so anything without a two-or-more dimensional shape answers None
    rather than raising.
    """
    shape = getattr(getattr(frame, "pixels", None), "shape", None)
    if shape is None or len(shape) < 2:
        return None, None
    return int(shape[-1]), int(shape[-2])
