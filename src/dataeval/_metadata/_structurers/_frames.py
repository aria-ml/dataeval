"""One frame's contribution to a tracking dataset, as it comes off the stream."""

__all__ = []

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class FrameRows(NamedTuple):
    """One frame's contribution to a tracking dataset: its own keys, plus its detections.

    A named tuple rather than a bare one because the walk needs nine fields per frame,
    and ``rows.track_ids`` at the call site says what ``rows[8]`` cannot.

    ``width`` and ``height`` are the decoded frame's pixel dimensions, read off the shape
    of an array the walk already holds. They are what tells a box touching the border from
    one in open space, and nothing downstream can recover them once the frame is released.
    Like the timings they are None for a frame that does not supply them.
    """

    frame_index: int
    time_s: float | None
    pts: int | None
    width: int | None
    height: int | None
    labels: NDArray[np.intp]
    boxes: NDArray[np.float32]
    scores: NDArray[np.float32]
    track_ids: NDArray[np.intp]
