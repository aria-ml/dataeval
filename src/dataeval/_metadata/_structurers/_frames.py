"""One frame's contribution to a tracking dataset, as it comes off the stream."""

__all__ = []

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class FrameRows(NamedTuple):
    """One frame's contribution to a tracking dataset: its own keys, plus its detections.

    A named tuple rather than a bare one because the walk needs seven fields per frame,
    and ``rows.track_ids`` at the call site says what ``rows[6]`` cannot.
    """

    frame_index: int
    time_s: float | None
    pts: int | None
    labels: NDArray[np.intp]
    boxes: NDArray[np.float32]
    scores: NDArray[np.float32]
    track_ids: NDArray[np.intp]
