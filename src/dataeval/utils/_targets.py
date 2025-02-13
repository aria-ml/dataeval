from __future__ import annotations

__all__ = []

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Targets:
    """
    Dataclass defining targets for image classification or object detection.

    Attributes
    ----------
    labels : NDArray[np.intp]
        Labels (N,) for N images or objects
    scores : NDArray[np.float32]
        Probability scores (N,M) for N images of M classes or confidence score (N,) of objects
    bboxes : NDArray[np.float32] | None
        Bounding boxes (N,4) for N objects in (x0,y0,x1,y1) format
    source : NDArray[np.intp] | None
        Source image index (N,) for N objects
    """

    labels: NDArray[np.intp]
    scores: NDArray[np.float32]
    bboxes: NDArray[np.float32] | None
    source: NDArray[np.intp] | None
