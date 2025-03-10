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

    def __postinit__(self) -> None:
        if self.bboxes is None != self.source is None:
            raise ValueError("Either both bboxes and source must be provided or neither.")
        if (
            len(self.labels) != len(self.scores)
            or self.bboxes is not None
            and len(self.labels) != len(self.bboxes)
            or self.source is not None
            and len(self.labels) != len(self.source)
        ):
            raise ValueError("Labels, scores, bboxes and source must be the same length (if provided).")

    def __len__(self) -> int:
        return len(self.labels)

    def at(self, idx: int) -> Targets:
        if self.source is None or self.bboxes is None:
            return Targets(self.labels[idx], self.scores[idx], None, None)
        else:
            mask = np.where(self.source == idx, True, False)
            return Targets(self.labels[mask], self.scores[mask], self.bboxes[mask], self.source[mask])
