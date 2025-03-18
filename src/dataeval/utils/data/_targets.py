from __future__ import annotations

__all__ = []

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _len(arr: NDArray, dim: int) -> int:
    return 0 if len(arr) == 0 else len(np.atleast_1d(arr) if dim == 1 else np.atleast_2d(arr))


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

    def __post_init__(self) -> None:
        if (self.bboxes is None) != (self.source is None):
            raise ValueError("Either both bboxes and source must be provided or neither.")

        labels = _len(self.labels, 1)
        scores = _len(self.scores, 2) if self.bboxes is None else _len(self.scores, 1)
        bboxes = labels if self.bboxes is None else _len(self.bboxes, 2)
        source = labels if self.source is None else _len(self.source, 1)

        if labels != scores or labels != bboxes or labels != source:
            raise ValueError(
                "Labels, scores, bboxes and source must be the same length (if provided).\n"
                + f"    labels: {self.labels.shape}\n"
                + f"    scores: {self.scores.shape}\n"
                + f"    bboxes: {None if self.bboxes is None else self.bboxes.shape}\n"
                + f"    source: {None if self.source is None else self.source.shape}\n"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def at(self, idx: int) -> Targets:
        if self.source is None or self.bboxes is None:
            return Targets(
                np.atleast_1d(self.labels[idx]),
                np.atleast_2d(self.scores[idx]),
                None,
                None,
            )
        else:
            mask = np.where(self.source == idx, True, False)
            return Targets(
                np.atleast_1d(self.labels[mask]),
                np.atleast_1d(self.scores[mask]),
                np.atleast_2d(self.bboxes[mask]),
                np.atleast_1d(self.source[mask]),
            )
