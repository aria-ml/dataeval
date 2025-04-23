from __future__ import annotations

__all__ = []

from typing import Sequence

import numpy as np

from dataeval.data._selection import Select, Selection, SelectionStage
from dataeval.typing import Array, ImageClassificationDatum
from dataeval.utils._array import as_numpy


class ClassFilter(Selection[ImageClassificationDatum]):
    """
    Filter the dataset by class.

    Parameters
    ----------
    classes : Sequence[int]
        The classes to filter by.
    """

    stage = SelectionStage.FILTER

    def __init__(self, classes: Sequence[int]) -> None:
        self.classes = classes

    def __call__(self, dataset: Select[ImageClassificationDatum]) -> None:
        if not self.classes:
            return

        selection = []
        for idx in dataset._selection:
            target = dataset._dataset[idx][1]
            if isinstance(target, Array):
                label = int(np.argmax(as_numpy(target)))
            else:
                # ObjectDetectionTarget and SegmentationTarget not supported yet
                raise TypeError("ClassFilter only supports classification targets as an array of confidence scores.")
            if label in self.classes:
                selection.append(idx)

        dataset._selection = selection
