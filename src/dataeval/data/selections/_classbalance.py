from __future__ import annotations

__all__ = []

import numpy as np

from dataeval.data._selection import Select, Selection, SelectionStage
from dataeval.typing import Array, ImageClassificationDatum
from dataeval.utils._array import as_numpy


class ClassBalance(Selection[ImageClassificationDatum]):
    """
    Select indices of a dataset that will equalize the occurrences of all classes.

    Note
    ----
    1. The total number of instances of each class will be equalized which may result
    in a lower total number of instances than specified by the selection limit.
    2. This selection currently only supports classification tasks
    """

    stage = SelectionStage.FILTER

    def __call__(self, dataset: Select[ImageClassificationDatum]) -> None:
        class_indices: dict[int, list[int]] = {}
        for i, idx in enumerate(dataset._selection):
            target = dataset._dataset[idx][1]
            if isinstance(target, Array):
                label = int(np.argmax(as_numpy(target)))
            else:
                # ObjectDetectionTarget and SegmentationTarget not supported yet
                raise TypeError("ClassBalance only supports classification targets as an array of class probabilities.")
            class_indices.setdefault(label, []).append(i)

        per_class_limit = min(min(len(c) for c in class_indices.values()), dataset._size_limit // len(class_indices))
        subselection = sorted([i for v in class_indices.values() for i in v[:per_class_limit]])
        dataset._selection = [dataset._selection[i] for i in subselection]
