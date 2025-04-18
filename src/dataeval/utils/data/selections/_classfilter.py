from __future__ import annotations

__all__ = []

from typing import Sequence, TypeVar

import numpy as np

from dataeval.typing import Array, ImageClassificationDatum
from dataeval.utils._array import as_numpy
from dataeval.utils.data._selection import Select, Selection, SelectionStage

TImageClassificationDatum = TypeVar("TImageClassificationDatum", bound=ImageClassificationDatum)


class ClassFilter(Selection[TImageClassificationDatum]):
    """
    Filter and balance the dataset by class.

    Parameters
    ----------
    classes : Sequence[int] or None, default None
        The classes to filter by. If None, all classes are included.
    balance : bool, default False
        Whether to balance the classes.

    Note
    ----
    If `balance` is True, the total number of instances of each class will
    be equalized. This may result in a lower total number of instances.
    """

    stage = SelectionStage.FILTER

    def __init__(self, classes: Sequence[int] | None = None, balance: bool = False) -> None:
        self.classes = classes
        self.balance = balance

    def __call__(self, dataset: Select[TImageClassificationDatum]) -> None:
        if self.classes is None and not self.balance:
            return

        per_class_limit = dataset._size_limit // len(self.classes) if self.classes and self.balance else 0
        class_indices: dict[int, list[int]] = {} if self.classes is None else {k: [] for k in self.classes}
        for i, idx in enumerate(dataset._selection):
            target = dataset._dataset[idx][1]
            if isinstance(target, Array):
                label = int(np.argmax(as_numpy(target)))
            else:
                # ObjectDetectionTarget and SegmentationTarget not supported yet
                raise TypeError("ClassFilter only supports classification targets as an array of confidence scores.")
            if not self.classes or label in self.classes:
                class_indices.setdefault(label, []).append(i)
            if per_class_limit and all(len(indices) >= per_class_limit for indices in class_indices.values()):
                break

        per_class_limit = min(len(c) for c in class_indices.values()) if self.balance else dataset._size_limit
        subselection = sorted([i for v in class_indices.values() for i in v[:per_class_limit]])
        dataset._selection = [dataset._selection[i] for i in subselection]
