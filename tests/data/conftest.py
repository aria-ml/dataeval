from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from dataeval.protocols import DatumMetadata


class MockICDataset:
    """Minimal image-classification dataset: one-hot targets + index2label."""

    def __init__(self, labels: Sequence[int], index2label: Mapping[int, str]) -> None:
        self._labels = list(labels)
        self._n_classes = len(index2label)  # fixed at init so metadata edits don't break __getitem__
        self.metadata = {"id": "mock-ic", "index2label": dict(index2label)}

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int):
        onehot = np.zeros(self._n_classes, dtype=np.float32)
        onehot[self._labels[index]] = 1.0
        return np.zeros((3, 2, 2), dtype=np.float32), onehot, {"id": index}


class _ODTarget:
    """One image's detections, scored in either layout MAITE permits.

    ``classes`` is None for one confidence per box, or the vocabulary size for the
    ``(N, CLASSES)`` layout every ground-truth dataset in ``maite_datasets`` emits.
    """

    def __init__(self, labels: Sequence[int], classes: int | None = None, confidence: float = 1.0) -> None:
        box = [0.0, 0.0, 1.0, 1.0]
        self.labels = np.asarray(labels, dtype=np.intp)
        self.boxes = np.asarray([box] * len(labels), dtype=np.float32)
        if classes is None:
            self.scores = np.full(len(labels), confidence, dtype=np.float32)
        else:
            self.scores = np.zeros((len(labels), classes), dtype=np.float32)
            self.scores[np.arange(len(labels)), self.labels] = confidence


class MockODDataset:
    """Minimal object-detection dataset: per-image detection label lists.

    ``per_class`` scores every class of ``index2label`` rather than every box, the layout
    a vocabulary-sized score array arrives in.
    """

    def __init__(
        self,
        detections: Sequence[Sequence[int]],
        index2label: Mapping[int, str],
        *,
        per_class: bool = False,
        confidence: float = 1.0,
        dataset_id: str = "mock-od",
    ) -> None:
        self._detections = [list(d) for d in detections]
        self._classes = len(index2label) if per_class else None
        self._confidence = confidence
        self.metadata = {"id": dataset_id, "index2label": dict(index2label)}

    def __len__(self) -> int:
        return len(self._detections)

    def __getitem__(self, index: int) -> tuple[Any, Any, DatumMetadata]:
        target = _ODTarget(self._detections[index], self._classes, self._confidence)
        return np.zeros((3, 4, 4), dtype=np.float32), target, {"id": index}


@pytest.fixture
def ic_dataset():
    return MockICDataset


@pytest.fixture
def od_dataset():
    return MockODDataset
