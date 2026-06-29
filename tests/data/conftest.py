from collections.abc import Mapping, Sequence

import numpy as np
import pytest


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
    def __init__(self, labels: Sequence[int]) -> None:
        box = [0.0, 0.0, 1.0, 1.0]
        self.labels = np.asarray(labels, dtype=np.intp)
        self.boxes = np.asarray([box] * len(labels), dtype=np.float32)
        self.scores = np.asarray([1.0] * len(labels), dtype=np.float32)


class MockODDataset:
    """Minimal object-detection dataset: per-image detection label lists."""

    def __init__(self, detections: Sequence[Sequence[int]], index2label: Mapping[int, str]) -> None:
        self._detections = [list(d) for d in detections]
        self.metadata = {"id": "mock-od", "index2label": dict(index2label)}

    def __len__(self) -> int:
        return len(self._detections)

    def __getitem__(self, index: int):
        return np.zeros((3, 4, 4), dtype=np.float32), _ODTarget(self._detections[index]), {"id": index}


@pytest.fixture
def ic_dataset():
    return MockICDataset


@pytest.fixture
def od_dataset():
    return MockODDataset
