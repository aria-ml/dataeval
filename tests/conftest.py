from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal, TypeVar
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.random import randint, random
from numpy.typing import NDArray

from dataeval.config import set_seed
from dataeval.data import Metadata
from dataeval.typing import ObjectDetectionTarget

BoxLike = (
    NDArray[np.number] | Sequence[int] | Sequence[float] | tuple[int, int, int, int] | tuple[float, float, float, float]
)

_TLabels = TypeVar("_TLabels", Sequence[int], Sequence[Sequence[int]])

TEMP_CONTENTS = "ABCDEF1234567890"

pytest_plugins = ["tests.fixtures.metadata"]

set_seed(0, all_generators=True)


class MockDataset:
    def __init__(
        self,
        datum_type: Literal["ic", "od"],
        images: NDArray | Sequence[NDArray],
        labels: _TLabels,
        classes: Sequence[str] | None,
    ):
        self._images = images
        self._labels = labels
        self._classes = classes if classes is not None else [str(i) for i in range(MockDataset._find_max(labels) + 1)]
        self._index2label = dict(enumerate([str(x) for x in self._classes]))
        self._id = f"{len(self._images)}_image_{len(self._index2label)}_class_{datum_type}_dataset"

    @property
    def metadata(self) -> dict[str, Any]:
        return {"id": self._id, "index2label": self._index2label}

    def __len__(self):
        return len(self._images)

    @staticmethod
    def _find_max(arr: Any) -> Any:
        if not isinstance(arr, bytes | str) and isinstance(arr, Iterable | Sequence | list):
            nested = [x for x in [MockICDataset._find_max(x) for x in arr] if x is not None]
            return max(nested) if len(nested) > 0 else None
        return arr


class MockICDataset(MockDataset):
    def __init__(
        self,
        images: NDArray | Sequence[NDArray],
        labels: Sequence[int],
        metadata: Sequence[dict[str, Any]] | None = None,
        classes: Sequence[str] | None = None,
    ):
        super().__init__("ic", images, labels, classes)

        self._metadata = MockICDataset._listify_dict(metadata) if metadata else None

    def __getitem__(self, idx: int, /) -> tuple[NDArray, NDArray, dict[str, Any]]:
        one_hot = [0.0] * len(self._index2label)
        labels_at_idx = self._labels[idx]
        if isinstance(labels_at_idx, Sequence):
            raise TypeError("Labels must be 1xDim Sequence[int] type!")

        labels_at_idx = int(labels_at_idx)
        if isinstance(labels_at_idx, int):
            one_hot[labels_at_idx] = 1.0
            merged_metadata = {"id": idx} | self._metadata[idx] if self._metadata else {"id": idx}
            return (
                self._images[idx],
                np.asarray(one_hot),
                merged_metadata,
            )
        raise TypeError("Labels must be Sequence[int] type!")

    @staticmethod
    def _listify_dict(input_dict: Sequence[dict[str, Any]] | dict[str, Sequence[Any]]):
        if isinstance(input_dict, dict):
            return [{k: v[i] for k, v in input_dict.items()} for i in range(len(next(iter(input_dict.values()))))]
        return input_dict


class MockODDataset(MockDataset):
    def __init__(
        self,
        images: NDArray | Sequence[NDArray],
        labels: Sequence[Sequence[int]],
        bboxes: NDArray | Sequence[NDArray] | Sequence[Sequence[NDArray]] | Sequence[Sequence[Sequence[float]]],
        classes: Sequence[str] | None = None,
    ):
        super().__init__("od", images, labels, classes)

        # self._bboxes = bboxes
        self._bboxes = [
            [np.asarray(box).tolist() if not isinstance(box, list) else box for box in bbox] for bbox in bboxes
        ]
        self._metadata = None

    def _get_mock_od_target(self, labels: Sequence[int], bboxes: Sequence[Sequence[float]], class_count: int):
        return_mock = Mock()
        return_mock._labels = labels
        return_mock.labels = labels
        return_mock._boxes = bboxes
        return_mock.boxes = bboxes
        one_hot = [[0.0] * class_count] * len(labels)
        for i, label in enumerate(labels):
            one_hot[i][label] = 1.0
        return_mock._scores = one_hot
        return_mock.scores = one_hot
        return_mock.__class__.return_value = ObjectDetectionTarget
        return return_mock

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx: int, /) -> tuple[NDArray, NDArray, dict[str, Any]]:
        merged_metadata = {"id": idx} | self._metadata[idx] if self._metadata else {"id": idx}
        label_sequence = self._labels[idx]
        if isinstance(label_sequence, Sequence):
            return (
                self._images[idx],
                self._get_mock_od_target(label_sequence, self._bboxes[idx], len(self._classes)),
                merged_metadata,
            )
        raise TypeError("Labels must be Sequence[int] type!")


def to_metadata(factors, class_labels, continuous_factor_bins=None, exclude=None, include=None):
    # exclude autogenerated id by default
    if exclude is None:
        exclude = ["id"]
    if isinstance(class_labels[0], str):
        class_names = sorted(set(class_labels))
        class_labels = [class_names.index(i) for i in class_labels]
    else:
        class_names = None
    images = np.zeros((len(class_labels), 1, 16, 16))

    mock_dataset = MockICDataset(images, class_labels, metadata=factors, classes=class_names)
    metadata = Metadata(
        mock_dataset,  # type: ignore
        continuous_factor_bins=continuous_factor_bins,
        exclude=exclude,
        include=include,
    )

    return metadata


def get_images(count: int, channels: int | None, dims: int | None):
    channels = channels or 3
    dims = dims or 64
    return [np.random.random((channels, dims, dims)) for _ in range(count)]


def get_bboxes(count: int, boxes_per_image: int, as_float: bool):
    boxes = []
    for _ in range(count):
        box = []
        for _ in range(boxes_per_image):
            if as_float:
                x0, y0 = (random() * 24), (random() * 24)
                x1, y1 = x0 + 1 + (random() * 23), y0 + 1 + (random() * 23)
            else:
                x0, y0 = randint(0, 24), randint(0, 24)
                x1, y1 = x0 + randint(1, 24), y0 + randint(1, 24)
            box.append([x0, y0, x1, y1])
        boxes.append(np.asarray(box))
    return boxes


@pytest.fixture(scope="session")
def DATA_1():
    return get_images(10, 1, 64)


@pytest.fixture(scope="session")
def DATA_3():
    return get_images(10, 3, 64)


def _get_dataset(
    images: list[np.ndarray] | tuple[int, int | None, int | None] | int | None,
    targets_per_image: int | None = None,
    as_float: bool = False,
    override: list[BoxLike] | dict[int, list[BoxLike]] | None = None,
):
    if images is None or isinstance(images, int):
        images = (images or 10, None, None)
    if isinstance(images, tuple):
        images = get_images(images[0], images[1], images[2])
    length = len(images)
    override_dict = dict(enumerate(override)) if isinstance(override, list) else override
    if targets_per_image:
        labels = [[0 for _ in range(targets_per_image)] for _ in range(length)]
        bboxes = get_bboxes(length, targets_per_image, as_float)
        if override_dict is not None:
            for i, boxes in override_dict.items():
                bboxes[i] = boxes
        return MockODDataset(images, labels, bboxes)
    else:
        labels = [0 for _ in range(length)]
        return MockICDataset(images, labels)


@pytest.fixture
def get_ic_dataset():
    return _get_dataset


@pytest.fixture
def get_od_dataset():
    return _get_dataset


@pytest.fixture(scope="session")
def RNG() -> np.random.Generator:
    return np.random.default_rng(0)
