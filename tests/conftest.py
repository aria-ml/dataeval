from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, overload

import numpy as np
import pytest
from maite_datasets import to_image_classification_dataset, to_object_detection_dataset
from numpy.random import randint, random
from numpy.typing import NDArray

from dataeval.config import set_seed
from dataeval.data import Metadata
from dataeval.typing import ImageClassificationDataset, ObjectDetectionDataset

BoxLike = (
    NDArray[np.number] | Sequence[int] | Sequence[float] | tuple[int, int, int, int] | tuple[float, float, float, float]
)

TEMP_CONTENTS = "ABCDEF1234567890"

pytest_plugins = ["tests.fixtures.metadata"]

set_seed(0, all_generators=True)


def to_metadata(factors, class_labels, continuous_factor_bins=None, exclude=None, include=None):
    # exclude autogenerated id by default
    if exclude is None:
        exclude = ["id"]
    if isinstance(class_labels[0], str):
        class_names = sorted(set(class_labels))
        class_labels = [class_names.index(i) for i in class_labels]
    else:
        class_names = None
    dataset = to_image_classification_dataset(
        np.zeros((len(class_labels), 1, 16, 16)), labels=class_labels, metadata=factors, classes=class_names
    )
    metadata = Metadata(
        dataset,  # type: ignore
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


@overload
def _get_dataset(
    images: list[np.ndarray] | tuple[int, int | None, int | None] | int | None,
    targets_per_image: int,
    as_float: bool = False,
    override: list[BoxLike] | dict[int, list[BoxLike]] | None = None,
) -> ObjectDetectionDataset: ...


@overload
def _get_dataset(
    images: list[np.ndarray] | tuple[int, int | None, int | None] | int | None,
    targets_per_image: Literal[None] = None,
    as_float: bool = False,
    override: list[BoxLike] | dict[int, list[BoxLike]] | None = None,
) -> ImageClassificationDataset: ...


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
        return to_object_detection_dataset(images, labels, bboxes, None, None)
    else:
        labels = [0 for _ in range(length)]
        return to_image_classification_dataset(images, labels, None, None)


def _get_ic_dataset(
    images: list[np.ndarray] | tuple[int, int | None, int | None] | int | None,
    as_float: bool = False,
    override: list[BoxLike] | dict[int, list[BoxLike]] | None = None,
) -> ImageClassificationDataset:
    return _get_dataset(images, None, as_float, override)


def _get_od_dataset(
    images: list[np.ndarray] | tuple[int, int | None, int | None] | int | None,
    targets_per_image: int,
    as_float: bool = False,
    override: list[BoxLike] | dict[int, list[BoxLike]] | None = None,
) -> ObjectDetectionDataset:
    return _get_dataset(images, targets_per_image, as_float, override)


@pytest.fixture
def get_ic_dataset():
    return _get_ic_dataset


@pytest.fixture
def get_od_dataset():
    return _get_od_dataset


@pytest.fixture(scope="session")
def RNG() -> np.random.Generator:
    return np.random.default_rng(0)
