from __future__ import annotations

__all__ = []

from collections.abc import Iterator
from itertools import tee
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval.protocols import Array, Dataset, ObjectDetectionTarget
from dataeval.utils._array import as_numpy
from dataeval.utils._boundingbox import BoundingBox

T = TypeVar("T")


class SizedIterator(Generic[T]):
    def __init__(self, iterator: Iterator[T], length: int) -> None:
        self._iterator = iterator
        self._length = length

    def __iter__(self) -> Iterator[T]:
        return self._iterator

    def __next__(self) -> T:
        return next(self._iterator)

    def __len__(self) -> int:
        return self._length


def unzip_dataset(
    dataset: Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]], per_target: bool
) -> tuple[Iterator[NDArray[Any]], Iterator[list[BoundingBox] | None] | None]:
    """
    Unzips a dataset into separate generators for images and targets.

    This preserves performance by only loading each item from the dataset once.

    Parameters
    ----------
    dataset : Dataset
        The dataset to unzip, which may contain images and optional targets.
    per_target : bool
        If True, extract bounding box targets from the dataset.

    Returns
    -------
    tuple[Iterator[NDArray[Any]], Iterator[list[BoundingBox] | None] | None]
        Two iterators, one for images and one for targets.
    """

    def _generate_pairs() -> Iterator[tuple[NDArray[Any], list[BoundingBox] | None]]:
        for i in range(len(dataset)):
            d = dataset[i]
            image = np.asarray(d[0] if isinstance(d, tuple) else d)
            if per_target and isinstance(d, tuple) and isinstance(d[1], ObjectDetectionTarget):
                try:
                    boxes = d[1].boxes if isinstance(d[1].boxes, Array) else as_numpy(d[1].boxes)
                    target = [BoundingBox(box[0], box[1], box[2], box[3], image_shape=image.shape) for box in boxes]
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid bounding box format for image {i}: {d[1].boxes}")
            else:
                target = None
            yield image, target

    # Create two independent iterators from the generator
    iter1, iter2 = tee(_generate_pairs(), 2)

    # Extract images and targets separately
    images_iter = SizedIterator((pair[0] for pair in iter1), len(dataset))
    targets_iter = (pair[1] for pair in iter2) if per_target else None

    return images_iter, targets_iter
