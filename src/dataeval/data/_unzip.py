"""Unzip a dataset into separate image and target iterators."""

__all__ = []

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import Array, Dataset, ObjectDetectionTarget
from dataeval.utils._internal import as_numpy, unwrap_image
from dataeval.utils._validate import validate_dataset
from dataeval.utils.preprocessing import BoundingBox


class _SizedIterator:
    def __init__(self, iterator: Iterator[Any], length: int) -> None:
        self._iterator = iterator
        self._length = length

    def __iter__(self) -> Iterator[Any]:
        return self._iterator

    def __next__(self) -> Any:
        return next(self._iterator)

    def __len__(self) -> int:
        return self._length


def unzip_dataset(
    dataset: Dataset[Any] | Dataset[tuple[Any, Any, Any]],
    per_target: bool,
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
    from itertools import tee

    # Per-target extraction reads d[1].boxes, so the dataset must be an OD-shaped MAITE
    # dataset. Image-only callers can still pass a bare image dataset.
    validate_dataset(
        dataset,
        expected="object_detection" if per_target else "image_only",
        caller="unzip_dataset",
    )

    def _generate_pairs() -> Iterator[tuple[NDArray[Any], list[BoundingBox] | None]]:
        for i in range(len(dataset)):
            d = dataset[i]
            image = np.asarray(unwrap_image(d))
            if per_target and isinstance(d, tuple) and isinstance(d[1], ObjectDetectionTarget):
                try:
                    boxes = d[1].boxes if isinstance(d[1].boxes, Array) else as_numpy(d[1].boxes)
                    target = [BoundingBox(box[0], box[1], box[2], box[3], image_shape=image.shape) for box in boxes]
                except (ValueError, IndexError) as err:
                    raise ValueError(f"Invalid bounding box format for image {i}: {d[1].boxes}") from err
            else:
                target = None
            yield image, target

    # Create two independent iterators from the generator
    iter1, iter2 = tee(_generate_pairs(), 2)

    # Extract images and targets separately
    images_iter = _SizedIterator((pair[0] for pair in iter1), len(dataset))
    targets_iter = (pair[1] for pair in iter2) if per_target else None

    return images_iter, targets_iter
