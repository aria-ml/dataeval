"""Unzip a dataset into separate image and target iterators."""

__all__ = []

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.exceptions import MaiteShapeError
from dataeval.protocols import Array, Dataset, MultiobjectTrackingTarget, ObjectDetectionTarget, _is_protocol_instance
from dataeval.utils._annotations import Annotations, build_annotations
from dataeval.utils._array import as_numpy
from dataeval.utils.data import unwrap_image, validate_dataset
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


def _reject_tracking(dataset: Dataset[Any], per_target: bool) -> None:
    """Refuse a tracking dataset, which an image-only read would otherwise pass.

    An image-only read accepts any 3-tuple, so a tracking dataset gets through with its video
    streams as the images, and fails in the first reader that treats one as an array. A read of
    targets is already refused by validation, which names the target it expected.
    """
    if per_target or len(dataset) == 0:
        return
    datum = dataset[0]
    if isinstance(datum, tuple) and len(datum) == 3 and _is_protocol_instance(datum[1], MultiobjectTrackingTarget):
        raise MaiteShapeError(
            "unzip_dataset: argument 'dataset' is a multi-object tracking dataset, whose images are "
            "video streams rather than arrays. Present it as frames first, with SequenceFrames(dataset)."
        )


def _dataset_labels(labels: Any, item: int) -> NDArray[Any]:
    """
    Read one image's labels from a dataset, taking a floating array as class ids.

    A dataset supplies its own labels, so its caller cannot cast them. A one-dimensional floating array is read as
    follows: a finite whole number is that class id, and a nonfinite entry, which is how a float array says "unlabeled",
    is -1, the missing label. A finite entry that is not a whole number within the ``int64`` range is not a class id
    and raises. Any other array is returned as it is, so a wrong length or dimension is still refused by
    ``build_annotations``, which also keeps an explicit ``labels=`` strict.
    """
    array = as_numpy(labels)
    if array.ndim != 1 or array.dtype.kind != "f":
        return array
    finite = np.isfinite(array)
    bad = finite & ((array != np.round(array)) | (np.abs(array) >= 2.0**63))
    if bad.any():
        box = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"image {item}, box {box}: class ids must be whole numbers within the int64 range, got {array[box]!r}; "
            "a nonfinite label is read as missing."
        )
    return np.where(finite, array, -1).astype(np.int64)


def _unzip_annotations(
    dataset: Dataset[Any] | Dataset[tuple[Any, Any, Any]],
    per_target: bool,
    with_labels: bool = False,
) -> tuple[Iterator[NDArray[Any]], Iterator[Annotations | None] | None]:
    """
    Unzip a dataset into image and annotation iterators, keeping each box's coordinates as given.

    The shared body of :func:`unzip_dataset`, which returns only the converted boxes. This one returns the whole
    ``Annotations`` for each image, with its raw coordinates and, when `with_labels`,
    its labels.

    Parameters
    ----------
    dataset : Dataset
        The dataset to unzip.
    per_target : bool
        If True, extract each image's annotations.
    with_labels : bool, default False
        If True, also read each target's labels, a floating array as class ids (see ``_dataset_labels``). Left
        False, the labels are never read.

    Returns
    -------
    tuple[Iterator[NDArray[Any]], Iterator[Annotations | None] | None]
        Two iterators, one of images and one of annotations, which is None when `per_target` is False.
    """
    from itertools import tee

    # Per-target extraction reads d[1].boxes, so the dataset must be an OD-shaped MAITE
    # dataset. Image-only callers can still pass a bare image dataset.
    validate_dataset(
        dataset,
        expected="object_detection" if per_target else "image_only",
        caller="unzip_dataset",
    )

    _reject_tracking(dataset, per_target)

    def _generate_pairs() -> Iterator[tuple[NDArray[Any], Annotations | None]]:
        for i in range(len(dataset)):
            d = dataset[i]
            image = np.asarray(unwrap_image(d))
            if per_target and isinstance(d, tuple) and isinstance(d[1], ObjectDetectionTarget):
                boxes = d[1].boxes if isinstance(d[1].boxes, Array) else as_numpy(d[1].boxes)
                labels = _dataset_labels(d[1].labels, i) if with_labels else None
                annotations = build_annotations(boxes, image.shape, i, labels)
            else:
                annotations = None
            yield image, annotations

    # Create two independent iterators from the generator
    iter1, iter2 = tee(_generate_pairs(), 2)

    # Extract images and annotations separately
    images_iter = _SizedIterator((pair[0] for pair in iter1), len(dataset))
    annotations_iter = (pair[1] for pair in iter2) if per_target else None

    return images_iter, annotations_iter


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
    images, annotations = _unzip_annotations(dataset, per_target)
    targets = None if annotations is None else (a.boxes if a is not None else None for a in annotations)
    return images, targets
