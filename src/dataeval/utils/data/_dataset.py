from __future__ import annotations

__all__ = []

from typing import Any, Generic, Iterable, Literal, Sequence, TypeVar

from dataeval.typing import (
    Array,
    ArrayLike,
    DatasetMetadata,
    ImageClassificationDataset,
    ObjectDetectionDataset,
)
from dataeval.utils._array import as_numpy


def _validate_data(
    datum_type: Literal["ic", "od"],
    images: Array | Sequence[Array],
    labels: Sequence[int] | Sequence[Sequence[int]],
    bboxes: Sequence[Sequence[Sequence[float]]] | None,
    metadata: Sequence[dict[str, Any]] | None,
) -> None:
    # Validate inputs
    dataset_len = len(images)

    if not isinstance(images, (Sequence, Array)) or len(images[0].shape) != 3:
        raise ValueError("Images must be a sequence or array of 3 dimensional arrays (H, W, C).")
    if len(labels) != dataset_len:
        raise ValueError(f"Number of labels ({len(labels)}) does not match number of images ({dataset_len}).")
    if bboxes is not None and len(bboxes) != dataset_len:
        raise ValueError(f"Number of bboxes ({len(bboxes)}) does not match number of images ({dataset_len}).")
    if metadata is not None and len(metadata) != dataset_len:
        raise ValueError(f"Number of metadata ({len(metadata)}) does not match number of images ({dataset_len}).")

    if datum_type == "ic":
        if not isinstance(labels, Sequence) or not isinstance(labels[0], int):
            raise TypeError("Labels must be a sequence of integers for image classification.")
    elif datum_type == "od":
        if not isinstance(labels, Sequence) or not isinstance(labels[0], Sequence) or not isinstance(labels[0][0], int):
            raise TypeError("Labels must be a sequence of sequences of integers for object detection.")
        if (
            bboxes is None
            or not isinstance(bboxes, (Sequence, Array))
            or not isinstance(bboxes[0], (Sequence, Array))
            or not isinstance(bboxes[0][0], (Sequence, Array))
            or not len(bboxes[0][0]) == 4
        ):
            raise TypeError("Boxes must be a sequence of sequences of (x0, y0, x1, y1) for object detection.")


def _find_max(arr: ArrayLike) -> Any:
    if isinstance(arr[0], (Iterable, Sequence, Array)):
        return max([_find_max(x) for x in arr])  # type: ignore
    else:
        return max(arr)


_TLabels = TypeVar("_TLabels", Sequence[int], Sequence[Sequence[int]])


class BaseAnnotatedDataset(Generic[_TLabels]):
    def __init__(
        self,
        datum_type: Literal["ic", "od"],
        images: Array | Sequence[Array],
        labels: _TLabels,
        metadata: Sequence[dict[str, Any]] | None,
        classes: Sequence[str] | None,
        name: str | None = None,
    ) -> None:
        self._classes = classes if classes is not None else [str(i) for i in range(_find_max(labels) + 1)]
        self._index2label = dict(enumerate(self._classes))
        self._images = images
        self._labels = labels
        self._metadata = metadata
        self._id = name or f"{len(self._images)}_image_{len(self._index2label)}_class_{datum_type}_dataset"

    @property
    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(id=self._id, index2label=self._index2label)

    def __len__(self) -> int:
        return len(self._images)


class CustomImageClassificationDataset(BaseAnnotatedDataset[Sequence[int]], ImageClassificationDataset):
    def __init__(
        self,
        images: Array | Sequence[Array],
        labels: Sequence[int],
        metadata: Sequence[dict[str, Any]] | None,
        classes: Sequence[str] | None,
        name: str | None = None,
    ) -> None:
        super().__init__("ic", images, labels, metadata, classes)
        if name is not None:
            self.__name__ = name
            self.__class__.__name__ = name
            self.__class__.__qualname__ = name

    def __getitem__(self, idx: int, /) -> tuple[Array, Array, dict[str, Any]]:
        one_hot = [0.0] * len(self._index2label)
        one_hot[self._labels[idx]] = 1.0
        return (
            self._images[idx],
            as_numpy(one_hot),
            self._metadata[idx] if self._metadata is not None else {},
        )


class CustomObjectDetectionDataset(BaseAnnotatedDataset[Sequence[Sequence[int]]], ObjectDetectionDataset):
    class ObjectDetectionTarget:
        def __init__(self, labels: Sequence[int], bboxes: Sequence[Sequence[float]]) -> None:
            self._labels = labels
            self._bboxes = bboxes
            self._scores = [1.0] * len(labels)

        @property
        def labels(self) -> Sequence[int]:
            return self._labels

        @property
        def boxes(self) -> Sequence[Sequence[float]]:
            return self._bboxes

        @property
        def scores(self) -> Sequence[float]:
            return self._scores

    def __init__(
        self,
        images: Array | Sequence[Array],
        labels: Sequence[Sequence[int]],
        bboxes: Sequence[Sequence[Sequence[float]]],
        metadata: Sequence[dict[str, Any]] | None,
        classes: Sequence[str] | None,
        name: str | None = None,
    ) -> None:
        super().__init__("od", images, labels, metadata, classes)
        if name is not None:
            self.__name__ = name
            self.__class__.__name__ = name
            self.__class__.__qualname__ = name
        self._bboxes = bboxes

    @property
    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(id=self._id, index2label=self._index2label)

    def __getitem__(self, idx: int, /) -> tuple[Array, ObjectDetectionTarget, dict[str, Any]]:
        return (
            self._images[idx],
            self.ObjectDetectionTarget(self._labels[idx], self._bboxes[idx]),
            self._metadata[idx] if self._metadata is not None else {},
        )


def to_image_classification_dataset(
    images: Array | Sequence[Array],
    labels: Sequence[int],
    metadata: Sequence[dict[str, Any]] | None,
    classes: Sequence[str] | None,
    name: str | None = None,
) -> ImageClassificationDataset:
    """
    Helper function to create custom ImageClassificationDataset classes.

    Parameters
    ----------
    images : Array | Sequence[Array]
        The images to use in the dataset.
    labels : Sequence[int]
        The labels to use in the dataset.
    metadata : Sequence[dict[str, Any]] | None
        The metadata to use in the dataset.
    classes : Sequence[str] | None
        The classes to use in the dataset.

    Returns
    -------
    ImageClassificationDataset
    """
    _validate_data("ic", images, labels, None, metadata)
    return CustomImageClassificationDataset(images, labels, metadata, classes, name)


def to_object_detection_dataset(
    images: Array | Sequence[Array],
    labels: Sequence[Sequence[int]],
    bboxes: Sequence[Sequence[Sequence[float]]],
    metadata: Sequence[dict[str, Any]] | None,
    classes: Sequence[str] | None,
    name: str | None = None,
) -> ObjectDetectionDataset:
    """
    Helper function to create custom ObjectDetectionDataset classes.

    Parameters
    ----------
    images : Array | Sequence[Array]
        The images to use in the dataset.
    labels : Sequence[Sequence[int]]
        The labels to use in the dataset.
    bboxes : Sequence[Sequence[Sequence[float]]]
        The bounding boxes (x0,y0,x1,y0) to use in the dataset.
    metadata : Sequence[dict[str, Any]] | None
        The metadata to use in the dataset.
    classes : Sequence[str] | None
        The classes to use in the dataset.

    Returns
    -------
    ObjectDetectionDataset
    """
    _validate_data("od", images, labels, bboxes, metadata)
    return CustomObjectDetectionDataset(images, labels, bboxes, metadata, classes, name)
