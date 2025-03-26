"""
Common type hints used for interoperability with DataEval.
"""

__all__ = [
    "Array",
    "ArrayLike",
    "Dataset",
    "AnnotatedDataset",
    "DatasetMetadata",
    "ImageClassificationDatum",
    "ImageClassificationDataset",
    "ObjectDetectionTarget",
    "ObjectDetectionDatum",
    "ObjectDetectionDataset",
    "SegmentationTarget",
    "SegmentationDatum",
    "SegmentationDataset",
]


import sys
from typing import Any, Generic, Iterator, Mapping, Protocol, Sequence, TypedDict, TypeVar, Union, runtime_checkable

from typing_extensions import NotRequired, Required

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias


@runtime_checkable
class Array(Protocol):
    """
    Protocol for array objects providing interoperability with DataEval.

    Supports common array representations with popular libraries like
    PyTorch, Tensorflow and JAX, as well as NumPy arrays.

    Example
    -------
    >>> import numpy as np
    >>> import torch
    >>> from dataeval.typing import Array

    Create array objects

    >>> ndarray = np.random.random((10, 10))
    >>> tensor = torch.tensor([1, 2, 3])

    Check type at runtime

    >>> isinstance(ndarray, Array)
    True

    >>> isinstance(tensor, Array)
    True
    """

    @property
    def shape(self) -> tuple[int, ...]: ...
    def __array__(self) -> Any: ...
    def __getitem__(self, key: Any, /) -> Any: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __len__(self) -> int: ...


_T_co = TypeVar("_T_co", covariant=True)
_ScalarType = Union[int, float, bool, str]
ArrayLike: TypeAlias = Union[Sequence[_ScalarType], Sequence[Sequence[_ScalarType]], Sequence[Array], Array]
"""
Type alias for array-like objects used for interoperability with DataEval.

This includes native Python sequences, as well as objects that conform to
the :class:`Array` protocol.
"""


class DatasetMetadata(TypedDict, total=False):
    """
    Dataset level metadata required for all `AnnotatedDataset` classes.

    Attributes
    ----------
    id : Required[str]
        A unique identifier for the dataset
    index2label : NotRequired[dict[int, str]]
        A lookup table converting label value to class name
    """

    id: Required[str]
    index2label: NotRequired[dict[int, str]]


@runtime_checkable
class Dataset(Generic[_T_co], Protocol):
    """
    Protocol for a generic `Dataset`.

    Methods
    -------
    __getitem__(index: int)
        Returns datum at specified index.
    __len__()
        Returns dataset length.
    """

    def __getitem__(self, index: int, /) -> _T_co: ...
    def __len__(self) -> int: ...


@runtime_checkable
class AnnotatedDataset(Dataset[_T_co], Generic[_T_co], Protocol):
    """
    Protocol for a generic `AnnotatedDataset`.

    Attributes
    ----------
    metadata : :class:`.DatasetMetadata` or derivatives.

    Methods
    -------
    __getitem__(index: int)
        Returns datum at specified index.
    __len__()
        Returns dataset length.

    Notes
    -----
    Inherits from :class:`.Dataset`.
    """

    @property
    def metadata(self) -> DatasetMetadata: ...


# ========== IMAGE CLASSIFICATION DATASETS ==========


ImageClassificationDatum: TypeAlias = tuple[Array, Array, Mapping[str, Any]]
"""
A type definition for an image classification datum tuple.

- :class:`Array` of shape (C, H, W) - Image data in channel, height, width format.
- :class:`Array` of shape (N,) - Class label as one-hot encoded ground-truth or prediction confidences.
- dict[str, Any] - Datum level metadata.
"""


ImageClassificationDataset: TypeAlias = AnnotatedDataset[ImageClassificationDatum]
"""
A type definition for an :class:`AnnotatedDataset` of :class:`ImageClassificationDatum` elements.
"""

# ========== OBJECT DETECTION DATASETS ==========


@runtime_checkable
class ObjectDetectionTarget(Protocol):
    """
    A protocol for targets in an Object Detection dataset.

    Attributes
    ----------
    boxes : :class:`ArrayLike` of shape (N, 4)
    labels : :class:`ArrayLike` of shape (N,)
    scores : :class:`ArrayLike` of shape (N, M)
    """

    @property
    def boxes(self) -> ArrayLike: ...

    @property
    def labels(self) -> ArrayLike: ...

    @property
    def scores(self) -> ArrayLike: ...


ObjectDetectionDatum: TypeAlias = tuple[Array, ObjectDetectionTarget, Mapping[str, Any]]
"""
A type definition for an object detection datum tuple.

- :class:`Array` of shape (C, H, W) - Image data in channel, height, width format.
- :class:`ObjectDetectionTarget` - Object detection target information for the image.
- dict[str, Any] - Datum level metadata.
"""


ObjectDetectionDataset: TypeAlias = AnnotatedDataset[ObjectDetectionDatum]
"""
A type definition for an :class:`AnnotatedDataset` of :class:`ObjectDetectionDatum` elements.
"""


# ========== SEGMENTATION DATASETS ==========


@runtime_checkable
class SegmentationTarget(Protocol):
    """
    A protocol for targets in a Segmentation dataset.

    Attributes
    ----------
    mask : :class:`ArrayLike`
    labels : :class:`ArrayLike`
    scores : :class:`ArrayLike`
    """

    @property
    def mask(self) -> ArrayLike: ...

    @property
    def labels(self) -> ArrayLike: ...

    @property
    def scores(self) -> ArrayLike: ...


SegmentationDatum: TypeAlias = tuple[Array, SegmentationTarget, Mapping[str, Any]]
"""
A type definition for an image classification datum tuple.

- :class:`Array` of shape (C, H, W) - Image data in channel, height, width format.
- :class:`SegmentationTarget` - Segmentation target information for the image.
- dict[str, Any] - Datum level metadata.
"""

SegmentationDataset: TypeAlias = AnnotatedDataset[SegmentationDatum]
"""
A type definition for an :class:`AnnotatedDataset` of :class:`SegmentationDatum` elements.
"""
