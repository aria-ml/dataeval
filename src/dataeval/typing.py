"""
Common type protocols used for interoperability with DataEval.
"""

__all__ = [
    "AnnotatedDataset",
    "Array",
    "ArrayLike",
    "Dataset",
    "DatasetMetadata",
    "DeviceLike",
    "ImageClassificationDatum",
    "ImageClassificationDataset",
    "ObjectDetectionTarget",
    "ObjectDetectionDatum",
    "ObjectDetectionDataset",
    "SegmentationTarget",
    "SegmentationDatum",
    "SegmentationDataset",
    "Transform",
]


from collections.abc import Iterator, Mapping
from typing import (
    Any,
    Generic,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeVar,
    runtime_checkable,
)

import numpy.typing
import torch
from typing_extensions import NotRequired, ReadOnly, Required

ArrayLike: TypeAlias = numpy.typing.ArrayLike
"""
Type alias for a `Union` representing objects that can be coerced into an array.

See Also
--------
`NumPy ArrayLike <https://numpy.org/doc/stable/reference/typing.html#numpy.typing.ArrayLike>`_
"""


DeviceLike: TypeAlias = int | str | tuple[str, int] | torch.device
"""
Type alias for a `Union` representing types that specify a torch.device.

See Also
--------
`torch.device <https://pytorch.org/docs/stable/tensor_attributes.html#torch.device>`_
"""


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


_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)


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

    id: Required[ReadOnly[str]]
    index2label: NotRequired[ReadOnly[dict[int, str]]]


class ModelMetadata(TypedDict, total=False):
    """
    Model metadata required for all `AnnotatedModel` classes.

    Attributes
    ----------
    id : Required[str]
        A unique identifier for the model
    index2label : NotRequired[dict[int, str]]
        A lookup table converting label value to class name
    """

    id: Required[ReadOnly[str]]
    index2label: NotRequired[ReadOnly[dict[int, str]]]


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


ImageClassificationDatum: TypeAlias = tuple[ArrayLike, ArrayLike, Mapping[str, Any]]
"""
Type alias for an image classification datum tuple.

- :class:`ArrayLike` of shape (C, H, W) - Image data in channel, height, width format.
- :class:`ArrayLike` of shape (N,) - Class label as one-hot encoded ground-truth or prediction confidences.
- dict[str, Any] - Datum level metadata.
"""


ImageClassificationDataset: TypeAlias = AnnotatedDataset[ImageClassificationDatum]
"""
Type alias for an :class:`AnnotatedDataset` of :class:`ImageClassificationDatum` elements.
"""

# ========== OBJECT DETECTION DATASETS ==========


@runtime_checkable
class ObjectDetectionTarget(Protocol):
    """
    Protocol for targets in an Object Detection dataset.

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


ObjectDetectionDatum: TypeAlias = tuple[ArrayLike, ObjectDetectionTarget, Mapping[str, Any]]
"""
Type alias for an object detection datum tuple.

- :class:`ArrayLike` of shape (C, H, W) - Image data in channel, height, width format.
- :class:`ObjectDetectionTarget` - Object detection target information for the image.
- dict[str, Any] - Datum level metadata.
"""


ObjectDetectionDataset: TypeAlias = AnnotatedDataset[ObjectDetectionDatum]
"""
Type alias for an :class:`AnnotatedDataset` of :class:`ObjectDetectionDatum` elements.
"""


# ========== SEGMENTATION DATASETS ==========


@runtime_checkable
class SegmentationTarget(Protocol):
    """
    Protocol for targets in a Segmentation dataset.

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


SegmentationDatum: TypeAlias = tuple[ArrayLike, SegmentationTarget, Mapping[str, Any]]
"""
Type alias for an image classification datum tuple.

- :class:`ArrayLike` of shape (C, H, W) - Image data in channel, height, width format.
- :class:`SegmentationTarget` - Segmentation target information for the image.
- dict[str, Any] - Datum level metadata.
"""

SegmentationDataset: TypeAlias = AnnotatedDataset[SegmentationDatum]
"""
Type alias for an :class:`AnnotatedDataset` of :class:`SegmentationDatum` elements.
"""

# ========== MODEL ==========


@runtime_checkable
class AnnotatedModel(Protocol):
    """
    Protocol for an annotated model.
    """

    @property
    def metadata(self) -> ModelMetadata: ...


# ========== TRANSFORM ==========


@runtime_checkable
class Transform(Generic[_T], Protocol):
    """
    Protocol defining a transform function.

    Requires a `__call__` method that returns transformed data.

    Example
    -------
    >>> from typing import Any
    >>> from numpy.typing import NDArray

    >>> class MyTransform:
    ...     def __init__(self, divisor: float) -> None:
    ...         self.divisor = divisor
    ...
    ...     def __call__(self, data: NDArray[Any], /) -> NDArray[Any]:
    ...         return data / self.divisor

    >>> my_transform = MyTransform(divisor=255.0)
    >>> isinstance(my_transform, Transform)
    True
    >>> my_transform(np.array([1, 2, 3]))
    array([0.004, 0.008, 0.012])
    """

    def __call__(self, data: _T, /) -> _T: ...
