"""
Common type hints used for interoperability with DataEval.
"""

__all__ = ["Array", "ArrayLike"]


import sys
from typing import Any, Generic, Iterator, Protocol, Sequence, TypedDict, TypeVar, Union, runtime_checkable

if sys.version_info >= (3, 11):
    from typing import NotRequired, Required
else:
    from typing_extensions import NotRequired, Required


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


T = TypeVar("T", covariant=True)
TArray = TypeVar("TArray", bound=Array)

ArrayLike = Union[Sequence[Any], Array]
"""
Type alias for array-like objects used for interoperability with DataEval.

This includes native Python sequences, as well as objects that conform to
the `Array` protocol.
"""


class DatasetMetadata(TypedDict):
    id: Required[str]
    index2label: NotRequired[dict[int, str]]


TDatasetMetadata = TypeVar("TDatasetMetadata", bound=DatasetMetadata)


@runtime_checkable
class Dataset(Generic[T], Protocol):
    def __getitem__(self, index: int, /) -> T: ...


@runtime_checkable
class SizedDataset(Dataset[T], Protocol):
    def __len__(self) -> int: ...


@runtime_checkable
class AnnotatedDataset(SizedDataset[T], Generic[T, TDatasetMetadata], Protocol):
    metadata: TDatasetMetadata


@runtime_checkable
class ObjectDetectionTarget(Generic[TArray], Protocol):
    boxes: TArray
    labels: TArray
    scores: TArray


@runtime_checkable
class SegmentationTarget(Generic[TArray], Protocol):
    mask: TArray
    labels: TArray
    scores: TArray


DatumMetadata = dict[str, Any]

ImageClassificationDatum = tuple[TArray, TArray, DatumMetadata]
ImageClassificationDataset = AnnotatedDataset[ImageClassificationDatum[TArray], TDatasetMetadata]

ObjectDetectionDatum = tuple[TArray, ObjectDetectionTarget[TArray], DatumMetadata]
ObjectDetectionDataset = AnnotatedDataset[ObjectDetectionDatum[TArray], TDatasetMetadata]

SegmentationDatum = tuple[TArray, SegmentationTarget[TArray], DatumMetadata]
SegmentationDataset = AnnotatedDataset[SegmentationDatum[TArray], TDatasetMetadata]
