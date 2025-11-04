"""Data types used in DataEval."""

from __future__ import annotations

__all__ = [
    "Array1D",
    "Array2D",
    "Array3D",
    "Array4D",
    "Array5D",
    "Array6D",
    "Array7D",
    "Array8D",
    "Array9D",
    "ArrayND",
]

from collections.abc import Iterator
from typing import Any, NamedTuple, Protocol, TypeAlias, TypeVar, overload, runtime_checkable

from dataeval.protocols import Array

DType = TypeVar("DType", covariant=True)


@runtime_checkable
class SequenceLike(Protocol[DType]):
    """Protocol for sequence-like objects that can be indexed and iterated."""

    @overload
    def __getitem__(self, key: int, /) -> DType: ...
    @overload
    def __getitem__(self, key: Any, /) -> DType | SequenceLike[DType]: ...
    def __iter__(self) -> Iterator[DType]: ...
    def __len__(self) -> int: ...


Array1D: TypeAlias = Array | SequenceLike[DType]
Array2D: TypeAlias = Array | SequenceLike[Array1D[DType]]
Array3D: TypeAlias = Array | SequenceLike[Array2D[DType]]
Array4D: TypeAlias = Array | SequenceLike[Array3D[DType]]
Array5D: TypeAlias = Array | SequenceLike[Array4D[DType]]
Array6D: TypeAlias = Array | SequenceLike[Array5D[DType]]
Array7D: TypeAlias = Array | SequenceLike[Array6D[DType]]
Array8D: TypeAlias = Array | SequenceLike[Array7D[DType]]
Array9D: TypeAlias = Array | SequenceLike[Array8D[DType]]
ArrayND: TypeAlias = (
    Array
    | Array1D[DType]
    | Array2D[DType]
    | Array3D[DType]
    | Array4D[DType]
    | Array5D[DType]
    | Array6D[DType]
    | Array7D[DType]
    | Array8D[DType]
    | Array9D[DType]
)


class SourceIndex(NamedTuple):
    """
    The indices of the source image, box and channel.

    Attributes
    ----------
    image: int
        Index of the source image
    box : int | None
        Index of the box of the source image (if applicable)
    channel : int | None
        Index of the channel of the source image (if applicable)
    """

    image: int
    box: int | None
    channel: int | None
