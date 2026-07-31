"""Array shape aliases and array-valued mappings."""

__all__ = [
    "Array1D",
    "Array2D",
    "Array3D",
    "Array4D",
    "Array5D",
    "ArrayND",
    "StatsMap",
]

from collections.abc import Mapping
from typing import Any, TypeAlias, TypeVar

from numpy.typing import NDArray

from dataeval.protocols import Array, SequenceLike

_DType = TypeVar("_DType", covariant=True)


Array1D: TypeAlias = Array | SequenceLike[_DType]
Array2D: TypeAlias = Array | SequenceLike[Array1D[_DType]]
Array3D: TypeAlias = Array | SequenceLike[Array2D[_DType]]
Array4D: TypeAlias = Array | SequenceLike[Array3D[_DType]]
Array5D: TypeAlias = Array | SequenceLike[Array4D[_DType]]
ArrayND: TypeAlias = Array | Array1D[_DType] | Array2D[_DType] | Array3D[_DType] | Array4D[_DType] | Array5D[_DType]

StatsMap: TypeAlias = Mapping[str, NDArray[Any]]
"""
A mapping of metric names to their corresponding numpy array values.
Each array should have the same length along the first dimension, representing the number of samples.
"""
