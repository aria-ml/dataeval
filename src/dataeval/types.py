"""Data structures used in DataEval."""

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
    "ClusterData",
    "CondensedTree",
]

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import Array

DType = TypeVar("DType", covariant=True)


class SequenceLike(Protocol[DType]):
    """Protocol for sequence-like objects that can be indexed and iterated."""

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


class CondensedTree(NamedTuple):
    """
    Derived from fast_hdbscan.cluster_trees.CondensedTree

    Attributes
    ----------
    parent : NDArray[np.int64]
    child : NDArray[np.int64]
    lambda_val : NDArray[np.float32]
    child_size : NDArray[np.float32]
    """

    parent: NDArray[np.int64]
    child: NDArray[np.int64]
    lambda_val: NDArray[np.float32]
    child_size: NDArray[np.float32]


@dataclass
class ClusterData:
    """
    Cluster output data structure.

    Attributes
    ----------
    clusters : NDArray[np.intp]
        Assigned clusters
    mst : NDArray[np.float32]
        The minimum spanning tree of the data
    linkage_tree : NDArray[np.float32]
        The linkage array of the data
    condensed_tree : CondensedTree
        The condensed tree of the data
    membership_strengths : NDArray[np.float32]
        The strength of the data point belonging to the assigned cluster
    k_neighbors : NDArray[np.int32]
        Indices of the nearest points in the population matrix.
    k_distances : NDArray[np.float32]
        Array representing the lengths to points.
    """

    clusters: NDArray[np.intp]
    mst: NDArray[np.float32]
    linkage_tree: NDArray[np.float32]
    condensed_tree: CondensedTree
    membership_strengths: NDArray[np.float32]
    k_neighbors: NDArray[np.int32]
    k_distances: NDArray[np.float32]
