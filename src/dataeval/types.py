"""Data structures used in DataEval."""

__all__ = [
    "ClusterData",
    "CondensedTree",
]

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


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
