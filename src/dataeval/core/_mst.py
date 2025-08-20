# Adapted from fast_hdbscan python module
# Original Authors: Leland McInnes <https://github.com/TutteInstitute/fast_hdbscan>
# Adapted for DataEval by Ryan Wood
# License: BSD 2-Clause

from __future__ import annotations

__all__ = []

import warnings
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from dataeval.config import get_max_processes
from dataeval.utils._array import flatten


def _compute_nn(dataA: NDArray[Any], dataB: NDArray[Any], k: int) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    distances, neighbors = (
        NearestNeighbors(n_neighbors=k + 1, algorithm="brute", n_jobs=get_max_processes()).fit(dataA).kneighbors(dataB)
    )
    neighbors = np.array(neighbors[:, 1 : k + 1], dtype=np.int32)
    distances = np.array(distances[:, 1 : k + 1], dtype=np.float32)
    return neighbors, distances


def _compute_cluster_neighbors(
    data: NDArray[Any], groups: list[NDArray[np.intp]], point_array: NDArray[Any]
) -> tuple[NDArray[np.uint32], NDArray[np.float32]]:
    """Rerun nearest neighbor based on clusters"""
    cluster_neighbors = np.zeros(point_array.size, dtype=np.uint32)
    cluster_nbr_distances = np.full(point_array.size, np.inf, dtype=np.float32)

    for i in range(len(groups)):
        selectionA = groups[i]
        groupA = data[selectionA]
        selectionB = np.concatenate([arr for j, arr in enumerate(groups) if j != i])
        groupB = data[selectionB]
        new_neighbors, new_distances = _compute_nn(groupB, groupA, 2)
        cluster_neighbors[selectionA] = selectionB[new_neighbors[:, 1]]
        cluster_nbr_distances[selectionA] = new_distances[:, 1]

    return cluster_neighbors, cluster_nbr_distances


def minimum_spanning_tree_edges(
    data: NDArray[Any], neighbors: NDArray[np.int32], distances: NDArray[np.float32]
) -> NDArray[np.float32]:
    # Delay load numba compiled functions
    from dataeval.core._numba import _cluster_edges, _init_tree, _update_tree_by_distance

    # Transpose arrays to get number of samples along a row
    k_neighbors = neighbors.T.astype(np.uint32).copy()
    k_distances = distances.T.astype(np.float32).copy()

    # Create cluster merging tracker
    merge_tracker = np.full((k_neighbors.shape[0] + 1, k_neighbors.shape[1]), -1, dtype=np.int32)

    # Initialize tree
    tree, int_tree, tree_disjoint_set, merge_tracker[0] = _init_tree(k_neighbors[0], k_distances[0])

    # Loop through all of the neighbors, updating the tree
    k_max = k_neighbors.shape[0]
    k_now = 0  # to catch k_neighbors.shape[0] == 1 edge case
    for k_now in range(1, k_max):
        tree, int_tree, tree_disjoint_set, merge_tracker[k_now] = _update_tree_by_distance(
            tree, int_tree, tree_disjoint_set, k_neighbors[k_now], k_distances[k_now]
        )

        time_to_stop = len(np.unique(merge_tracker[k_now])) == 1

        if time_to_stop:
            break
    else:
        # Exhausted k-nearest neighbors without achieving connectivity
        warnings.warn(
            f"Exhausted k-nearest neighbors (k={k_neighbors.shape[0] - 1}) "
            f"before finding connected spanning tree. "
            f"Consider increasing k parameter. "
            f"Falling back to inter-cluster connection heuristics.",
            RuntimeWarning,
        )
    final_merge_idx = k_now

    # Identify final clusters
    cluster_ids = np.unique(merge_tracker[final_merge_idx])
    if cluster_ids.size > 1:
        # Determining the edge points
        edge_points = _cluster_edges(merge_tracker[final_merge_idx], final_merge_idx, k_distances)

        # Run nearest neighbor again between clusters to reach single cluster
        additional_neighbors, additional_distances = _compute_cluster_neighbors(
            data, edge_points, merge_tracker[final_merge_idx]
        )

        # Update clusters
        next_merge_idx = final_merge_idx + 1
        tree, int_tree, tree_disjoint_set, merge_tracker[next_merge_idx] = _update_tree_by_distance(
            tree, int_tree, tree_disjoint_set, additional_neighbors, additional_distances
        )

    return tree


def minimum_spanning_tree(X: NDArray[Any], k: int = 15) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    X = flatten(X)

    # Get k-nearest neighbors and build MST
    neighbors, distances = compute_neighbor_distances(X, k=k)
    mst_edges = minimum_spanning_tree_edges(X, neighbors, distances)

    rows = mst_edges[:, 0].astype(np.intp)
    cols = mst_edges[:, 1].astype(np.intp)

    return rows, cols


def compute_neighbor_distances(data: np.ndarray, k: int = 10) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    # Note that k is the number of neighbors sought, excluding self. However, NearestNeighbors includes self.
    # That is why the n_neighbors keyword is defined the way it is.
    distances, neighbors = (
        NearestNeighbors(n_neighbors=min(k + 1, data.shape[0]), algorithm="brute", n_jobs=get_max_processes())
        .fit(data)
        .kneighbors(data)
    )

    neighbors = np.array(neighbors[:, 1 : k + 1], dtype=np.int32)
    distances = np.array(distances[:, 1 : k + 1], dtype=np.float32)
    return neighbors, distances


def compute_neighbors(
    A: NDArray[Any],
    B: NDArray[Any],
    k: int = 1,
    algorithm: Literal["auto", "ball_tree", "kd_tree"] = "auto",
) -> NDArray[Any]:
    """
    For each sample in A, compute the nearest neighbor in B

    Parameters
    ----------
    A, B : NDArray
        The n_samples and n_features respectively
    k : int
        The number of neighbors to find
    algorithm : Literal
        Tree method for nearest neighbor (auto, ball_tree or kd_tree)

    Note
    ----
        Do not use kd_tree if n_features > 20

    Returns
    -------
    List:
        Closest points to each point in A and B

    Raises
    ------
    ValueError
        If algorithm is not "auto", "ball_tree", or "kd_tree"

    See Also
    --------
    sklearn.neighbors.NearestNeighbors
    """

    if k < 1:
        raise ValueError("k must be >= 1")
    if algorithm not in ["auto", "ball_tree", "kd_tree"]:
        raise ValueError("Algorithm must be 'auto', 'ball_tree', or 'kd_tree'")

    A = flatten(A)
    B = flatten(B)

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm=algorithm, n_jobs=get_max_processes()).fit(B)
    nns = nbrs.kneighbors(A)[1]
    return nns[:, 1:].squeeze()
