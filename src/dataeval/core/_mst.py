# Adapted from fast_hdbscan python module
# Original Authors: Leland McInnes <https://github.com/TutteInstitute/fast_hdbscan>
# Adapted for DataEval by Ryan Wood
# License: BSD 2-Clause

from __future__ import annotations

__all__ = []

import warnings
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from dataeval.config import get_max_processes
from dataeval.utils._array import flatten


@overload
def _compute_nearest_neighbors(
    data_fit: NDArray[Any],
    data_query: NDArray[Any] | None,
    k: int,
    *,
    algorithm: Literal["auto", "ball_tree", "brute", "kd_tree"] = "auto",
    exclude_self: bool = True,
    return_distances: Literal[True],
) -> tuple[NDArray[np.int32], NDArray[np.float32]]: ...


@overload
def _compute_nearest_neighbors(
    data_fit: NDArray[Any],
    data_query: NDArray[Any] | None,
    k: int,
    *,
    algorithm: Literal["auto", "ball_tree", "brute", "kd_tree"] = "auto",
    exclude_self: bool = True,
    return_distances: Literal[False],
) -> NDArray[np.int32]: ...


def _compute_nearest_neighbors(
    data_fit: NDArray[Any],
    data_query: NDArray[Any] | None,
    k: int,
    *,
    algorithm: Literal["auto", "ball_tree", "brute", "kd_tree"] = "auto",
    exclude_self: bool = True,
    return_distances: bool = True,
) -> tuple[NDArray[np.int32], NDArray[np.float32]] | NDArray[np.int32]:
    """
    Core nearest neighbors computation function.

    Parameters
    ----------
    data_fit : NDArray
        Data to fit the nearest neighbors model with shape (n_samples_fit, n_features)
    data_query : NDArray or None
        Data to query for neighbors with shape (n_samples_query, n_features).
        If None, uses data_fit (self-query)
    k : int
        Number of neighbors to find (excluding self if exclude_self=True)
    algorithm : {"auto", "ball_tree", "brute", "kd_tree"}, default="auto"
        Algorithm to use for nearest neighbor search
    exclude_self : bool, default=True
        If True, exclude the point itself from neighbors (when querying fitted data)
    return_distances : bool, default=True
        If True, return both neighbors and distances; otherwise only neighbors

    Returns
    -------
    neighbors : NDArray[np.int32]
        Indices of k nearest neighbors with shape (n_samples_query, k)
    distances : NDArray[np.float32], optional
        Distances to k nearest neighbors with shape (n_samples_query, k).
        Only returned if return_distances=True
    """
    if data_query is None:
        data_query = data_fit

    # Compute n_neighbors accounting for self-exclusion
    n_neighbors = k + 1 if exclude_self else k
    n_neighbors = min(n_neighbors, data_fit.shape[0])

    # Fit and query
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, n_jobs=get_max_processes())
    nbrs.fit(data_fit)
    distances, neighbors = nbrs.kneighbors(data_query)

    # Slice to exclude self if needed
    if exclude_self:
        start_idx = 1
        end_idx = k + 1
    else:
        start_idx = 0
        end_idx = k

    neighbors = np.array(neighbors[:, start_idx:end_idx], dtype=np.int32)

    if return_distances:
        distances = np.array(distances[:, start_idx:end_idx], dtype=np.float32)
        return neighbors, distances
    return neighbors


def _compute_cluster_neighbors(
    data: NDArray[Any], cluster_groups: list[NDArray[np.intp]], cluster_labels: NDArray[Any]
) -> tuple[NDArray[np.uint32], NDArray[np.float32]]:
    """
    Rerun nearest neighbor computation based on clusters.

    For each cluster, finds the second-nearest neighbor in other clusters
    for each point in the cluster.

    Parameters
    ----------
    data : NDArray
        The full dataset with shape (n_samples, n_features)
    cluster_groups : list[NDArray[np.intp]]
        List of arrays, each containing indices of points in a cluster
    cluster_labels : NDArray
        Array tracking cluster assignments for each point

    Returns
    -------
    cluster_neighbors : NDArray[np.uint32]
        Index of nearest inter-cluster neighbor for each point
    cluster_distances : NDArray[np.float32]
        Distance to nearest inter-cluster neighbor for each point
    """
    cluster_neighbors = np.zeros(cluster_labels.size, dtype=np.uint32)
    cluster_distances = np.full(cluster_labels.size, np.inf, dtype=np.float32)

    n_clusters = len(cluster_groups)
    for i in range(n_clusters):
        # Get current cluster points
        current_cluster_idx = cluster_groups[i]
        current_cluster_data = data[current_cluster_idx]

        # Get all other cluster points
        other_cluster_groups = [cluster_groups[j] for j in range(n_clusters) if j != i]
        if not other_cluster_groups:
            continue

        other_cluster_idx = np.concatenate(other_cluster_groups)
        other_cluster_data = data[other_cluster_idx]

        # Find 2-nearest neighbors and use the second one
        neighbors, distances = _compute_nearest_neighbors(
            other_cluster_data,
            current_cluster_data,
            2,
            algorithm="brute",
            exclude_self=True,
            return_distances=True,
        )

        cluster_neighbors[current_cluster_idx] = other_cluster_idx[neighbors[:, 1]]
        cluster_distances[current_cluster_idx] = distances[:, 1]

    return cluster_neighbors, cluster_distances


def minimum_spanning_tree_edges(
    data: NDArray[Any], neighbors: NDArray[np.int32], distances: NDArray[np.float32]
) -> NDArray[np.float32]:
    """
    Compute minimum spanning tree edges from k-nearest neighbor graph.

    Uses Prim's algorithm with a disjoint set data structure to efficiently build
    a minimum spanning tree from the k-nearest neighbor graph. If the k-NN graph
    is not fully connected, attempts to connect clusters using inter-cluster edges.

    Parameters
    ----------
    data : NDArray
        The full dataset with shape (n_samples, n_features)
    neighbors : NDArray[np.int32]
        K-nearest neighbor indices for each point, shape (n_samples, k)
    distances : NDArray[np.float32]
        Distances to k-nearest neighbors, shape (n_samples, k)

    Returns
    -------
    tree : NDArray[np.float32]
        Minimum spanning tree edges as array of shape (n_samples - 1, 3).
        Each row contains [point_i, point_j, distance] representing an edge.

    Warnings
    --------
    RuntimeWarning
        If k-nearest neighbors are exhausted before achieving full connectivity.
        Falls back to inter-cluster connection heuristics in this case.

    Notes
    -----
    The algorithm proceeds in two phases:
    1. Build MST from k-NN graph using sorted edge distances
    2. If disconnected clusters remain, connect them using cluster edge points

    See Also
    --------
    minimum_spanning_tree : Higher-level interface that also computes k-NN
    """
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


def minimum_spanning_tree(data: NDArray[Any], k: int = 15) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """
    Compute the minimum spanning tree of a dataset.

    This is a high-level interface that computes k-nearest neighbors and then
    constructs the minimum spanning tree from the resulting graph.

    Parameters
    ----------
    data : NDArray
        Input data with shape (n_samples, n_features) or can be flattened
    k : int, default=15
        Number of nearest neighbors to use for building the k-NN graph.
        Higher values increase connectivity but add computational cost.
        Should be large enough to ensure graph connectivity.

    Returns
    -------
    rows : NDArray[np.intp]
        Source node indices for each edge in the MST with shape (n_samples - 1,)
    cols : NDArray[np.intp]
        Target node indices for each edge in the MST with shape (n_samples - 1,)

    Notes
    -----
    The MST is represented as two arrays (rows, cols) defining edges.
    Together they form n_samples - 1 edges connecting all points.

    Examples
    --------
    >>> import numpy as np
    >>> from dataeval.core._mst import minimum_spanning_tree
    >>> data = np.random.rand(100, 10)
    >>> rows, cols = minimum_spanning_tree(data, k=15)
    >>> len(rows)  # Should be n_samples - 1
    99

    See Also
    --------
    minimum_spanning_tree_edges : Lower-level function that returns edge weights
    compute_neighbor_distances : Computes the k-NN graph
    """
    data = flatten(data)

    # Get k-nearest neighbors and build MST
    neighbors, distances = compute_neighbor_distances(data, k=k)
    mst_edges = minimum_spanning_tree_edges(data, neighbors, distances)

    rows = mst_edges[:, 0].astype(np.intp)
    cols = mst_edges[:, 1].astype(np.intp)

    return rows, cols


def compute_neighbor_distances(data: NDArray[Any], k: int = 10) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """
    Compute k nearest neighbors for each point in data (self-query, excluding self).

    Parameters
    ----------
    data : NDArray
        Input data array with shape (n_samples, n_features)
    k : int, default=10
        Number of neighbors to find (excluding self)

    Returns
    -------
    neighbors : NDArray[np.int32]
        Indices of k nearest neighbors for each point with shape (n_samples, k)
    distances : NDArray[np.float32]
        Distances to k nearest neighbors for each point with shape (n_samples, k)

    See Also
    --------
    compute_neighbors : For querying neighbors between two different datasets
    """
    return _compute_nearest_neighbors(data, None, k, algorithm="brute", exclude_self=True, return_distances=True)


def compute_neighbors(
    data_fit: NDArray[Any],
    data_query: NDArray[Any],
    k: int = 1,
    algorithm: Literal["auto", "ball_tree", "kd_tree"] = "auto",
) -> NDArray[Any]:
    """
    For each sample in data_query, compute the k nearest neighbors in data_fit.

    Parameters
    ----------
    data_fit : NDArray
        Reference points to search with shape (n_samples_fit, n_features).
        This is the dataset that will be indexed for neighbor search.
    data_query : NDArray
        Query points with shape (n_samples_query, n_features).
        For each of these points, find k nearest neighbors in data_fit.
    k : int, default=1
        The number of neighbors to find
    algorithm : {"auto", "ball_tree", "kd_tree"}, default="auto"
        Tree method for nearest neighbor computation

    Returns
    -------
    NDArray
        Indices of k nearest neighbors in data_fit for each point in data_query.
        Shape is (n_samples_query,) if k=1, otherwise (n_samples_query, k)

    Raises
    ------
    ValueError
        If k < 1 or if algorithm is not "auto", "ball_tree", or "kd_tree"

    Notes
    -----
    Do not use kd_tree if n_features > 20

    Examples
    --------
    >>> import numpy as np
    >>> from dataeval.core import compute_neighbors
    >>> reference_data = np.random.rand(100, 5)  # 100 reference points
    >>> query_data = np.random.rand(10, 5)  # 10 query points
    >>> neighbors = compute_neighbors(reference_data, query_data, k=3)
    >>> neighbors.shape
    (10, 3)

    See Also
    --------
    sklearn.neighbors.NearestNeighbors : Similar sklearn interface
    compute_neighbor_distances : For self-query (single dataset)
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if algorithm not in ["auto", "ball_tree", "kd_tree"]:
        raise ValueError("Algorithm must be 'auto', 'ball_tree', or 'kd_tree'")

    data_fit = flatten(data_fit)
    data_query = flatten(data_query)

    # Note: exclude_self=True handles the case where data_query and data_fit may overlap
    # but we want neighbors from data_fit, not self-matches
    neighbors = _compute_nearest_neighbors(
        data_fit, data_query, k, algorithm=algorithm, exclude_self=True, return_distances=False
    )
    return neighbors.squeeze()
