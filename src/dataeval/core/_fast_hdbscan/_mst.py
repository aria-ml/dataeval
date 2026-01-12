"""Numba-accelerated functions for minimum spanning tree construction and duplicate detection.

This module provides performance-critical operations for building minimum spanning trees (MST)
from k-nearest neighbor graphs using Prim's algorithm with disjoint set data structures.
The functions are JIT-compiled with Numba for optimal performance.

Algorithm Overview
------------------
1. Build initial MST from first k-nearest neighbors using union-find
2. Iteratively add more k-nearest neighbors until full connectivity
3. If disconnected clusters remain, connect them via inter-cluster edges
4. Analyze MST edge distances within clusters to identify duplicates

Functions work together in the following pipeline:
    _init_tree -> _update_tree_by_distance -> _cluster_edges -> compare_links_to_cluster_std

Adapted from fast_hdbscan python module:
    https://github.com/TutteInstitute/fast_hdbscan
    Copyright (c) 2020, Leland McInnes
    License: BSD 2-Clause

Adapted for DataEval by Ryan Wood based on boruvka.py
"""

__all__ = []

import numba
import numpy as np
from numpy.typing import NDArray

from dataeval.core._fast_hdbscan._disjoint_set import ds_find, ds_union_by_rank

# Constants for cluster edge detection thresholds
CLUSTER_SIZE_LOG_THRESHOLD = 2  # Threshold for large vs small clusters (10^2 = 100 samples)
LARGE_CLUSTER_EDGE_RATIO = 0.01  # Edge points as fraction of cluster size for large clusters (1%)
SMALL_CLUSTER_EDGE_RATIO = 0.1  # Edge points as fraction of cluster size for small clusters (10%)
MIN_EDGE_POINTS = 10  # Minimum number of edge points to consider

# Constants for duplicate detection
EXACT_DUPLICATE_MAGNITUDE_OFFSET = -3  # Orders of magnitude below mean to consider exact duplicate
EXACT_DUPLICATE_FALLBACK_OFFSET = 3  # Additional offset when mean is very small


@numba.njit(cache=True)
def _expand_tree(
    additional_size: int,
    tree: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    Initialize minimum spanning tree from first set of k-nearest neighbors.

    This creates the initial MST structure by processing the closest neighbor for each point.
    Uses a disjoint set data structure to track connected components and avoid cycles.

    Parameters
    ----------
    n_neighbors : NDArray[np.int64]
        Array of neighbor indices, shape (n_samples,). For each sample i, n_neighbors[i]
        gives the index of its nearest neighbor.
    n_distance : NDArray[np.float32]
        Array of distances to nearest neighbors, shape (n_samples,). n_distance[i] is the
        distance from sample i to its nearest neighbor n_neighbors[i].

    Returns
    -------
    tree : NDArray[np.float32]
        MST edge array with shape (n_samples - 1, 3). Each row is [point_i, point_j, distance].
        Note: point indices are stored as float32 for array homogeneity.
    total_edge : int
        Number of edges currently in the tree (may be < n_samples - 1 if disconnected)
    disjoint_set : tuple[NDArray[np.int64], NDArray[np.int64]]
        Disjoint set data structure tracking connected components
    cluster_points : NDArray[np.int64]
        Cluster assignment for each point (root of its set in disjoint_set)

    Notes
    -----
    The tree array stores node indices as float32 rather than integers for homogeneity
    with distance values. Callers should cast back to int when extracting node indices.
    """
    new_tree = np.full((tree.shape[0] + additional_size - 1, 3), -1, dtype=np.float32)
    new_tree[: tree.shape[0]] = tree
    return new_tree


@numba.njit(locals={"i": numba.types.intp, "nbr": numba.types.intp, "dist": numba.types.float32}, cache=True)
def _update_tree(
    tree: NDArray[np.float32],
    total_edge: int,
    disjoint_set: tuple[NDArray[np.int64], NDArray[np.int64]],
    cluster_points: NDArray[np.int64],
    n_neighbors: NDArray[np.int64],
    n_distance: NDArray[np.float32],
    points: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.float32], int, tuple[NDArray[np.int64], NDArray[np.int64]], NDArray[np.int64]]:
    """
    Initialize minimum spanning tree from first set of k-nearest neighbors.

    This creates the initial MST structure by processing the closest neighbor for each point.
    Uses a disjoint set data structure to track connected components and avoid cycles.

    Parameters
    ----------
    n_neighbors : NDArray[np.int64]
        Array of neighbor indices, shape (n_samples,). For each sample i, n_neighbors[i]
        gives the index of its nearest neighbor.
    n_distance : NDArray[np.float32]
        Array of distances to nearest neighbors, shape (n_samples,). n_distance[i] is the
        distance from sample i to its nearest neighbor n_neighbors[i].

    Returns
    -------
    tree : NDArray[np.float32]
        MST edge array with shape (n_samples - 1, 3). Each row is [point_i, point_j, distance].
        Note: point indices are stored as float32 for array homogeneity.
    total_edge : int
        Number of edges currently in the tree (may be < n_samples - 1 if disconnected)
    disjoint_set : tuple[NDArray[np.int64], NDArray[np.int64]]
        Disjoint set data structure tracking connected components
    cluster_points : NDArray[np.int64]
        Cluster assignment for each point (root of its set in disjoint_set)

    Notes
    -----
    The tree array stores node indices as float32 rather than integers for homogeneity
    with distance values. Callers should cast back to int when extracting node indices.
    """
    if points is None:
        points = np.arange(n_neighbors.size, dtype=np.int64)

    # Expand variables to contain all new points if necessary
    if tree.shape[0] - total_edge < points.max():
        tree = _expand_tree(points.max(), tree)

    for i in range(points.size):
        point = points[i]
        nbr = n_neighbors[i]
        if ds_union_by_rank(disjoint_set, point, np.int64(nbr)) and nbr >= 0:
            dist = n_distance[i]
            # Store edge as (point_i, neighbor, distance)
            # Note: Storing indices as float32 for array homogeneity with distances
            tree[total_edge] = (np.float32(point), np.float32(nbr), dist)
            total_edge += 1

    # Determine cluster membership for each point
    for i in range(cluster_points.size):
        cluster_points[i] = ds_find(disjoint_set, np.int64(i))

    return tree, total_edge, disjoint_set, cluster_points


@numba.njit(locals={"i": numba.types.uint32}, cache=True)
def _cluster_edges(
    tracker: NDArray[np.int64], final_merge_idx: int, cluster_distances: NDArray[np.float32]
) -> list[NDArray[np.int64]]:
    """
    Identify edge points in each cluster based on distance statistics.

    Edge points are cluster members that are far from the cluster center, making them
    good candidates for connecting to other clusters. The threshold for "far" is
    cluster_mean + cluster_std. The number of edge points returned adapts based on
    cluster size.

    Parameters
    ----------
    tracker : NDArray[np.int64]
        Cluster assignment for each point, shape (n_samples,)
    final_merge_idx : int
        Index into cluster_distances indicating the final merge level to consider
    cluster_distances : NDArray[np.float32]
        Distance matrix with shape (n_merge_levels, n_samples) tracking distances
        at each level of the merge hierarchy

    Returns
    -------
    edge_points : list[NDArray[np.int64]]
        List of arrays, one per cluster. Each array contains indices of edge points
        for that cluster. If too few edge points are found, returns all cluster points.

    Notes
    -----
    For large clusters (>100 samples), uses 1% of cluster size as threshold.
    For small clusters (<=100 samples), uses 10% of cluster size.
    Always returns at least MIN_EDGE_POINTS (10) edge points if possible.
    """
    cluster_ids = np.unique(tracker)
    edge_points: list[NDArray[np.int64]] = []

    for idx in range(cluster_ids.size):
        cluster_points = np.nonzero(tracker == cluster_ids[idx])[0]
        cluster_size = cluster_points.size

        # Calculate distance statistics for this cluster
        cluster_mean = cluster_distances[: final_merge_idx + 1, cluster_points].mean()
        cluster_std = cluster_distances[: final_merge_idx + 1, cluster_points].std()
        threshold = cluster_mean + cluster_std

        # Find points with mean distance > threshold (potential edge points)
        points_mean = np.empty_like(cluster_points, dtype=np.float32)
        for i in range(cluster_size):
            points_mean[i] = cluster_distances[: final_merge_idx + 1, cluster_points[i]].mean()
        pts_to_add = cluster_points[np.nonzero(points_mean > threshold)[0]]

        # Determine minimum edge points based on cluster size
        # Large clusters (>100): use 1% of size, small clusters: use 10%
        if np.floor(np.log10(cluster_size)) > CLUSTER_SIZE_LOG_THRESHOLD:
            min_threshold = int(cluster_size * LARGE_CLUSTER_EDGE_RATIO)
        else:
            min_threshold = int(cluster_size * SMALL_CLUSTER_EDGE_RATIO)
        min_threshold = max(MIN_EDGE_POINTS, min_threshold)

        # Return edge points if we have enough, otherwise return all cluster points
        if pts_to_add.size > min_threshold:
            edge_points.append(pts_to_add)
        else:
            edge_points.append(cluster_points)

    return edge_points


@numba.njit(cache=True)
def _flatten_and_sort(
    neighbors: NDArray[np.int64], distances: NDArray[np.float32]
) -> tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.int64]]:
    """
    Flattens and sorts both the neighbors and distances arrays based on the sorted distance array.

    Due to the flattening/rearranging of the neighbors array, a row index array is created
    and returned to ensure that the correct neighbor-point pair is maintained.
    Array is created by numpy.arange(neighbors.shape[0]).repeat(neighbors.shape[1]).

    Parameters
    ----------
    neighbors : NDArray[np.int64]
        Array of neighbor indices, shape (n_samples, k neighbors). For each sample i, neighbors[i]
        gives the index of its nearest neighbor.
    distances : NDArray[np.float32]
        Array of distances to nearest neighbors, shape (n_samples, k neighbors). distances[i] is the
        distance from sample i to its nearest neighbor neighbors[i].

    Returns
    -------
    flattened_sorted_neighbors : NDArray[int64]
        Flattened neighbors array sorted by shortest distance
    flattened_sorted_distances : NDArray[float32]
        Flattened distances array sorted by shortest distance
    flattened_sorted_indicies : NDArray[int64]
        Flattened row index array sorted by shortest distance
    """
    flat_nbrs = neighbors.flatten()
    flat_dist = distances.flatten()
    sort_dist = np.argsort(flat_dist)
    dist_sorted = flat_dist[sort_dist]
    nbrs_sorted = flat_nbrs[sort_dist]
    indices = np.arange(neighbors.shape[0]).repeat(neighbors.shape[1])
    flat_index = indices.flatten()
    index_sorted = flat_index[sort_dist]
    return nbrs_sorted, dist_sorted, index_sorted


@numba.njit(locals={"i": numba.types.int32}, cache=True)
def compare_links_to_cluster_std(
    mst: NDArray[np.float32], clusters: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Identify exact and near duplicate pairs based on MST edge distances and cluster statistics.

    Analyzes edges in the minimum spanning tree to find duplicates. Exact duplicates are
    identified by edges with distances orders of magnitude smaller than the mean. Near
    duplicates are edges within each cluster that fall below the cluster's standard deviation.

    Parameters
    ----------
    mst : NDArray[np.float32]
        Minimum spanning tree edges with shape (n_samples - 1, 3).
        Each row is [point_i, point_j, distance].
    clusters : NDArray[np.int64]
        Cluster assignment for each point, shape (n_samples,)

    Returns
    -------
    exact_duplicates : NDArray[np.int64]
        Array of exact duplicate pairs with shape (n_exact, 2).
        Each row is [point_i, point_j] indices.
    near_duplicates : NDArray[np.int64]
        Array of near duplicate pairs with shape (n_near, 2).
        Each row is [point_i, point_j] indices. Excludes pairs already in exact_duplicates.

    Notes
    -----
    Performance: This function performs BETTER without parallel=True. Benchmarks show
    serial execution is 6-25x faster due to small workload size and parallel overhead.

    Exact duplicate threshold is calculated as:
    - If mean distance >= 1: threshold = 10^-3 (3 orders of magnitude below mean)
    - If mean distance < 1: threshold = mean * 10^(order_of_magnitude - 3)

    Near duplicates are found per-cluster by comparing edge distances to cluster std dev.
    """
    cluster_ids = np.unique(clusters)
    cluster_grouping = np.full(mst.shape[0], -1, dtype=np.int16)

    # Identify which edges connect points within the same cluster
    # Note: Using regular range instead of prange - serial is faster for this workload
    for i in range(mst.shape[0]):
        cluster_id = clusters[np.int64(mst[i, 0])]
        if cluster_id == clusters[np.int64(mst[i, 1])]:
            cluster_grouping[i] = np.int16(cluster_id)

    # Calculate threshold for exact duplicates based on mean distance
    overall_mean = mst.T[2].mean()
    order_mag = np.floor(np.log10(overall_mean)) if overall_mean > 0 else 0

    # Threshold is 3 orders of magnitude below mean (or offset by 3 if mean < 1)
    compare_mag = EXACT_DUPLICATE_MAGNITUDE_OFFSET if order_mag >= 0 else order_mag - EXACT_DUPLICATE_FALLBACK_OFFSET

    # Find exact duplicates: edges with very small distances
    exact_dup = np.full((mst.shape[0], 2), -1, dtype=np.int64)
    exact_dups_index = np.nonzero(mst[:, 2] < 10**compare_mag)[0]
    exact_dup[exact_dups_index] = mst[exact_dups_index, :2]

    # Find near duplicates: per-cluster edges below cluster std dev
    near_dup = np.full((mst.shape[0], 2), -1, dtype=np.int64)
    for i in range(cluster_ids.size):
        cluster_links = np.nonzero(cluster_grouping == cluster_ids[i])[0]
        cluster_std = mst[cluster_links, 2].std()

        # Edges in this cluster with distance < std dev are near duplicates
        near_dups = np.nonzero(mst[cluster_links, 2] < cluster_std)[0]
        near_dups_index = cluster_links[near_dups]
        near_dup[near_dups_index] = mst[near_dups_index, :2]

    # Remove exact duplicates from near duplicates list
    exact_idx = np.nonzero(exact_dup.T[0] != -1)[0]
    near_dup[exact_idx] = np.full((exact_idx.size, 2), -1, dtype=np.int64)
    near_idx = np.nonzero(near_dup.T[0] != -1)[0]

    return exact_dup[exact_idx], near_dup[near_idx]
