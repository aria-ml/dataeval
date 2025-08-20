# Adapted from fast_hdbscan python module
# Original Authors: Leland McInnes <https://github.com/TutteInstitute/fast_hdbscan>
# Adapted for DataEval by Ryan Wood
# License: BSD 2-Clause

from __future__ import annotations

__all__ = []

import warnings
from typing import Any

import numba
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    from fast_hdbscan.disjoint_set import ds_find, ds_rank_create

from numpy.typing import NDArray


@numba.njit()
def _ds_union_by_rank(disjoint_set: tuple[NDArray[np.int32], NDArray[np.int32]], point: int, nbr: int) -> int:
    y = ds_find(disjoint_set, point)
    x = ds_find(disjoint_set, nbr)

    if x == y:
        return 0

    if disjoint_set[1][x] < disjoint_set[1][y]:
        x, y = y, x

    disjoint_set[0][y] = x
    if disjoint_set[1][x] == disjoint_set[1][y]:
        disjoint_set[1][x] += 1
    return 1


@numba.njit(locals={"i": numba.types.uint32, "nbr": numba.types.uint32, "dist": numba.types.float32})
def _init_tree(
    n_neighbors: NDArray[np.intp], n_distance: NDArray[np.float32]
) -> tuple[NDArray[np.float32], int, tuple[NDArray[np.int32], NDArray[np.int32]], NDArray[np.uint32]]:
    # Initial graph to hold tree connections
    tree = np.zeros((n_neighbors.size - 1, 3), dtype=np.float32)
    disjoint_set = ds_rank_create(n_neighbors.size)
    cluster_points = np.empty(n_neighbors.size, dtype=np.uint32)

    int_tree = 0
    for i in range(n_neighbors.size):
        nbr = n_neighbors[i]
        connect = _ds_union_by_rank(disjoint_set, i, nbr)
        if connect == 1:
            dist = n_distance[i]
            tree[int_tree] = (np.float32(i), np.float32(nbr), dist)
            int_tree += 1

    for i in range(cluster_points.size):
        cluster_points[i] = ds_find(disjoint_set, i)

    return tree, int_tree, disjoint_set, cluster_points


@numba.njit(locals={"i": numba.types.uint32, "nbr": numba.types.uint32})
def _update_tree_by_distance(
    tree: NDArray[np.float32],
    int_tree: int,
    disjoint_set: tuple[NDArray[np.int32], NDArray[np.int32]],
    n_neighbors: NDArray[np.uint32],
    n_distance: NDArray[np.float32],
) -> tuple[NDArray[np.float32], int, tuple[NDArray[np.int32], NDArray[np.int32]], NDArray[np.uint32]]:
    cluster_points = np.empty(n_neighbors.size, dtype=np.uint32)
    sort_dist = np.argsort(n_distance)
    dist_sorted = n_distance[sort_dist]
    nbrs_sorted = n_neighbors[sort_dist]
    points = np.arange(n_neighbors.size)
    point_sorted = points[sort_dist]

    for i in range(n_neighbors.size):
        point = point_sorted[i]
        nbr = nbrs_sorted[i]
        connect = _ds_union_by_rank(disjoint_set, point, nbr)
        if connect == 1:
            dist = dist_sorted[i]
            tree[int_tree] = (np.float32(point), np.float32(nbr), dist)
            int_tree += 1

    for i in range(cluster_points.size):
        cluster_points[i] = ds_find(disjoint_set, i)

    return tree, int_tree, disjoint_set, cluster_points


@numba.njit(locals={"i": numba.types.uint32})
def _cluster_edges(
    tracker: NDArray[Any], final_merge_idx: int, cluster_distances: NDArray[Any]
) -> list[NDArray[np.intp]]:
    cluster_ids = np.unique(tracker)
    edge_points: list[NDArray[np.intp]] = []
    for idx in range(cluster_ids.size):
        cluster_points = np.nonzero(tracker == cluster_ids[idx])[0]
        cluster_size = cluster_points.size
        cluster_mean = cluster_distances[: final_merge_idx + 1, cluster_points].mean()
        cluster_std = cluster_distances[: final_merge_idx + 1, cluster_points].std()
        threshold = cluster_mean + cluster_std
        points_mean = np.empty_like(cluster_points, dtype=np.float32)
        for i in range(cluster_size):
            points_mean[i] = cluster_distances[: final_merge_idx + 1, cluster_points[i]].mean()
        pts_to_add = cluster_points[np.nonzero(points_mean > threshold)[0]]
        threshold = int(cluster_size * 0.01) if np.floor(np.log10(cluster_size)) > 2 else int(cluster_size * 0.1)
        threshold = max(10, threshold)
        if pts_to_add.size > threshold:
            edge_points.append(pts_to_add)
        else:
            edge_points.append(cluster_points)
    return edge_points


@numba.njit(parallel=True, locals={"i": numba.types.int32})
def compare_links_to_cluster_std(
    mst: NDArray[np.float32], clusters: NDArray[np.intp]
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    cluster_ids = np.unique(clusters)
    cluster_grouping = np.full(mst.shape[0], -1, dtype=np.int16)

    for i in numba.prange(mst.shape[0]):
        cluster_id = clusters[np.int32(mst[i, 0])]
        if cluster_id == clusters[np.int32(mst[i, 1])]:
            cluster_grouping[i] = np.int16(cluster_id)

    overall_mean = mst.T[2].mean()
    order_mag = np.floor(np.log10(overall_mean)) if overall_mean > 0 else 0
    compare_mag = -3 if order_mag >= 0 else order_mag - 3

    exact_dup = np.full((mst.shape[0], 2), -1, dtype=np.int32)
    exact_dups_index = np.nonzero(mst[:, 2] < 10**compare_mag)[0]
    exact_dup[exact_dups_index] = mst[exact_dups_index, :2]

    near_dup = np.full((mst.shape[0], 2), -1, dtype=np.int32)
    for i in range(cluster_ids.size):
        cluster_links = np.nonzero(cluster_grouping == cluster_ids[i])[0]
        cluster_std = mst[cluster_links, 2].std()

        near_dups = np.nonzero(mst[cluster_links, 2] < cluster_std)[0]
        near_dups_index = cluster_links[near_dups]
        near_dup[near_dups_index] = mst[near_dups_index, :2]

    exact_idx = np.nonzero(exact_dup.T[0] != -1)[0]
    near_dup[exact_idx] = np.full((exact_idx.size, 2), -1, dtype=np.int32)
    near_idx = np.nonzero(near_dup.T[0] != -1)[0]

    return exact_dup[exact_idx], near_dup[near_idx]
