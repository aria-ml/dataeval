# Adapted from fast_hdbscan python module
# Original Authors: Leland McInnes <https://github.com/TutteInstitute/fast_hdbscan>
# Adapted for DataEval by Ryan Wood
# License: BSD 2-Clause

__all__ = []

import warnings

import numba
import numpy as np
from sklearn.neighbors import NearestNeighbors

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    from fast_hdbscan.disjoint_set import ds_find, ds_rank_create


@numba.njit()
def _ds_union_by_rank(disjoint_set, point, nbr):
    y = ds_find(disjoint_set, point)
    x = ds_find(disjoint_set, nbr)

    if x == y:
        return 0

    if disjoint_set.rank[x] < disjoint_set.rank[y]:
        x, y = y, x

    disjoint_set.parent[y] = x
    if disjoint_set.rank[x] == disjoint_set.rank[y]:
        disjoint_set.rank[x] += 1
    return 1


@numba.njit(locals={"i": numba.types.uint32, "nbr": numba.types.uint32, "dist": numba.types.float32})
def _init_tree(n_neighbors, n_distance):
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
def _update_tree_by_distance(tree, int_tree, disjoint_set, n_neighbors, n_distance):
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
def _cluster_edges(tracker, last_idx, cluster_distances):
    cluster_ids = np.unique(tracker)
    edge_points = []
    for idx in range(cluster_ids.size):
        cluster_points = np.nonzero(tracker == cluster_ids[idx])[0]
        cluster_size = cluster_points.size
        cluster_mean = cluster_distances[: last_idx + 1, cluster_points].mean()
        cluster_std = cluster_distances[: last_idx + 1, cluster_points].std()
        threshold = cluster_mean + cluster_std
        points_mean = np.empty_like(cluster_points, dtype=np.float32)
        for i in range(cluster_size):
            points_mean[i] = cluster_distances[: last_idx + 1, cluster_points[i]].mean()
        pts_to_add = cluster_points[np.nonzero(points_mean > threshold)[0]]
        threshold = int(cluster_size * 0.01) if np.floor(np.log10(cluster_size)) > 2 else int(cluster_size * 0.1)
        threshold = max(10, threshold)
        if pts_to_add.size > threshold:
            edge_points.append(pts_to_add)
        else:
            edge_points.append(cluster_points)
    return edge_points


def _compute_nn(dataA, dataB, k):
    distances, neighbors = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(dataA).kneighbors(dataB)
    neighbors = np.array(neighbors[:, 1 : k + 1], dtype=np.int32)
    distances = np.array(distances[:, 1 : k + 1], dtype=np.float32)
    return neighbors, distances


def _calculate_cluster_neighbors(data, groups, point_array):
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


def minimum_spanning_tree(data, neighbors, distances):
    # Transpose arrays to get number of samples along a row
    k_neighbors = neighbors.T.astype(np.uint32).copy()
    k_distances = distances.T.astype(np.float32).copy()

    # Create cluster merging tracker
    merge_tracker = np.full((k_neighbors.shape[0] + 1, k_neighbors.shape[1]), -1, dtype=np.int32)

    # Initialize tree
    tree, int_tree, tree_disjoint_set, merge_tracker[0] = _init_tree(k_neighbors[0], k_distances[0])

    # Loop through all of the neighbors, updating the tree
    last_idx = 0
    for i in range(1, k_neighbors.shape[0]):
        tree, int_tree, tree_disjoint_set, merge_tracker[i] = _update_tree_by_distance(
            tree, int_tree, tree_disjoint_set, k_neighbors[i], k_distances[i]
        )
        last_idx = i
        if (merge_tracker[i] == merge_tracker[i - 1]).all():
            last_idx -= 1
            break

    # Identify final clusters
    cluster_ids = np.unique(merge_tracker[last_idx])
    if cluster_ids.size > 1:
        # Determining the edge points
        edge_points = _cluster_edges(merge_tracker[last_idx], last_idx, k_distances)

        # Run nearest neighbor again between clusters to reach single cluster
        additional_neighbors, additional_distances = _calculate_cluster_neighbors(
            data, edge_points, merge_tracker[last_idx]
        )

        # Update clusters
        last_idx += 1
        tree, int_tree, tree_disjoint_set, merge_tracker[last_idx] = _update_tree_by_distance(
            tree, int_tree, tree_disjoint_set, additional_neighbors, additional_distances
        )

    return tree


def calculate_neighbor_distances(data: np.ndarray, k: int = 10):
    # Have the potential to add in other distance calculations - supported calculations:
    # https://github.com/lmcinnes/pynndescent/blob/master/pynndescent/pynndescent_.py#L524
    try:
        from pynndescent import NNDescent

        max_descent = 30 if k <= 20 else k + 16
        index = NNDescent(
            data,
            metric="euclidean",
            n_neighbors=max_descent,
        )
        neighbors, distances = index.neighbor_graph
    except ImportError:
        distances, neighbors = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(data).kneighbors(data)

    neighbors = np.array(neighbors[:, 1 : k + 1], dtype=np.int32)
    distances = np.array(distances[:, 1 : k + 1], dtype=np.float32)
    return neighbors, distances
