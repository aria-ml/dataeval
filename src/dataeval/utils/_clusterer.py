from __future__ import annotations

__all__ = []

import warnings
from dataclasses import dataclass
from typing import Any

import numba
import numpy as np
from numpy.typing import NDArray

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    from fast_hdbscan.cluster_trees import (
        CondensedTree,
        cluster_tree_from_condensed_tree,
        condense_tree,
        ds_find,
        ds_rank_create,
        ds_union_by_rank,
        extract_eom_clusters,
        get_cluster_label_vector,
        get_point_membership_strength_vector,
        mst_to_linkage_tree,
    )

from dataeval.typing import ArrayLike
from dataeval.utils._array import flatten, to_numpy
from dataeval.utils._fast_mst import calculate_neighbor_distances, minimum_spanning_tree


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


@dataclass
class ClusterData:
    clusters: NDArray[np.intp]
    mst: NDArray[np.float32]
    linkage_tree: NDArray[np.float32]
    condensed_tree: CondensedTree
    membership_strengths: NDArray[np.float32]
    k_neighbors: NDArray[np.int32]
    k_distances: NDArray[np.float32]


def cluster(data: ArrayLike) -> ClusterData:
    single_cluster = False
    cluster_selection_epsilon = 0.0
    # cluster_selection_method = "eom"

    x: NDArray[Any] = flatten(to_numpy(data))
    samples, features = x.shape  # Due to flatten(), we know shape has a length of 2
    if samples < 2:
        raise ValueError(f"Data should have at least 2 samples; got {samples}")
    if features < 1:
        raise ValueError(f"Samples should have at least 1 feature; got {features}")

    num_samples = len(x)
    min_num = int(num_samples * 0.05)
    min_cluster_size: int = min(max(5, min_num), 100)

    max_neighbors = min(25, num_samples - 1)
    kneighbors, kdistances = calculate_neighbor_distances(x, max_neighbors)
    unsorted_mst: NDArray[np.float32] = minimum_spanning_tree(x, kneighbors, kdistances)
    mst: NDArray[np.float32] = unsorted_mst[np.argsort(unsorted_mst.T[2])]
    linkage_tree: NDArray[np.float32] = mst_to_linkage_tree(mst).astype(np.float32)
    condensed_tree: CondensedTree = condense_tree(linkage_tree, min_cluster_size, None)

    cluster_tree = cluster_tree_from_condensed_tree(condensed_tree)

    selected_clusters = extract_eom_clusters(condensed_tree, cluster_tree, allow_single_cluster=single_cluster)

    # Uncomment if cluster_selection_method is made a parameter
    # if cluster_selection_method != "eom":
    #     selected_clusters = extract_leaves(condensed_tree, allow_single_cluster=single_cluster)

    # Uncomment if cluster_selection_epsilon is made a parameter
    # if len(selected_clusters) > 1 and cluster_selection_epsilon > 0.0:
    #     selected_clusters = cluster_epsilon_search(
    #         selected_clusters,
    #         cluster_tree,
    #         min_persistence=cluster_selection_epsilon,
    #     )

    clusters = get_cluster_label_vector(
        condensed_tree,
        selected_clusters,
        cluster_selection_epsilon,
        n_samples=x.shape[0],
    )

    membership_strengths = get_point_membership_strength_vector(condensed_tree, selected_clusters, clusters)

    return ClusterData(clusters, mst, linkage_tree, condensed_tree, membership_strengths, kneighbors, kdistances)


def sorted_union_find(index_groups: NDArray[np.int32]) -> list[list[np.int32]]:
    """Merges and sorts groups of indices that share any common index"""
    groups: list[list[np.int32]] = [[np.int32(x) for x in range(0)] for y in range(0)]
    uniques, inverse = np.unique(index_groups, return_inverse=True)
    inverse = inverse.flatten()
    disjoint_set = ds_rank_create(uniques.size)
    cluster_points = np.empty(uniques.size, dtype=np.uint32)
    for i in range(index_groups.shape[0]):
        point, nbr = np.int32(inverse[i * 2]), np.int32(inverse[i * 2 + 1])
        ds_union_by_rank(disjoint_set, point, nbr)
    for i in range(uniques.size):
        cluster_points[i] = ds_find(disjoint_set, i)
    for i in range(uniques.size):
        dups = np.nonzero(cluster_points == i)[0]
        if dups.size > 0:
            groups.append(uniques[dups].tolist())
    return sorted(groups)
