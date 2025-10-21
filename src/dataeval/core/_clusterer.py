from __future__ import annotations

__all__ = []

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.core._mst import compute_neighbor_distances, minimum_spanning_tree_edges
from dataeval.protocols import ArrayLike
from dataeval.types import ClusterData, CondensedTree
from dataeval.utils._array import flatten, to_numpy


def _find_outliers(clusters: NDArray[np.intp]) -> NDArray[np.intp]:
    """
    Retrieves Outliers based on when the sample was added to the cluster
    and how far it was from the cluster when it was added

    Returns
    -------
    NDArray[int]
        A numpy array of the outlier indices
    """
    return np.nonzero(clusters == -1)[0]


def _find_duplicates(
    mst: NDArray[np.float32], clusters: NDArray[np.intp]
) -> tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]:
    """
    Finds duplicate and near duplicate data based on cluster average distance

    Returns
    -------
    Tuple[List[List[int]], List[List[int]]]
        The exact :term:`duplicates<Duplicates>` and near duplicates as lists of related indices
    """
    # Delay load numba compiled functions
    from dataeval.core._numba import compare_links_to_cluster_std

    exact_indices, near_indices = compare_links_to_cluster_std(mst, clusters)  # type: ignore
    exact_dupes = _sorted_union_find(exact_indices)
    near_dupes = _sorted_union_find(near_indices)

    return [[int(ii) for ii in il] for il in exact_dupes], [[int(ii) for ii in il] for il in near_dupes]


def _sorted_union_find(index_groups: NDArray[np.int32]) -> list[list[np.int32]]:
    """Merges and sorts groups of indices that share any common index"""
    # Delay load fast_hdbscan functions
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        from fast_hdbscan.cluster_trees import (
            ds_find,
            ds_rank_create,
            ds_union_by_rank,
        )

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


def cluster(data: ArrayLike) -> ClusterData:
    """
    Uses hierarchical clustering on the flattened data and returns clustering
    information.

    Parameters
    ----------
    data : ArrayLike, shape - (N, ...)
        A dataset in an ArrayLike format. Function expects the data to have 2
        or more dimensions which will flatten to (N, P) where N number of
        observations in a P-dimensional space.

    Returns
    -------
    :class:`.ClusterData`

    Notes
    -----
    The cluster function works best when the length of the feature dimension,
    P, is less than 500. If flattening a CxHxW image results in a dimension
    larger than 500, then it is recommended to reduce the dimensions.

    Example
    -------
    >>> cluster(clusterer_images).clusters
    array([ 2,  0,  0,  0,  0,  0,  4,  0,  3,  1,  1,  0,  2,  0,  0,  0,  0,
            4,  2,  0,  0,  1,  2,  0,  1,  3,  0,  3,  3,  4,  0,  0,  3,  0,
            3, -1,  0,  0,  2,  4,  3,  4,  0,  1,  0, -1,  3,  0,  0,  0])
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        from fast_hdbscan.cluster_trees import (
            cluster_tree_from_condensed_tree,
            condense_tree,
            extract_eom_clusters,
            get_cluster_label_vector,
            get_point_membership_strength_vector,
            mst_to_linkage_tree,
        )

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
    kneighbors, kdistances = compute_neighbor_distances(x, max_neighbors)
    unsorted_mst: NDArray[np.float32] = minimum_spanning_tree_edges(x, kneighbors, kdistances)
    mst: NDArray[np.float32] = unsorted_mst[np.argsort(unsorted_mst.T[2])]
    linkage_tree: NDArray[np.float32] = mst_to_linkage_tree(mst).astype(np.float32)
    condensed_tree: CondensedTree = CondensedTree(*condense_tree(linkage_tree, min_cluster_size))

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
