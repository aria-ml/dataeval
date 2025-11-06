from __future__ import annotations

__all__ = []

from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from dataeval.core._mst import compute_neighbor_distances, minimum_spanning_tree_edges
from dataeval.types import ArrayND
from dataeval.utils._array import flatten, to_numpy


class CondensedTree(TypedDict):
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


class ClusterResult(TypedDict):
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
        Indices of the nearest points in the population matrix
    k_distances : NDArray[np.float32]
        Array representing the lengths to points
    """

    clusters: NDArray[np.intp]
    mst: NDArray[np.float32]
    linkage_tree: NDArray[np.float32]
    condensed_tree: CondensedTree
    membership_strengths: NDArray[np.float32]
    k_neighbors: NDArray[np.int32]
    k_distances: NDArray[np.float32]


class ClusterStats(TypedDict):
    """
    Pre-calculated statistics for adaptive outlier detection.

    Attributes
    ----------
    cluster_ids : NDArray[np.intp]
        Array of unique cluster IDs (excluding -1)
    centers : NDArray[np.floating]
        Cluster centers, shape (n_clusters, n_features)
    cluster_distances_mean : NDArray[np.floating]
        Mean distance from points to their cluster center, shape (n_clusters,)
    cluster_distances_std : NDArray[np.floating]
        Standard deviation of distances within each cluster, shape (n_clusters,)
    distances : NDArray[np.floating]
        Distance from each point to its nearest cluster center, shape (n_samples,)
    nearest_cluster_idx : NDArray[np.intp]
        Index of nearest cluster center for each point, shape (n_samples,)
    """

    cluster_ids: NDArray[np.intp]
    centers: NDArray[np.floating]
    cluster_distances_mean: NDArray[np.floating]
    cluster_distances_std: NDArray[np.floating]
    distances: NDArray[np.floating]
    nearest_cluster_idx: NDArray[np.intp]


def compute_cluster_stats(
    embeddings: NDArray[np.floating],
    clusters: NDArray[np.intp],
) -> ClusterStats:
    """
    Compute cluster centers and distance statistics for adaptive outlier detection.

    Parameters
    ----------
    embeddings : NDArray[np.floating]
        The embedding vectors, shape (n_samples, n_features)
    clusters : NDArray[np.intp]
        Cluster labels from HDBSCAN (-1 for HDBSCAN outliers)

    Returns
    -------
    ClusterStats
        Pre-calculated statistics with empty arrays if no valid clusters found
    """
    # Get unique clusters (excluding -1)
    unique_clusters = np.unique(clusters[clusters >= 0])
    n_samples = len(embeddings)

    if len(unique_clusters) == 0:
        return ClusterStats(
            cluster_ids=np.array([], dtype=np.intp),
            centers=np.array([], dtype=embeddings.dtype),
            cluster_distances_mean=np.array([], dtype=embeddings.dtype),
            cluster_distances_std=np.array([], dtype=embeddings.dtype),
            distances=np.full(n_samples, np.inf, dtype=embeddings.dtype),
            nearest_cluster_idx=np.full(n_samples, -1, dtype=np.intp),
        )

    n_clusters = len(unique_clusters)
    n_features = embeddings.shape[1]

    centers = np.zeros((n_clusters, n_features), dtype=embeddings.dtype)
    cluster_distances_mean = np.zeros(n_clusters, dtype=embeddings.dtype)
    cluster_distances_std = np.zeros(n_clusters, dtype=embeddings.dtype)

    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = clusters == cluster_id
        cluster_points = embeddings[cluster_mask]
        cluster_center = cluster_points.mean(axis=0)

        # Calculate distances from center
        distances = np.linalg.norm(cluster_points - cluster_center, axis=1)

        centers[i] = cluster_center
        cluster_distances_mean[i] = distances.mean()
        cluster_distances_std[i] = distances.std()

    # Pre-calculate distance from each point to its nearest cluster center
    # Shape: (n_samples, n_clusters)
    all_distances = np.linalg.norm(embeddings[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    # Get minimum distance and nearest cluster index for each point
    nearest_cluster_idx = np.argmin(all_distances, axis=1)
    min_distances = all_distances[np.arange(n_samples), nearest_cluster_idx]

    return ClusterStats(
        cluster_ids=unique_clusters,
        centers=centers,
        cluster_distances_mean=cluster_distances_mean,
        cluster_distances_std=cluster_distances_std,
        distances=min_distances,
        nearest_cluster_idx=nearest_cluster_idx,
    )


def cluster(
    embeddings: ArrayND[float],
    n_expected_clusters: int | None = None,
) -> ClusterResult:
    """
    Uses hierarchical clustering on the flattened data and returns clustering
    information.

    Parameters
    ----------
    embeddings : ArrayND, shape - (N, ...)
        A dataset that can be a list, or array-like object. Function expects
        the data to have 2 or more dimensions which will flatten to (N, P) where N is
        the number of observations in a P-dimensional space.
    n_expected_clusters : int, optional
        Hint for the expected number of clusters (e.g., number of classes in dataset).
        If provided, adaptively adjusts min_cluster_size to encourage finding
        approximately this many clusters. Useful when you have domain knowledge
        about the data structure.

    Returns
    -------
    ClusterResult
        Mapping with keys:
        - clusters : NDArray[np.intp] - Assigned clusters
        - mst : NDArray[np.float32] - The minimum spanning tree of the data
        - linkage_tree : NDArray[np.float32] - The linkage array of the data
        - condensed_tree : CondensedTree(Mapping) - Derived from fast_hdbscan.cluster_trees.CondensedTree
        - membership_strengths : NDArray[np.float32] - The strength of the data point belonging to the assigned cluster
        - k_neighbors : NDArray[np.int32] - Indices of the nearest points in the population matrix
        - k_distances : NDArray[np.float32] - Array representing the lengths to points

    Notes
    -----
    The cluster function works best when the length of the feature dimension,
    P, is less than 500. If flattening a CxHxW image results in a dimension
    larger than 500, then it is recommended to reduce the dimensions.

    Examples
    --------
    >>> output = cluster(clusterer_images)
    >>> output["clusters"]
    array([ 2,  0,  0,  0,  0,  0,  4,  0,  3,  1,  1,  0,  2,  0,  0,  0,  0,
            4,  2,  0,  0,  1,  2,  0,  1,  3,  0,  3,  3,  4,  0,  0,  3,  0,
            3, -1,  0,  0,  2,  4,  3,  4,  0,  1,  0, -1,  3,  0,  0,  0])
    """
    # Import from our cached cluster_trees implementation
    from dataeval.core._fast_hdbscan._cluster_trees import (
        cluster_tree_from_condensed_tree,
        condense_tree,
        extract_eom_clusters,
        get_cluster_label_vector,
        get_point_membership_strength_vector,
        mst_to_linkage_tree,
    )

    single_cluster = True
    cluster_selection_epsilon = 0.0
    # cluster_selection_method = "eom"

    x: NDArray[Any] = flatten(to_numpy(embeddings))
    samples, features = x.shape  # Due to flatten(), we know shape has a length of 2
    if samples < 2:
        raise ValueError(f"Data should have at least 2 samples; got {samples}")
    if features < 1:
        raise ValueError(f"Samples should have at least 1 feature; got {features}")

    num_samples = len(x)

    # Adaptive min_cluster_size based on expected clusters hint
    if n_expected_clusters is not None:
        # Encourage finding approximately n_expected_clusters
        # Divide by 3 to allow smaller, more granular clusters
        min_cluster_size = max(5, num_samples // (n_expected_clusters * 3))
    else:
        # Default behavior: use 5% but cap at 100
        min_num = int(num_samples * 0.05)
        min_cluster_size = min(max(5, min_num), 100)

    max_neighbors = min(25, num_samples - 1)
    kneighbors, kdistances = compute_neighbor_distances(x, max_neighbors)
    unsorted_mst: NDArray[np.float32] = minimum_spanning_tree_edges(x, kneighbors, kdistances)
    mst: NDArray[np.float32] = unsorted_mst[np.argsort(unsorted_mst.T[2])]
    linkage_tree: NDArray[np.float32] = mst_to_linkage_tree(mst).astype(np.float32)
    condensed_tree = condense_tree(linkage_tree, min_cluster_size)
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

    clusters: NDArray[np.intp] = get_cluster_label_vector(
        condensed_tree,
        selected_clusters,
        cluster_selection_epsilon,
        n_samples=x.shape[0],
    )

    membership_strengths: NDArray[np.float32] = get_point_membership_strength_vector(
        condensed_tree,
        selected_clusters,
        clusters,
    )

    return ClusterResult(
        clusters=clusters,
        mst=mst,
        linkage_tree=linkage_tree,
        condensed_tree=CondensedTree(**condensed_tree._asdict()),
        membership_strengths=membership_strengths,
        k_neighbors=kneighbors,
        k_distances=kdistances,
    )
