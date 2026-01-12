# Adapted from fast_hdbscan python module
# Original Authors: Leland McInnes <https://github.com/TutteInstitute/fast_hdbscan>
# Adapted for DataEval by Ryan Wood
# License: BSD 2-Clause


__all__ = []

import logging
from typing import Any, Literal, TypedDict, overload

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from dataeval.config import get_max_processes
from dataeval.types import ArrayND
from dataeval.utils.arrays import as_numpy, flatten_samples

_logger = logging.getLogger(__name__)


class MSTResult(TypedDict):
    """
    Type definition for minimum spanning tree output.

    Attributes
    ----------
    source : NDArray[np.int64]
        Source node indices for each edge in the MST
    target : NDArray[np.int64]
        Target node indices for each edge in the MST
    """

    source: NDArray[np.int64]
    target: NDArray[np.int64]


@overload
def _compute_nearest_neighbors(
    data_fit: NDArray[Any],
    data_query: NDArray[Any] | None,
    k: int,
    *,
    algorithm: Literal["auto", "ball_tree", "brute", "kd_tree"] = "auto",
    return_distances: Literal[True],
) -> tuple[NDArray[np.int64], NDArray[np.float32]]: ...


@overload
def _compute_nearest_neighbors(
    data_fit: NDArray[Any],
    data_query: NDArray[Any] | None,
    k: int,
    *,
    algorithm: Literal["auto", "ball_tree", "brute", "kd_tree"] = "auto",
    return_distances: Literal[False],
) -> NDArray[np.int64]: ...


def _compute_nearest_neighbors(
    data_fit: NDArray[Any],
    data_query: NDArray[Any] | None,
    k: int,
    *,
    algorithm: Literal["auto", "ball_tree", "brute", "kd_tree"] = "auto",
    return_distances: bool = True,
) -> tuple[NDArray[np.int64], NDArray[np.float32]] | NDArray[np.int64]:
    """
    Core nearest neighbors computation function.

    Parameters
    ----------
    data_fit : NDArray
        Data to fit the nearest neighbors model with shape (n_samples_fit, n_features)
    data_query : NDArray or None
        Data to query for neighbors with shape (n_samples_query, n_features).
        If None, performs self-query
    k : int
        Number of neighbors to find (algorithm excludes self)
    algorithm : {"auto", "ball_tree", "brute", "kd_tree"}, default="auto"
        Algorithm to use for nearest neighbor search
    return_distances : bool, default=True
        If True, return both neighbors and distances; otherwise only neighbors

    Returns
    -------
    neighbors : NDArray[np.int64]
        Indices of k nearest neighbors with shape (n_samples_query, k)
    distances : NDArray[np.float32], optional
        Distances to k nearest neighbors with shape (n_samples_query, k).
        Only returned if return_distances=True
    """
    # Compute n_neighbors
    n_neighbors = min(k, data_fit.shape[0] - 1)

    # Fit and query
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, n_jobs=get_max_processes())
    nbrs.fit(data_fit)
    distances, neighbors = nbrs.kneighbors(data_query, return_distance=True)

    if return_distances:
        return neighbors, distances
    return neighbors


def _compute_cluster_neighbors(
    data: NDArray[Any], cluster_labels: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """
    Rerun nearest neighbor computation based on clusters.

    For each cluster, finds the second-nearest neighbor in other clusters
    for each point in the cluster.

    Parameters
    ----------
    data : NDArray
        The full dataset with shape (n_samples, n_features)
    cluster_groups : list[NDArray[np.int64]]
        List of arrays, each containing indices of points in a cluster
    cluster_labels : NDArray
        Array tracking cluster assignments for each point

    Returns
    -------
    cluster_neighbors : NDArray[np.int64]
        Index of nearest inter-cluster neighbor for each point
    cluster_distances : NDArray[np.float32]
        Distance to nearest inter-cluster neighbor for each point
    """
    n_clusters = np.unique(cluster_labels).tolist()
    cluster_neighbors = np.full((len(n_clusters), cluster_labels.size), -1, dtype=np.int64)
    cluster_distances = np.full((len(n_clusters), cluster_labels.size), np.inf, dtype=np.float32)

    for i, lbl in enumerate(n_clusters):
        # Get current cluster points
        current_cluster_idx = np.nonzero(cluster_labels == lbl)[0]
        current_cluster_data = data[current_cluster_idx]

        # Get all other cluster points
        other_cluster_groups = [np.nonzero(cluster_labels == lbl2)[0] for j, lbl2 in enumerate(n_clusters) if j > i]
        if not other_cluster_groups:
            continue

        other_cluster_idx = np.concatenate(other_cluster_groups)
        other_cluster_data = data[other_cluster_idx]

        # Find nearest neighbors
        neighbors, distances = _compute_nearest_neighbors(
            other_cluster_data,
            current_cluster_data,
            1,
            algorithm="brute",
            return_distances=True,
        )

        cluster_neighbors[i, current_cluster_idx] = other_cluster_idx[neighbors.squeeze()]
        cluster_distances[i, current_cluster_idx] = distances.squeeze()

    return cluster_neighbors.T, cluster_distances.T


def minimum_spanning_tree_edges(
    data: NDArray[Any], neighbors: NDArray[np.int64], distances: NDArray[np.float32]
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
    neighbors : NDArray[np.int64]
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
    from dataeval.core._fast_hdbscan._disjoint_set import ds_rank_create
    from dataeval.core._fast_hdbscan._mst import _flatten_and_sort, _update_tree

    # Flatten arrays and sort them by distance
    nbrs_sorted, dist_sorted, point_sorted = _flatten_and_sort(neighbors, distances)

    # Initialize tree, disjoint_set and cluster merging tracker
    size = np.int64(neighbors.shape[0])
    tree = np.zeros((size - 1, 3), dtype=np.float32)
    tree_disjoint_set = ds_rank_create(np.int64(size))
    total_edge = 0  # total_edge tracks current number of edges contained in the tree
    merge_tracker = np.full((neighbors.shape[1] + 1, neighbors.shape[0]), -1, dtype=np.int64)

    # Update tree
    merge_idx = 0
    tree, total_edge, tree_disjoint_set, merge_tracker[merge_idx] = _update_tree(
        tree, total_edge, tree_disjoint_set, merge_tracker[merge_idx], nbrs_sorted, dist_sorted, point_sorted
    )

    # Identify clusters
    cluster_ids = np.unique(merge_tracker[merge_idx])
    while cluster_ids.size > 1:
        # Exhausted k-nearest neighbors without achieving connectivity
        _logger.debug(
            f"Exhausted k-nearest neighbors (k={neighbors.shape[1]}) "
            f"before finding connected spanning tree. "
            f"Computing cluster nearest neighbors.",
        )

        # Run nearest neighbor again between clusters to reach single cluster
        additional_neighbors, additional_distances = _compute_cluster_neighbors(data, merge_tracker[merge_idx])

        # Flatten arrays and sort them by distance
        nbrs_sorted, dist_sorted, point_sorted = _flatten_and_sort(additional_neighbors, additional_distances)

        # Update clusters
        merge_idx += 1
        tree, total_edge, tree_disjoint_set, merge_tracker[merge_idx] = _update_tree(
            tree, total_edge, tree_disjoint_set, merge_tracker[merge_idx], nbrs_sorted, dist_sorted, point_sorted
        )

        cluster_ids = np.unique(merge_tracker[merge_idx])

    tree_idx = np.nonzero(tree[:, 0] >= 0)[0]
    return tree[tree_idx]


def minimum_spanning_tree(embeddings: ArrayND[float], k: int = 15) -> MSTResult:
    """
    Compute the minimum spanning tree of a dataset.

    This is a high-level interface that computes k-nearest neighbors and then
    constructs the minimum spanning tree from the resulting graph.

    Parameters
    ----------
    embeddings : Array2D[float]
        Input data with shape (n_samples, n_features). Can be a 2D list, array-like
        object, or tensor that will be flattened if necessary.
    k : int, default=15
        Number of nearest neighbors to use for building the k-NN graph.
        Higher values increase connectivity but add computational cost.
        Should be large enough to ensure graph connectivity.

    Returns
    -------
    MSTResult
        Mapping with keys:
        - source : NDArray[np.int64] - Source node indices for each edge in the MST with shape (n_samples - 1,)
        - target : NDArray[np.int64] - Target node indices for each edge in the MST with shape (n_samples - 1,)

    Notes
    -----
    The MST is represented as two arrays (source, target) defining edges.
    Together they form n_samples - 1 edges connecting all points.

    Examples
    --------
    >>> import numpy as np
    >>> from dataeval.core import minimum_spanning_tree
    >>> data = np.random.rand(100, 10)

    >>> mst = minimum_spanning_tree(data, k=15)
    >>> len(mst["source"])  # Should be n_samples - 1
    99

    See Also
    --------
    minimum_spanning_tree_edges : Lower-level function that returns edge weights
    compute_neighbor_distances : Computes the k-NN graph
    """
    _logger.info("Starting minimum_spanning_tree calculation with k=%d", k)

    embeddings_np = flatten_samples(embeddings)
    _logger.debug("Embeddings shape: %s", embeddings_np.shape)

    # Get k-nearest neighbors and build MST
    neighbors, distances = compute_neighbor_distances(embeddings_np, k=k)
    mst_edges = minimum_spanning_tree_edges(embeddings_np, neighbors, distances)

    source = mst_edges[:, 0].astype(np.int64)
    target = mst_edges[:, 1].astype(np.int64)

    _logger.info("MST calculation complete: %d edges computed", len(source))

    return {"source": source, "target": target}


def compute_neighbor_distances(
    embeddings: ArrayND[float], neighbor_embeddings: ArrayND[float] | None = None, k: int = 10
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """
    Compute k nearest neighbors for each point in data (self-query, excluding self).

    Parameters
    ----------
    embeddings : ArrayND[float]
        Input data array with shape (n_samples, n_features). Can be an N dimensional list,
        or array-like object.
    k : int, default=10
        Number of neighbors to find (excluding self)

    Returns
    -------
    neighbors : NDArray[np.int64]
        Indices of k nearest neighbors for each point with shape (n_samples, k)
    distances : NDArray[np.float32]
        Distances to k nearest neighbors for each point with shape (n_samples, k)

    See Also
    --------
    compute_neighbors : For querying neighbors between two different datasets
    """
    _logger.debug("Computing neighbor distances with k=%d", k)
    embeddings_np = as_numpy(embeddings, required_ndim=2)
    nbr_embeddings_np = None if neighbor_embeddings is None else as_numpy(neighbor_embeddings, required_ndim=2)
    return _compute_nearest_neighbors(embeddings_np, nbr_embeddings_np, k, algorithm="brute", return_distances=True)


def compute_neighbors(
    data_fit: ArrayND[float],
    data_query: ArrayND[float] | None = None,
    k: int = 1,
    algorithm: Literal["auto", "ball_tree", "kd_tree"] = "auto",
) -> NDArray[np.int64]:
    """
    For each sample in data_query, compute the k nearest neighbors in data_fit.

    Parameters
    ----------
    data_fit : ArrayND[float]
        Reference points to search with shape (n_samples_fit, n_features).
        Can be an N dimensional list, or array-like object. This is the dataset
        that will be indexed for neighbor search.
    data_query : ArrayND[float]
        Query points with shape (n_samples_query, n_features).
        Can be an N dimensional list, or array-like object. For each of these
        points, find k nearest neighbors in data_fit.
    k : int, default=1
        The number of neighbors to find
    algorithm : {"auto", "ball_tree", "kd_tree"}, default="auto"
        Tree method for nearest neighbor computation

    Returns
    -------
    NDArray[np.int64]
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

    data_fit = flatten_samples(data_fit)
    if data_query is not None:
        data_query = flatten_samples(data_query)

    # Note: exclude_self=True handles the case where data_query and data_fit may overlap
    # but we want neighbors from data_fit, not self-matches
    neighbors = _compute_nearest_neighbors(data_fit, data_query, k, algorithm=algorithm, return_distances=False)
    return neighbors.squeeze()
