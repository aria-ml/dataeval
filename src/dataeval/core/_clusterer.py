__all__ = []

import logging
from abc import ABC, abstractmethod
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from dataeval.config import get_seed
from dataeval.core._mst import compute_neighbor_distances, minimum_spanning_tree_edges
from dataeval.types import ArrayND
from dataeval.utils.arrays import flatten_samples, to_numpy

_logger = logging.getLogger(__name__)


class _Clusters:
    __slots__ = ["labels", "cluster_centers", "unique_labels"]

    labels: NDArray[np.intp]
    cluster_centers: NDArray[np.float64]
    unique_labels: NDArray[np.intp]

    def __init__(self, labels: NDArray[np.intp], cluster_centers: NDArray[np.float64]) -> None:
        self.labels = labels
        self.cluster_centers = cluster_centers
        self.unique_labels = np.unique(labels)

    def _dist2center(self, embeddings: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
        """Calculate distance from each point to its assigned cluster center."""
        dist = np.full(self.labels.shape, np.inf, dtype=np.float32)
        for lab in self.unique_labels:
            mask = self.labels == lab
            dist[mask] = np.linalg.norm(embeddings[mask, :] - self.cluster_centers[lab, :], axis=1)
        return dist

    def _complexity(self, embeddings: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculate complexity-based probability weights for each cluster."""
        num_clst_intra = int(np.maximum(np.minimum(int(self.unique_labels.shape[0] / 5), 20), 1))
        d_intra = np.zeros(self.unique_labels.shape)
        d_inter = np.zeros(self.unique_labels.shape)
        for cdx, lab in enumerate(self.unique_labels):
            d_intra[cdx] = np.mean(
                np.linalg.norm(embeddings[self.labels == lab, :] - self.cluster_centers[cdx, :], axis=1)
            )
            d_inter[cdx] = np.mean(
                np.linalg.norm(self.cluster_centers - self.cluster_centers[cdx, :], axis=1)[:num_clst_intra]
            )
        cj = d_intra * d_inter
        tau = 0.1
        exp = np.exp(cj / tau)
        prob: NDArray[np.float64] = exp / np.sum(exp)
        return prob

    def _sort_by_weights(self, embeddings: NDArray[np.float64]) -> NDArray[np.intp]:
        """Sort samples using complexity-based weighted sampling."""
        pr = self._complexity(embeddings)
        d2c = self._dist2center(embeddings)
        inds_per_clst: list[NDArray[np.intp]] = []
        for lab in self.unique_labels:
            inds = np.nonzero(self.labels == lab)[0]
            # 'hardest' first == farthest distance
            srt_inds = np.argsort(d2c[inds])[::-1]
            inds_per_clst.append(inds[srt_inds])
        glob_inds: list[NDArray[np.intp]] = []
        while not bool(np.all([arr.size == 0 for arr in inds_per_clst])):
            clst_ind = np.random.choice(self.unique_labels, 1, p=pr)[0]
            if inds_per_clst[clst_ind].size > 0:
                glob_inds.append(inds_per_clst[clst_ind][0])
                inds_per_clst[clst_ind] = inds_per_clst[clst_ind][1:]
        # sorted hardest first; reverse for consistency
        return np.array(glob_inds[::-1])


class _Sorter(ABC):
    """Base class for sorting/ranking algorithms."""

    scores: NDArray[np.float32] | None = None

    def __init__(self, *args: int, **kwargs: int | Literal["auto"] | Literal["kmeans", "hdbscan"] | None) -> None: ...

    @abstractmethod
    def _sort(self, embeddings: NDArray[Any], reference: NDArray[Any] | None = None) -> NDArray[np.intp]: ...


class _KNNSorter(_Sorter):
    """Sort samples by k-nearest neighbor distance."""

    def __init__(self, samples: int, k: int | None) -> None:
        if k is None or k <= 0:
            k = int(np.sqrt(samples))
            _logger._log(logging.INFO, f"Setting k to default value of {k}", {"k": k, "samples": samples})
        elif k >= samples:
            raise ValueError(f"k={k} should be less than dataset size ({samples})")
        elif k >= samples / 10 and k > np.sqrt(samples):
            _logger.warning(
                f"Variable k={k} is large with respect to dataset size but valid; "
                + f"a nominal recommendation is k={int(np.sqrt(samples))}"
            )
        self._k = k

    def _sort(self, embeddings: NDArray[Any], reference: NDArray[Any] | None = None) -> NDArray[np.intp]:
        _logger.debug("Computing KNN distances with k=%d", self._k)
        if reference is None:
            dists = pairwise_distances(embeddings, embeddings).astype(np.float32)
            np.fill_diagonal(dists, np.inf)
        else:
            dists = pairwise_distances(embeddings, reference).astype(np.float32)
        self.scores = np.sort(dists, axis=1)[:, self._k]
        _logger.debug(
            "KNN scores computed: min=%.4f, max=%.4f, mean=%.4f",
            np.min(self.scores),
            np.max(self.scores),
            np.mean(self.scores),
        )
        return np.argsort(self.scores)


class _KMeansSorter:
    """Perform K-means clustering."""

    def __init__(self, samples: int, c: int | None, n_init: int | Literal["auto"] = "auto") -> None:
        if c is None or c <= 0:
            c = int(np.sqrt(samples))
            _logger._log(logging.INFO, f"Setting the value of num_clusters to a default value of {c}", {})
        if c >= samples:
            raise ValueError(f"c={c} should be less than dataset size ({samples})")
        self._c = c
        self._n_init = n_init

    def _get_clusters(self, embeddings: NDArray[Any]) -> _Clusters:
        _logger.debug("Computing KMeans clustering with c=%d clusters", self._c)
        clst = KMeans(n_clusters=self._c, init="k-means++", n_init=self._n_init, random_state=get_seed())  # type: ignore - n_init allows int but is typed as str
        clst.fit(embeddings)
        if clst.labels_ is None or clst.cluster_centers_ is None:
            raise ValueError("Clustering failed to produce labels or cluster centers")
        n_samples_per_cluster = np.bincount(clst.labels_)
        _logger.debug(
            "KMeans clustering complete: %d clusters, samples per cluster: min=%d, max=%d, mean=%.1f",
            self._c,
            np.min(n_samples_per_cluster),
            np.max(n_samples_per_cluster),
            np.mean(n_samples_per_cluster),
        )
        return _Clusters(clst.labels_, clst.cluster_centers_)


class _HDBSCANSorter:
    """Perform HDBSCAN clustering."""

    def __init__(
        self,
        samples: int,
        n_expected_clusters: int | None = None,
        max_cluster_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        if n_expected_clusters is not None and n_expected_clusters >= samples:
            raise ValueError(f"n_expected_clusters={n_expected_clusters} should be less than dataset size ({samples})")
        self.n_expected_clusters = n_expected_clusters
        self.max_cluster_size = max_cluster_size

    def _get_clusters(self, embeddings: NDArray[Any]) -> _Clusters:
        _logger.debug("Computing HDBSCAN clustering with expected clusters=%s", self.n_expected_clusters)
        clst = _HDBSCAN(n_clusters=self.n_expected_clusters, max_cluster_size=self.max_cluster_size)
        clst.fit(embeddings)
        if clst.labels_ is None or clst.cluster_centers_ is None:
            raise ValueError("Clustering failed to produce labels or cluster centers")
        if (clst.labels_ == -1).any():
            all_distances = np.linalg.norm(
                embeddings[:, np.newaxis, :] - clst.cluster_centers_[np.newaxis, :, :], axis=2
            )
            # Get nearest cluster index for each point
            labels = np.argmin(all_distances, axis=1)
        else:
            labels = clst.labels_

        n_samples_per_cluster = np.bincount(labels)
        _logger.debug(
            "HDBSCAN clustering complete: %d clusters, samples per cluster: min=%d, max=%d, mean=%.1f",
            clst.unique_clusters,
            np.min(n_samples_per_cluster),
            np.max(n_samples_per_cluster),
            np.mean(n_samples_per_cluster),
        )
        return _Clusters(labels, clst.cluster_centers_)


class _DistanceSorter(_Sorter):
    """Sort samples by distance to cluster centers."""

    def __init__(
        self,
        samples: int,
        algorithm: Literal["kmeans", "hdbscan"] = "kmeans",
        c: int | None = None,
        n_init: int | Literal["auto"] = "auto",
        max_cluster_size: int | None = None,
    ) -> None:
        # Create the appropriate clustering sorter
        if algorithm == "kmeans":
            self._clusterer = _KMeansSorter(samples, c, n_init)
        else:  # hdbscan
            self._clusterer = _HDBSCANSorter(samples, c, max_cluster_size)

    def _sort(self, embeddings: NDArray[Any], reference: NDArray[Any] | None = None) -> NDArray[np.intp]:
        clst = self._clusterer._get_clusters(embeddings if reference is None else reference)
        self.scores = clst._dist2center(embeddings)
        _logger.debug(
            "Distance to center scores: min=%.4f, max=%.4f, mean=%.4f",
            np.min(self.scores),
            np.max(self.scores),
            np.mean(self.scores),
        )
        return np.argsort(self.scores)


class _ComplexitySorter(_Sorter):
    """Sort samples using cluster complexity weighting."""

    def __init__(
        self,
        samples: int,
        algorithm: Literal["kmeans", "hdbscan"] = "kmeans",
        c: int | None = None,
        n_init: int | Literal["auto"] = "auto",
        max_cluster_size: int | None = None,
    ) -> None:
        # Create the appropriate clustering sorter
        if algorithm == "kmeans":
            self._clusterer = _KMeansSorter(samples, c, n_init)
        else:  # hdbscan
            self._clusterer = _HDBSCANSorter(samples, c, max_cluster_size)

    def _sort(self, embeddings: NDArray[Any], reference: NDArray[Any] | None = None) -> NDArray[np.intp]:
        clst = self._clusterer._get_clusters(embeddings if reference is None else reference)
        _logger.debug("Sorting by complexity weights")
        return clst._sort_by_weights(embeddings)


_SORTER_MAP: dict[str, type[_Sorter]] = {
    "knn": _KNNSorter,
    "kmeans_distance": _DistanceSorter,
    "kmeans_complexity": _ComplexitySorter,
    "hdbscan_distance": _DistanceSorter,
    "hdbscan_complexity": _ComplexitySorter,
}


class _HDBSCAN:
    """
    Uses hierarchical clustering on the flattened data and returns clustering
    information.

    Parameters
    ----------
    embeddings : ArrayND, shape - (N, ...)
        A dataset that can be a list, or array-like object. Function expects
        the data to have 2 or more dimensions which will flatten to (N, P) where N is
        the number of observations in a P-dimensional space.
    n_clusters : int, optional
        Hint for the expected number of clusters (e.g., number of classes in dataset).
        This adaptively adjusts min_cluster_size to encourage finding
        approximately this many clusters. Useful when you have
        domain knowledge about the data structure.
    max_cluster_size : int, optional
        Option to limit the size of the identified clusters. Useful when you have
        domain knowledge about the data structure.

    Attributes
    ----------
    labels_ : NDArray[np.intp]
        Assigned clusters
    cluster_centers_ : NDArray[np.floating]
        Cluster centers, shape (n_clusters, n_features)
    unique_clusters : NDArray[np.intp]
        Array of unique cluster IDs (excluding -1)
    mst : NDArray[np.float32]
        The minimum spanning tree of the data
    linkage_tree : NDArray[np.float32]
        The linkage array of the data
    membership_strengths : NDArray[np.float32]
        The strength of the data point belonging to the assigned cluster
    k_neighbors : NDArray[np.int64]
        Indices of the nearest points in the population matrix
    k_distances : NDArray[np.float32]
        Array representing the lengths to points
    """

    def __init__(self, n_clusters: int | None = None, max_cluster_size: int | None = None) -> None:
        self.n_expected_clusters = n_clusters
        self.max_cluster_size = max_cluster_size
        self.single_cluster = True
        self.cluster_selection_epsilon = 0.0
        self.cluster_selection_method = "eom"

    def fit(self, embeddings: NDArray[np.floating]) -> "_HDBSCAN":
        """
        Find clusters based on hierarchical density-based clustering.

        Parameters
        ----------
        embeddings : NDArray[np.floating]
            The embedding vectors, shape (n_samples, n_features)

        Returns
        -------
        self : _HDBSCAN
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

        _logger.info("Starting HDBSCAN cluster calculation")

        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings should be a 2 dimensional array, got {embeddings.ndim} dimensions")

        samples, features = embeddings.shape  # Due to flatten_samples(), we know shape has a length of 2

        _logger.debug("Input embeddings shape after flattening: (%d samples, %d features)", samples, features)

        if samples < 2:
            raise ValueError(f"Data should have at least 2 samples; got {samples}")
        if features < 1:
            raise ValueError(f"Samples should have at least 1 feature; got {features}")

        # Adaptive min_cluster_size based on expected clusters hint
        if self.n_expected_clusters is not None:
            # Encourage finding approximately n_expected_clusters
            # Divide by 3 to allow smaller, more granular clusters
            self.min_cluster_size = max(5, samples // (self.n_expected_clusters * 3))
            _logger.debug(
                "Using adaptive min_cluster_size=%d for %d expected clusters",
                self.min_cluster_size,
                self.n_expected_clusters,
            )
        else:
            # Default behavior: use 5% but cap at 100
            min_num = int(samples * 0.05)
            self.min_cluster_size = min(max(5, min_num), 100)
            _logger.debug("Using default min_cluster_size=%d", self.min_cluster_size)

        max_num = self.max_cluster_size if self.max_cluster_size is not None else np.inf
        if max_num <= self.min_cluster_size:
            _logger.warning("Provided max_cluster_size is smaller than min_cluster_size. Resetting to infinity.")
            max_num = np.inf

        max_neighbors = min(25, samples - 1)
        _logger.debug("Computing neighbors with max_neighbors=%d", max_neighbors)
        kneighbors, kdistances = compute_neighbor_distances(embeddings, k=max_neighbors)
        unsorted_mst: NDArray[np.float32] = minimum_spanning_tree_edges(embeddings, kneighbors, kdistances)
        mst: NDArray[np.float32] = unsorted_mst[np.argsort(unsorted_mst.T[2])]
        linkage_tree: NDArray[np.float32] = mst_to_linkage_tree(mst).astype(np.float32)
        condensed_tree = condense_tree(linkage_tree, self.min_cluster_size, max_cluster_size=max_num)
        cluster_tree = cluster_tree_from_condensed_tree(condensed_tree)

        selected_clusters = extract_eom_clusters(
            condensed_tree, cluster_tree, max_cluster_size=max_num, allow_single_cluster=self.single_cluster
        )

        # Uncomment if cluster_selection_method is made a parameter
        # if self.cluster_selection_method != "eom":
        #     selected_clusters = extract_leaves(condensed_tree, allow_single_cluster=single_cluster)

        # Uncomment if cluster_selection_epsilon is made a parameter
        # if len(selected_clusters) > 1 and self.cluster_selection_epsilon > 0.0:
        #     selected_clusters = cluster_epsilon_search(
        #         selected_clusters,
        #         cluster_tree,
        #         min_epsilon=self.cluster_selection_epsilon,
        #     )

        cluster_labels: NDArray[np.intp] = get_cluster_label_vector(
            condensed_tree,
            selected_clusters,
            self.cluster_selection_epsilon,
            n_samples=mst.shape[0] + 1,
        )

        membership_strengths: NDArray[np.float32] = get_point_membership_strength_vector(
            condensed_tree,
            selected_clusters,
            cluster_labels,
        )

        # If everything is an outlier, then treat it as a single cluster
        if (cluster_labels == -1).all():
            cluster_labels += 1
        self.unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
        n_clusters = len(self.unique_clusters)
        n_outliers = np.sum(cluster_labels == -1)
        _logger.info(
            "Cluster calculation complete: found %d clusters, %d outliers out of %d samples",
            n_clusters,
            n_outliers,
            samples,
        )

        # Calculate cluster centers
        centers = np.zeros((n_clusters, features), dtype=np.float64)
        for i, cluster_id in enumerate(self.unique_clusters):
            cluster_mask = cluster_labels == cluster_id
            centers[i] = embeddings[cluster_mask].mean(axis=0)

        self.labels_ = cluster_labels
        self.cluster_centers_ = centers
        self.mst = mst
        self.linkage_tree = linkage_tree
        self.membership_strengths = membership_strengths
        self.k_neighbors = kneighbors
        self.k_distances = kdistances

        return self


class ClusterResult(TypedDict):
    """
    Cluster output data structure.

    Attributes
    ----------
    clusters : NDArray[np.int64]
        Assigned clusters
    mst : NDArray[np.float32]
        The minimum spanning tree of the data
    linkage_tree : NDArray[np.float32]
        The linkage array of the data
    membership_strengths : NDArray[np.float32]
        The strength of the data point belonging to the assigned cluster
    k_neighbors : NDArray[np.int64]
        Indices of the nearest points in the population matrix
    k_distances : NDArray[np.float32]
        Array representing the lengths to points
    """

    clusters: NDArray[np.int64]
    mst: NDArray[np.float32]
    linkage_tree: NDArray[np.float32]
    membership_strengths: NDArray[np.float32]
    k_neighbors: NDArray[np.int64]
    k_distances: NDArray[np.float32]


class ClusterStats(TypedDict):
    """
    Pre-calculated statistics for adaptive outlier detection.

    Attributes
    ----------
    cluster_ids : NDArray[np.int64]
        Array of unique cluster IDs (excluding -1)
    centers : NDArray[np.floating]
        Cluster centers, shape (n_clusters, n_features)
    cluster_distances_mean : NDArray[np.floating]
        Mean distance from points to their cluster center, shape (n_clusters,)
    cluster_distances_std : NDArray[np.floating]
        Standard deviation of distances within each cluster, shape (n_clusters,)
    distances : NDArray[np.floating]
        Distance from each point to its nearest cluster center, shape (n_samples,)
    nearest_cluster_idx : NDArray[np.int64]
        Index of nearest cluster center for each point, shape (n_samples,)
    """

    cluster_ids: NDArray[np.int64]
    centers: NDArray[np.floating]
    cluster_distances_mean: NDArray[np.floating]
    cluster_distances_std: NDArray[np.floating]
    distances: NDArray[np.floating]
    nearest_cluster_idx: NDArray[np.int64]


def compute_cluster_stats(
    embeddings: NDArray[np.floating],
    cluster_labels: _Clusters | NDArray[np.int64],
) -> ClusterStats:
    """
    Compute cluster centers and distance statistics for adaptive outlier detection.

    Parameters
    ----------
    embeddings : NDArray[np.floating]
        The embedding vectors, shape (n_samples, n_features)
    cluster_labels : NDArray[np.int64] | _Clusters
        Cluster labels returned from a clustering algorithm (-1 for outliers) or an internal _Clusters object

    Returns
    -------
    ClusterStats
        Pre-calculated statistics with empty arrays if no valid clusters found
    """
    _logger.debug("Computing cluster statistics for %d samples", len(embeddings))

    n_samples = len(embeddings)

    # Handle raw label arrays
    if isinstance(cluster_labels, _Clusters):
        cluster_obj = cluster_labels

    else:
        # Get unique clusters (excluding -1)
        unique_labels = np.unique(cluster_labels[cluster_labels >= 0])

        n_clusters = len(unique_labels)
        n_features = embeddings.shape[1]

        if n_clusters == 0:
            _logger.warning("No valid clusters found, returning empty statistics")
            return ClusterStats(
                cluster_ids=np.array([], dtype=np.int64),
                centers=np.array([], dtype=embeddings.dtype),
                cluster_distances_mean=np.array([], dtype=embeddings.dtype),
                cluster_distances_std=np.array([], dtype=embeddings.dtype),
                distances=np.full(n_samples, np.inf, dtype=embeddings.dtype),
                nearest_cluster_idx=np.full(n_samples, -1, dtype=np.int64),
            )

        # Compute centers
        centers = np.zeros((n_clusters, n_features), dtype=np.float64)
        for i, cluster_id in enumerate(unique_labels):
            cluster_mask = cluster_labels == cluster_id
            centers[i] = embeddings[cluster_mask].mean(axis=0)

        if (cluster_labels < 0).any():
            # Pre-calculate distance from each point to its nearest cluster center
            # Shape: (n_samples, n_clusters)
            all_distances = np.linalg.norm(embeddings[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
            # Get nearest cluster index for each point
            labels = np.argmin(all_distances, axis=1)
        else:
            labels = cluster_labels

        # Create a _Cluster object
        cluster_obj = _Clusters(labels, centers)

    nearest_cluster_idx = cluster_obj.labels
    unique_clusters = cluster_obj.unique_labels
    n_clusters = len(unique_clusters)

    if n_clusters == 0:
        _logger.warning("No valid clusters found, returning empty statistics")
        return ClusterStats(
            cluster_ids=np.array([], dtype=np.int64),
            centers=np.array([], dtype=embeddings.dtype),
            cluster_distances_mean=np.array([], dtype=embeddings.dtype),
            cluster_distances_std=np.array([], dtype=embeddings.dtype),
            distances=np.full(n_samples, np.inf, dtype=embeddings.dtype),
            nearest_cluster_idx=np.full(n_samples, -1, dtype=np.int64),
        )

    _logger.debug("Found %d unique clusters (excluding outliers)", len(unique_clusters))

    min_distances = cluster_obj._dist2center(embeddings)
    cluster_distances_mean = np.zeros(n_clusters, dtype=embeddings.dtype)
    cluster_distances_std = np.zeros(n_clusters, dtype=embeddings.dtype)
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = nearest_cluster_idx == cluster_id
        distances = min_distances[cluster_mask]
        cluster_distances_mean[i] = distances.mean()
        cluster_distances_std[i] = distances.std()

    _logger.debug("Cluster stats computed: %d centers with mean distances %s", n_clusters, cluster_distances_mean)

    return ClusterStats(
        cluster_ids=unique_clusters,
        centers=cluster_obj.cluster_centers,
        cluster_distances_mean=cluster_distances_mean,
        cluster_distances_std=cluster_distances_std,
        distances=min_distances,
        nearest_cluster_idx=nearest_cluster_idx,
    )


def cluster(
    embeddings: ArrayND[float],
    algorithm: Literal["kmeans", "hdbscan"] = "hdbscan",
    n_clusters: int | None = None,
    max_cluster_size: int | None = None,
    n_init: int | Literal["auto"] = "auto",
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
    algorithm : "kmeans" | "hdbscan", default "hdbscan"
        The clustering algorithm to use.
    n_clusters : int, optional
        The expected number of clusters (e.g., number of classes in dataset).
        For KMeans, this is the exact number of clusters to find.
        For HDBSCAN, adaptively adjusts min_cluster_size to encourage finding
        approximately this many clusters.
    max_cluster_size : int, optional
        Option to limit the size of the identified clusters. Useful when you have
        domain knowledge about the data structure. (HDBSCAN only)
    n_init : int | "auto", default "auto"
        Number of K-means initializations (KMeans only).

    Returns
    -------
    ClusterResult
        Mapping with keys:
        - clusters : NDArray[np.int64] - Assigned clusters
        - mst : NDArray[np.float32] - The minimum spanning tree of the data
        - linkage_tree : NDArray[np.float32] - The linkage array of the data
        - membership_strengths : NDArray[np.float32] - The strength of the data point belonging to the assigned cluster
        - k_neighbors : NDArray[np.int64] - Indices of the nearest points in the population matrix
        - k_distances : NDArray[np.float32] - Array representing the lengths to points

    Notes
    -----
    The cluster function works best when the length of the feature dimension,
    P, is less than 500. If flattening a CxHxW image results in a dimension
    larger than 500, then it is recommended to reduce the dimensions.

    Examples
    --------
    Two distinct clusters

    >>> import numpy as np
    >>> import sklearn.datasets as dsets
    >>> from dataeval.core import cluster
    >>> clusterer_images = dsets.make_blobs(
    ...     n_samples=50, centers=np.array([(-1, -1), (1, 1)]), cluster_std=0.5, random_state=33
    ... )[0]

    Clustering via HDBSCAN

    >>> output = cluster(clusterer_images, algorithm="hdbscan")
    >>> output["clusters"]
    array([0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0,
           0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0,
           1, 0, 0, 1, 1, 1])

    Clustering via KMeans

    >>> output = cluster(clusterer_images, algorithm="kmeans", n_clusters=2)
    >>> output["clusters"]
    array([0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0,
           0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0,
           1, 0, 0, 1, 1, 1])
    """
    _logger.info("Starting cluster calculation with algorithm=%s", algorithm)

    x: NDArray[Any] = flatten_samples(to_numpy(embeddings))
    samples, features = x.shape  # Due to flatten_samples(), we know shape has a length of 2

    _logger.debug("Input embeddings shape after flattening: (%d samples, %d features)", samples, features)

    if samples < 2:
        raise ValueError(f"Data should have at least 2 samples; got {samples}")
    if features < 1:
        raise ValueError(f"Samples should have at least 1 feature; got {features}")

    if algorithm == "hdbscan":
        clst = _HDBSCAN(n_clusters=n_clusters, max_cluster_size=max_cluster_size)
        clst.fit(x)

        return ClusterResult(
            clusters=clst.labels_,
            mst=clst.mst,
            linkage_tree=clst.linkage_tree,
            membership_strengths=clst.membership_strengths,
            k_neighbors=clst.k_neighbors,
            k_distances=clst.k_distances,
        )

    from dataeval.core._fast_hdbscan._cluster_trees import mst_to_linkage_tree

    # Compute neighbors for k_neighbors and k_distances output
    max_neighbors = min(25, samples - 1)
    _logger.debug("Computing neighbors with max_neighbors=%d", max_neighbors)
    kneighbors, kdistances = compute_neighbor_distances(x, k=max_neighbors)
    # Compute mst and linkage_tree
    unsorted_mst: NDArray[np.float32] = minimum_spanning_tree_edges(x, kneighbors, kdistances)
    mst: NDArray[np.float32] = unsorted_mst[np.argsort(unsorted_mst.T[2])]
    linkage_tree: NDArray[np.float32] = mst_to_linkage_tree(mst).astype(np.float32)

    # KMeans clustering
    if n_clusters is None or n_clusters <= 0:
        n_clusters = int(np.sqrt(samples))
        _logger.info(f"Setting n_expected_clusters to default value of {n_clusters}")

    if n_clusters >= samples:
        raise ValueError(f"n_expected_clusters={n_clusters} should be less than dataset size ({samples})")

    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=n_init, random_state=get_seed())  # type: ignore
    kmeans.fit(x)

    if kmeans.labels_ is None or kmeans.cluster_centers_ is None:
        raise ValueError("KMeans failed to produce labels or cluster centers")

    clusters = kmeans.labels_.astype(np.int64)
    centers = kmeans.cluster_centers_

    # Calculate membership strengths based on distance to cluster center
    distances = np.zeros(samples, dtype=np.float32)
    for i in range(n_clusters):
        mask = clusters == i
        cluster_points = x[mask]
        distances[mask] = np.linalg.norm(cluster_points - centers[i], axis=1)

    # Normalize distances to [0, 1] and invert for membership strength
    max_dist = distances.max()
    membership_strengths = 1.0 - distances / max_dist if max_dist > 0 else np.ones(samples, dtype=np.float32)

    _logger.info(
        "KMeans clustering complete: found %d clusters for %d samples",
        n_clusters,
        samples,
    )

    return ClusterResult(
        clusters=clusters,
        mst=mst,
        linkage_tree=linkage_tree,
        membership_strengths=membership_strengths,
        k_neighbors=kneighbors,
        k_distances=kdistances,
    )
