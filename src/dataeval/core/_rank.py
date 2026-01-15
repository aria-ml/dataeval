"""
Core ranking algorithms for embeddings.

This module provides pure algorithmic implementations that operate on
numpy arrays, independent of dataset abstractions.
"""

__all__ = []

import logging
from abc import ABC, abstractmethod
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from dataeval.config import EPSILON, get_seed

_logger = logging.getLogger(__name__)


class RankResult(TypedDict):
    """
    Type definition for ranking output.

    Attributes
    ----------
    indices : NDArray[np.intp]
        Indices that sort the data in order of priority according to the
        specified method and policy.
    scores : NDArray[np.float32] | None
        Ranking scores for each sample (only available for methods
        that compute scores: "knn" and "kmeans_distance"). Scores are ordered
        according to the original data order, not the ranked order.
    method : "knn", "kmeans_distance" or "kmeans_complexity"
        The ranking method that was used.
    policy : "hard_first", "easy_first", "stratified" or "class_balance"
        The selection policy that was applied.
    """

    indices: NDArray[np.intp]
    scores: NDArray[np.float32] | None
    method: Literal["knn", "kmeans_distance", "kmeans_complexity"]
    policy: Literal["hard_first", "easy_first", "stratified", "class_balance"]


class _Clusters:
    __slots__ = ["labels", "cluster_centers", "unique_labels"]

    labels: NDArray[np.intp]
    cluster_centers: NDArray[np.float64]
    unique_labels: NDArray[np.intp]

    def __init__(self, labels: NDArray[np.intp], cluster_centers: NDArray[np.float64]) -> None:
        self.labels = labels
        self.cluster_centers = cluster_centers
        self.unique_labels = np.unique(labels)

    def _dist2center(self, X: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
        dist = np.zeros(self.labels.shape, dtype=np.float32)
        for lab in self.unique_labels:
            dist[self.labels == lab] = np.linalg.norm(X[self.labels == lab, :] - self.cluster_centers[lab, :], axis=1)
        return dist

    def _complexity(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        num_clst_intra = int(np.maximum(np.minimum(int(self.unique_labels.shape[0] / 5), 20), 1))
        d_intra = np.zeros(self.unique_labels.shape)
        d_inter = np.zeros(self.unique_labels.shape)
        for cdx, lab in enumerate(self.unique_labels):
            d_intra[cdx] = np.mean(np.linalg.norm(X[self.labels == lab, :] - self.cluster_centers[cdx, :], axis=1))
            d_inter[cdx] = np.mean(
                np.linalg.norm(self.cluster_centers - self.cluster_centers[cdx, :], axis=1)[:num_clst_intra]
            )
        cj = d_intra * d_inter
        tau = 0.1
        exp = np.exp(cj / tau)
        prob: NDArray[np.float64] = exp / np.sum(exp)
        return prob

    def _sort_by_weights(self, X: NDArray[np.float64]) -> NDArray[np.intp]:
        pr = self._complexity(X)
        d2c = self._dist2center(X)
        inds_per_clst: list[NDArray[np.intp]] = []
        for lab in zip(self.unique_labels):
            inds = np.nonzero(self.labels == lab)[0]
            # 'hardest' first
            srt_inds = np.argsort(d2c[inds])[::-1]
            inds_per_clst.append(inds[srt_inds])
        glob_inds: list[NDArray[np.intp]] = []
        while not bool(np.all([arr.size == 0 for arr in inds_per_clst])):
            clst_ind = np.random.choice(self.unique_labels, 1, p=pr)[0]
            if inds_per_clst[clst_ind].size > 0:
                glob_inds.append(inds_per_clst[clst_ind][0])
            else:
                continue
            inds_per_clst[clst_ind] = inds_per_clst[clst_ind][1:]
        # sorted hardest first; reverse for consistency
        return np.array(glob_inds[::-1])


class _Sorter(ABC):
    scores: NDArray[np.float32] | None = None

    def __init__(self, *args: int, **kwargs: int | Literal["auto"] | None) -> None: ...

    @abstractmethod
    def _sort(self, embeddings: NDArray[Any], reference: NDArray[Any] | None = None) -> NDArray[np.intp]: ...


class _KNNSorter(_Sorter):
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


class _KMeansSorter(_Sorter):
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
        return _Clusters(clst.labels_.astype(np.intp), clst.cluster_centers_.astype(np.float64))


class _KMeansDistanceSorter(_KMeansSorter):
    def _sort(self, embeddings: NDArray[Any], reference: NDArray[Any] | None = None) -> NDArray[np.intp]:
        clst = self._get_clusters(embeddings if reference is None else reference)
        self.scores = clst._dist2center(embeddings)
        _logger.debug(
            "Distance to center scores: min=%.4f, max=%.4f, mean=%.4f",
            np.min(self.scores),
            np.max(self.scores),
            np.mean(self.scores),
        )
        return np.argsort(self.scores)


class _KMeansComplexitySorter(_KMeansSorter):
    def _sort(self, embeddings: NDArray[Any], reference: NDArray[Any] | None = None) -> NDArray[np.intp]:
        clst = self._get_clusters(embeddings if reference is None else reference)
        _logger.debug("Sorting by complexity weights")
        return clst._sort_by_weights(embeddings)


_SORTER_MAP: dict[str, type[_Sorter]] = {
    "knn": _KNNSorter,
    "kmeans_distance": _KMeansDistanceSorter,
    "kmeans_complexity": _KMeansComplexitySorter,
}


def _normalize(embeddings: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Normalize embeddings by maximum L2 norm."""
    emb = embeddings.copy()
    emb /= max(np.max(np.linalg.norm(emb, axis=1)), EPSILON)
    return emb


def _rank_base(
    embeddings: NDArray[np.floating[Any]],
    *,
    method: Literal["knn", "kmeans_distance", "kmeans_complexity"],
    k: int | None = None,
    c: int | None = None,
    n_init: int | Literal["auto"] = "auto",
    reference: NDArray[np.floating[Any]] | None = None,
) -> RankResult:
    """Internal function that performs the core ranking without policy application."""
    _logger.info(
        "Starting rank with method=%s, k=%s, c=%s, reference=%s",
        method,
        k,
        c,
        "provided" if reference is not None else "None",
    )

    # Validate method
    if method not in _SORTER_MAP:
        raise ValueError(f"Invalid method: {method}. Must be one of {list(_SORTER_MAP.keys())}")

    _logger.debug("Embeddings shape: %s", embeddings.shape)
    if reference is not None:
        _logger.debug("Reference shape: %s", reference.shape)

    # Normalize embeddings
    embeddings = _normalize(embeddings)
    if reference is not None:
        reference = _normalize(reference)

    # Create sorter based on method
    sorter_cls = _SORTER_MAP[method]
    sorter = sorter_cls(len(embeddings), k=k) if method == "knn" else sorter_cls(len(embeddings), c=c, n_init=n_init)

    # Sort (always returns easy_first order)
    indices = sorter._sort(embeddings, reference)

    _logger.info(
        "Rank complete: method=%s, %d samples ranked, has_scores=%s",
        method,
        len(indices),
        sorter.scores is not None,
    )

    return RankResult(
        indices=indices,
        scores=sorter.scores,
        method=method,
        policy="easy_first",
    )


def rank_knn(
    embeddings: NDArray[np.floating[Any]],
    k: int | None = None,
    reference: NDArray[np.floating[Any]] | None = None,
) -> RankResult:
    """
    Rank samples using k-nearest neighbors distance.

    Returns samples in easy-first order (low distance = prototypical samples).
    Use rerank_hard_first() to reverse the order, or other rerank_* functions
    to apply different selection policies.

    Parameters
    ----------
    embeddings : NDArray[np.floating]
        Embedding vectors to rank, shape (n_samples, n_features).
    k : int | None, default None
        Number of nearest neighbors. If None, uses sqrt(n_samples).
    reference : NDArray[np.floating] | None, default None
        Reference embeddings for comparative ranking. If provided, samples
        are ranked relative to the reference set rather than themselves.

    Returns
    -------
    RankResult
        Dictionary containing:

        - indices: NDArray[np.intp] - Ranked indices in easy-first order
        - scores: NDArray[np.float32] | None - KNN distance scores for each sample
        - method: str - "knn"
        - policy: str - "easy_first"

    Raises
    ------
    ValueError
        If k is invalid (>= dataset size or negative).

    Examples
    --------
    Basic ranking:

    >>> from dataeval.core import rank_knn
    >>> import numpy as np
    >>> embeddings = np.random.rand(100, 64).astype(np.float32)
    >>> result = rank_knn(embeddings, k=5)

    Hard-first order:

    >>> from dataeval.core import rank_knn, rerank_hard_first
    >>> result = rank_knn(embeddings, k=5)
    >>> result = rerank_hard_first(result)

    Rank relative to reference:

    >>> reference = np.random.rand(50, 64).astype(np.float32)
    >>> result = rank_knn(embeddings, k=5, reference=reference)
    """
    return _rank_base(embeddings, method="knn", k=k, reference=reference)


def rank_kmeans_distance(
    embeddings: NDArray[np.floating[Any]],
    c: int | None = None,
    n_init: int | Literal["auto"] = "auto",
    reference: NDArray[np.floating[Any]] | None = None,
) -> RankResult:
    """
    Rank samples using distance to cluster centers.

    Clusters embeddings using K-means and ranks by distance to assigned cluster
    centers. Returns samples in easy-first order (low distance = prototypical).

    Parameters
    ----------
    embeddings : NDArray[np.floating]
        Embedding vectors to rank, shape (n_samples, n_features).
    c : int | None, default None
        Number of clusters. If None, uses sqrt(n_samples).
    n_init : int | "auto", default "auto"
        Number of K-means initializations.
    reference : NDArray[np.floating] | None, default None
        Reference embeddings for comparative ranking. If provided, samples
        are ranked relative to the reference set rather than themselves.

    Returns
    -------
    RankResult
        Dictionary containing:

        - indices: NDArray[np.intp] - Ranked indices in easy-first order
        - scores: NDArray[np.float32] | None - Distance to cluster center for each sample
        - method: str - "kmeans_distance"
        - policy: str - "easy_first"

    Raises
    ------
    ValueError
        If c is invalid (>= dataset size or negative).

    Examples
    --------
    >>> from dataeval.core import rank_kmeans_distance
    >>> import numpy as np
    >>> embeddings = np.random.rand(100, 64).astype(np.float32)
    >>> result = rank_kmeans_distance(embeddings, c=10)
    """
    return _rank_base(embeddings, method="kmeans_distance", c=c, n_init=n_init, reference=reference)


def rank_kmeans_complexity(
    embeddings: NDArray[np.floating[Any]],
    c: int | None = None,
    n_init: int | Literal["auto"] = "auto",
    reference: NDArray[np.floating[Any]] | None = None,
) -> RankResult:
    """
    Rank samples using cluster complexity weighting.

    Uses a weighted sampling strategy based on intra-cluster and inter-cluster
    distances. Returns samples in easy-first order.

    Note: This method does not produce scores, so rerank_stratified() cannot
    be used with results from this function.

    Parameters
    ----------
    embeddings : NDArray[np.floating]
        Embedding vectors to rank, shape (n_samples, n_features).
    c : int | None, default None
        Number of clusters. If None, uses sqrt(n_samples).
    n_init : int | "auto", default "auto"
        Number of K-means initializations.
    reference : NDArray[np.floating] | None, default None
        Reference embeddings for comparative ranking. If provided, samples
        are ranked relative to the reference set rather than themselves.

    Returns
    -------
    RankResult
        Dictionary containing:

        - indices: NDArray[np.intp] - Ranked indices in easy-first order
        - scores: None (this method does not produce scores)
        - method: str - "kmeans_complexity"
        - policy: str - "easy_first"

    Raises
    ------
    ValueError
        If c is invalid (>= dataset size or negative).

    Examples
    --------
    >>> from dataeval.core import rank_kmeans_complexity
    >>> import numpy as np
    >>> embeddings = np.random.rand(100, 64).astype(np.float32)
    >>> result = rank_kmeans_complexity(embeddings, c=10)
    """
    return _rank_base(embeddings, method="kmeans_complexity", c=c, n_init=n_init, reference=reference)
