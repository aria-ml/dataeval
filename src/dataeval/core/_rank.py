"""
Core ranking algorithms for embeddings.

This module provides pure algorithmic implementations that operate on
numpy arrays, independent of dataset abstractions.
"""

from __future__ import annotations

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
        if reference is None:
            dists = pairwise_distances(embeddings, embeddings).astype(np.float32)
            np.fill_diagonal(dists, np.inf)
        else:
            dists = pairwise_distances(embeddings, reference).astype(np.float32)
        self.scores = np.sort(dists, axis=1)[:, self._k]
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
        clst = KMeans(n_clusters=self._c, init="k-means++", n_init=self._n_init, random_state=get_seed())  # type: ignore - n_init allows int but is typed as str
        clst.fit(embeddings)
        if clst.labels_ is None or clst.cluster_centers_ is None:
            raise ValueError("Clustering failed to produce labels or cluster centers")
        return _Clusters(clst.labels_, clst.cluster_centers_)


class _KMeansDistanceSorter(_KMeansSorter):
    def _sort(self, embeddings: NDArray[Any], reference: NDArray[Any] | None = None) -> NDArray[np.intp]:
        clst = self._get_clusters(embeddings if reference is None else reference)
        self.scores = clst._dist2center(embeddings)
        return np.argsort(self.scores)


class _KMeansComplexitySorter(_KMeansSorter):
    def _sort(self, embeddings: NDArray[Any], reference: NDArray[Any] | None = None) -> NDArray[np.intp]:
        clst = self._get_clusters(embeddings if reference is None else reference)
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


def _compute_bin_extents(scores: NDArray[np.floating[Any]]) -> tuple[np.float64, np.float64]:
    """
    Compute min/max bin extents for scores, padding outward by epsilon.

    Parameters
    ----------
    scores : NDArray[np.float64]
        Array of floats to bin

    Returns
    -------
    tuple[np.float64, np.float64]
        (min, max) scores padded outward by epsilon = 1e-6 * range(scores).
    """
    scores = scores.astype(np.float64)
    min_score = np.min(scores)
    max_score = np.max(scores)
    rng = max_score - min_score
    eps = rng * 1e-6
    return min_score - eps, max_score + eps


def _select_ordered_by_label(labels: NDArray[np.integer[Any]]) -> NDArray[np.intp]:
    """
    Given labels (class, group, bin, etc) sorted with decreasing priority,
    rerank so that we have approximate class/group balance.

    Parameters
    ----------
    labels : NDArray[np.integer]
        Class label or group ID per instance in order of decreasing priority

    Returns
    -------
    NDArray[np.intp]
        Indices that sort samples according to uniform class balance or
        group membership while respecting priority of the initial ordering.
    """
    labels = np.array(labels)
    num_samp = labels.shape[0]
    selected = np.zeros(num_samp, dtype=bool)
    # preserve ordering
    _, index = np.unique(labels, return_index=True)
    u_lab = labels[np.sort(index)]
    n_cls = len(u_lab)

    resort_inds = []
    cls_idx = 0
    n = 0
    while len(resort_inds) < num_samp:
        c0 = u_lab[cls_idx % n_cls]
        samples_available = (~selected) * (labels == c0)
        if any(samples_available):
            i0 = np.argmax(samples_available)  # selects first occurrence
            resort_inds.append(i0)
            selected[i0] = True
        cls_idx += 1
        n += 1
    return np.array(resort_inds).astype(np.intp)


def _stratified_rerank(
    scores: NDArray[np.floating[Any]],
    indices: NDArray[np.integer[Any]],
    num_bins: int = 50,
) -> NDArray[np.intp]:
    """
    Re-rank samples by sampling uniformly over binned scores.

    This de-weights selection of samples with similar scores and encourages
    both prototypical and challenging samples near the decision boundary.

    Parameters
    ----------
    scores : NDArray[np.floating]
        Ranking scores sorted in order of decreasing priority
    indices : NDArray[np.integer]
        Indices to be re-sorted according to stratified sampling of scores.
        Indices are ordered by decreasing priority.
    num_bins : int, default 50
        Number of bins for stratification.

    Returns
    -------
    NDArray[np.intp]
        Re-ranked indices
    """
    mn, mx = _compute_bin_extents(scores)
    bin_edges = np.linspace(mn, mx, num=num_bins + 1, endpoint=True)
    bin_label = np.digitize(scores, bin_edges)
    srt_inds = _select_ordered_by_label(bin_label)
    return indices[srt_inds].astype(np.intp)


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
    # Validate method
    if method not in _SORTER_MAP:
        raise ValueError(f"Invalid method: {method}. Must be one of {list(_SORTER_MAP.keys())}")

    # Normalize embeddings
    embeddings = _normalize(embeddings)
    if reference is not None:
        reference = _normalize(reference)

    # Create sorter based on method
    sorter_cls = _SORTER_MAP[method]
    sorter = sorter_cls(len(embeddings), k=k) if method == "knn" else sorter_cls(len(embeddings), c=c, n_init=n_init)

    # Sort (always returns easy_first order)
    indices = sorter._sort(embeddings, reference)

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
        - indices : Ranked indices in easy-first order
        - scores : KNN distance scores for each sample
        - method : "knn"
        - policy : "easy_first"

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
        - indices : Ranked indices in easy-first order
        - scores : Distance to cluster center for each sample
        - method : "kmeans_distance"
        - policy : "easy_first"

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
        - indices : Ranked indices in easy-first order
        - scores : None (this method does not produce scores)
        - method : "kmeans_complexity"
        - policy : "easy_first"

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


def rerank_hard_first(result: RankResult) -> RankResult:
    """
    Reverse ranking order to put hard samples first.

    Takes a RankResult (expected to be in easy_first order) and reverses
    the indices to produce hard_first order.

    Parameters
    ----------
    result : RankResult
        Ranking result, typically in easy_first order.

    Returns
    -------
    RankResult
        Dictionary containing:
        - indices : Reversed indices (hard samples first)
        - scores : Scores in original order (unchanged if present)
        - method : Same as input
        - policy : "hard_first"

    Examples
    --------
    >>> from dataeval.core import rank_knn, rerank_hard_first
    >>> import numpy as np
    >>> embeddings = np.random.rand(100, 64).astype(np.float32)
    >>> result = rank_knn(embeddings, k=5)
    >>> result = rerank_hard_first(result)
    """
    return RankResult(
        indices=result["indices"][::-1],
        scores=result["scores"],
        method=result["method"],
        policy="hard_first",
    )


def rerank_stratified(
    result: RankResult,
    num_bins: int = 50,
) -> RankResult:
    """
    Rerank by stratified sampling across score bins.

    Takes a RankResult (expected to be in easy_first order) and applies
    stratified sampling to balance selection across score bins. This
    encourages diversity by de-weighting samples with similar scores.

    The output is in hard_first order to maintain priority while balancing.

    Parameters
    ----------
    result : RankResult
        Ranking result with scores (must be from rank_knn or rank_kmeans_distance).
    num_bins : int, default 50
        Number of bins for stratification.

    Returns
    -------
    RankResult
        Dictionary containing:
        - indices : Reranked indices in hard_first order
        - scores : Scores in original order (unchanged)
        - method : Same as input
        - policy : "stratified"

    Raises
    ------
    ValueError
        If result does not contain scores (e.g., from rank_kmeans_complexity).

    Examples
    --------
    >>> from dataeval.core import rank_knn, rerank_stratified
    >>> import numpy as np
    >>> embeddings = np.random.rand(100, 64).astype(np.float32)
    >>> result = rank_knn(embeddings, k=5)
    >>> result = rerank_stratified(result, num_bins=20)
    """
    if result["scores"] is None:
        raise ValueError(
            "Ranking scores are necessary for stratified reranking. Use rank_knn or rank_kmeans_distance methods."
        )

    # Reverse to hard_first order for reranking (as expected by _stratified_rerank)
    indices_hard = result["indices"][::-1]
    scores_hard = result["scores"][::-1]

    # Apply stratified reranking
    reranked_indices = _stratified_rerank(scores_hard, indices_hard, num_bins)

    return RankResult(
        indices=reranked_indices,
        scores=result["scores"],
        method=result["method"],
        policy="stratified",
    )


def rerank_class_balance(
    result: RankResult,
    class_labels: NDArray[np.integer[Any]],
) -> RankResult:
    """
    Rerank to balance selection across class labels.

    Takes a RankResult (expected to be in easy_first order) and reranks
    to ensure balanced representation across classes while maintaining
    the priority order within each class.

    The output is in hard_first order to maintain priority while balancing.

    Parameters
    ----------
    result : RankResult
        Ranking result in any order.
    class_labels : NDArray[np.integer]
        Class label for each sample in the original dataset.

    Returns
    -------
    RankResult
        Dictionary containing:
        - indices : Reranked indices in hard_first order with class balance
        - scores : Scores in original order (unchanged if present)
        - method : Same as input
        - policy : "class_balance"

    Examples
    --------
    >>> from dataeval.core import rank_knn, rerank_class_balance
    >>> import numpy as np
    >>> embeddings = np.random.rand(100, 64).astype(np.float32)
    >>> labels = np.random.randint(0, 3, size=100)
    >>> result = rank_knn(embeddings, k=5)
    >>> result = rerank_class_balance(result, class_labels=labels)
    """
    # Reverse to hard_first order for reranking (as expected by _select_ordered_by_label)
    indices_hard = result["indices"][::-1]

    # Apply class balance reranking
    indices_reversed = _select_ordered_by_label(class_labels[indices_hard]).astype(np.int32)
    n = len(indices_reversed)
    reranked_indices = (n - 1 - indices_reversed).astype(np.intp)

    return RankResult(
        indices=reranked_indices,
        scores=result["scores"],
        method=result["method"],
        policy="class_balance",
    )
