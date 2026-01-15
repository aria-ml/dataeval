"""
Core ranking algorithms for embeddings.

This module provides pure algorithmic implementations that operate on
numpy arrays, independent of dataset abstractions.
"""

__all__ = []

import logging
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

from dataeval.config import EPSILON
from dataeval.core._clusterer import _SORTER_MAP

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
        that compute scores: "knn", "kmeans_distance", "hdbscan_distance").
        Scores are ordered according to the original data order, not the ranked order.
    method : "knn", "kmeans_distance", "kmeans_complexity", "hdbscan_distance", or "hdbscan_complexity"
        The ranking method that was used.
    policy : "hard_first", "easy_first", "stratified" or "class_balance"
        The selection policy that was applied.
    """

    indices: NDArray[np.intp]
    scores: NDArray[np.float32] | None
    method: Literal["knn", "kmeans_distance", "kmeans_complexity", "hdbscan_distance", "hdbscan_complexity"]
    policy: Literal["hard_first", "easy_first", "stratified", "class_balance"]


def _normalize(embeddings: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Normalize embeddings by maximum L2 norm."""
    emb = embeddings.copy()
    emb /= max(np.max(np.linalg.norm(emb, axis=1)), EPSILON)
    return emb


def _rank_base(
    embeddings: NDArray[np.floating[Any]],
    *,
    method: Literal["knn", "kmeans_distance", "kmeans_complexity", "hdbscan_distance", "hdbscan_complexity"],
    k: int | None = None,
    c: int | None = None,
    n_init: int | Literal["auto"] = "auto",
    max_cluster_size: int | None = None,
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

    # Determine which algorithm to use for distance/complexity sorters
    algorithm = "hdbscan" if method in ["hdbscan_distance", "hdbscan_complexity"] else "kmeans"

    # Create sorter based on method
    sorter_cls = _SORTER_MAP[method]
    sorter = (
        sorter_cls(len(embeddings), k=k)
        if method == "knn"
        else sorter_cls(len(embeddings), algorithm=algorithm, c=c, n_init=n_init, max_cluster_size=max_cluster_size)
    )

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


def rank_hdbscan_distance(
    embeddings: NDArray[np.floating[Any]],
    c: int | None = None,
    max_cluster_size: int | None = None,
    reference: NDArray[np.floating[Any]] | None = None,
) -> RankResult:
    """
    Rank samples using distance to HDBSCAN cluster centers.

    Clusters embeddings using HDBSCAN and ranks by distance to assigned cluster
    centers. Returns samples in easy-first order (low distance = prototypical).

    Parameters
    ----------
    embeddings : NDArray[np.floating]
        Embedding vectors to rank, shape (n_samples, n_features).
    c : int | None, default None
        Expected number of clusters (used as hint for min_cluster_size).
        If None, uses sqrt(n_samples).
    max_cluster_size : int | None, default None
        Maximum size limit for identified clusters.
    reference : NDArray[np.floating] | None, default None
        Reference embeddings for comparative ranking. If provided, samples
        are ranked relative to the reference set rather than themselves.

    Returns
    -------
    RankResult
        Dictionary containing:

        - indices: NDArray[np.intp] - Ranked indices in easy-first order
        - scores: NDArray[np.float32] | None - Distance to cluster center for each sample
        - method: str - "hdbscan_distance"
        - policy: str - "easy_first"

    Examples
    --------
    >>> from dataeval.core import rank_hdbscan_distance
    >>> import numpy as np
    >>> embeddings = np.random.rand(100, 64).astype(np.float32)
    >>> result = rank_hdbscan_distance(embeddings, c=10)
    """
    return _rank_base(
        embeddings, method="hdbscan_distance", c=c, max_cluster_size=max_cluster_size, reference=reference
    )


def rank_hdbscan_complexity(
    embeddings: NDArray[np.floating[Any]],
    c: int | None = None,
    max_cluster_size: int | None = None,
    reference: NDArray[np.floating[Any]] | None = None,
) -> RankResult:
    """
    Rank samples using HDBSCAN cluster complexity weighting.

    Uses a weighted sampling strategy based on intra-cluster and inter-cluster
    distances from HDBSCAN clustering. Returns samples in easy-first order.

    Note: This method does not produce scores, so rerank_stratified() cannot
    be used with results from this function.

    Parameters
    ----------
    embeddings : NDArray[np.floating]
        Embedding vectors to rank, shape (n_samples, n_features).
    c : int | None, default None
        Expected number of clusters (used as hint for min_cluster_size).
        If None, uses sqrt(n_samples).
    max_cluster_size : int | None, default None
        Maximum size limit for identified clusters.
    reference : NDArray[np.floating] | None, default None
        Reference embeddings for comparative ranking. If provided, samples
        are ranked relative to the reference set rather than themselves.

    Returns
    -------
    RankResult
        Dictionary containing:

        - indices: NDArray[np.intp] - Ranked indices in easy-first order
        - scores: None (this method does not produce scores)
        - method: str - "hdbscan_complexity"
        - policy: str - "easy_first"

    Examples
    --------
    >>> from dataeval.core import rank_hdbscan_complexity
    >>> import numpy as np
    >>> embeddings = np.random.rand(100, 64).astype(np.float32)
    >>> result = rank_hdbscan_complexity(embeddings, c=10)
    """
    return _rank_base(
        embeddings, method="hdbscan_complexity", c=c, max_cluster_size=max_cluster_size, reference=reference
    )
