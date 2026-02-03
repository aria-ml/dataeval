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
    Results from ranking data according to a specified method.

    Attributes
    ----------
    indices : NDArray[np.intp]
        Indices that sort the data in order of 'easy-first' priority according to the specified method.
    scores : NDArray[np.float32] | None
        Ranking scores for each sample (only available for methods that compute scores: "knn",
        "kmeans_distance", "hdbscan_distance").
    """

    indices: NDArray[np.intp]
    scores: NDArray[np.float32] | None


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

    return RankResult(indices=indices, scores=sorter.scores)


def rank_knn(
    embeddings: NDArray[np.floating[Any]],
    k: int | None = None,
    reference: NDArray[np.floating[Any]] | None = None,
) -> RankResult:
    """
    Rank samples using k-nearest neighbors distance.

    Computes the mean distance to k nearest neighbors for each sample and
    ranks them in easy-first order (low distance = prototypical samples).

    Parameters
    ----------
    embeddings : NDArray[np.floating]
        Embedding vectors to rank, shape (n_samples, n_features).
    k : int | None, default None
        Number of nearest neighbors. If None, uses sqrt(n_samples).
    reference : NDArray[np.floating] | None, default None
        Reference embeddings for comparative ranking. If provided, samples
        are ranked by distance to the reference set rather than to each other.

    Returns
    -------
    RankResult
        - indices: NDArray[np.intp] - Indices sorted in easy-first order
        - scores: NDArray[np.float32] - KNN distance scores (in index order)

    Raises
    ------
    ValueError
        If k is invalid (>= dataset size or negative).

    Examples
    --------
    >>> from dataeval.core import rank_knn
    >>> import numpy as np
    >>> embeddings = np.random.rand(100, 64).astype(np.float32)
    >>> result = rank_knn(embeddings, k=5)

    With reference embeddings:

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

        - indices: NDArray[np.intp] - Indices sorted in easy-first order
        - scores: NDArray[np.float32] | None - Distance to cluster center for each sample

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

    Note: This method does not produce scores, so `.stratified()` cannot
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
        Result with:

        - indices: NDArray[np.intp] - Indices sorted in easy-first order
        - scores: None (this method does not produce scores)

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

    Note: This method does not produce scores, so `.stratified()` cannot
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
        Result with:

        - indices: NDArray[np.intp] - Indices sorted in easy-first order
        - scores: None (this method does not produce scores)

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

    _logger.debug("Balancing selection across %d labels/groups for %d samples", n_cls, num_samp)

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


def _stratified_rerank(
    scores: NDArray[np.floating[Any]],
    indices: NDArray[np.integer[Any]],
    num_bins: int = 50,
) -> NDArray[np.intp]:
    """
    Re-rank samples by sampling uniformly over binned scores.

    This de-weights selection of samples with similar scores.

    Parameters
    ----------
    scores : NDArray[np.floating]
        Ranking scores sorted in order of priority (e.g. easy-first or hard-first).
    indices : NDArray[np.integer]
        Indices to be re-sorted according to stratified sampling of scores.
        Indices must match the order of `scores`.
    num_bins : int, default 50
        Number of bins for stratification.

    Returns
    -------
    NDArray[np.intp]
        Re-ranked indices respecting the original priority order.
    """
    _logger.debug("Stratified reranking with num_bins=%d", num_bins)
    mn, mx = _compute_bin_extents(scores)
    _logger.debug("Score range: min=%.4f, max=%.4f", mn, mx)

    bin_edges = np.linspace(mn, mx, num=num_bins + 1, endpoint=True)
    bin_label = np.digitize(scores, bin_edges)

    unique_bins = len(np.unique(bin_label))
    _logger.debug("Samples distributed across %d unique bins (out of %d)", unique_bins, num_bins)

    srt_inds = _select_ordered_by_label(bin_label)
    return indices[srt_inds].astype(np.intp)


def rank_result_stratified(
    result: RankResult,
    num_bins: int = 50,
) -> NDArray[np.intp]:
    """
    Transform RankResult indices using stratified sampling.

    Takes a RankResult and applies stratified sampling to balance selection across score bins.

    Parameters
    ----------
    result : RankResult
        Ranking result including indices and scores.
    num_bins : int, default 50
        Number of bins for stratification.

    Returns
    -------
    NDArray[np.intp]
        Reordered indices in easy_first order with stratified sampling applied.

    Raises
    ------
    ValueError
        If result does not contain scores.
    """
    if result["scores"] is None:
        raise ValueError(
            "Ranking scores are necessary for stratified policy. "
            "Use rank_knn, rank_kmeans_distance, or rank_hdbscan_distance methods."
        )

    _logger.debug(
        "Computing stratified indices: num_bins=%d, %d samples",
        num_bins,
        len(result["indices"]),
    )

    # Pass easy_first indices/scores directly.
    # The internal logic preserves the priority order of the input.
    return _stratified_rerank(result["scores"], result["indices"], num_bins)


def rank_result_class_balanced(
    result: RankResult,
    class_labels: NDArray[np.integer[Any]],
) -> NDArray[np.intp]:
    """
    Transform RankResult indices using class-balanced selection.

    Takes a RankResult and reranks to ensure balanced representation across classes.

    Parameters
    ----------
    result : RankResult
        Ranking result (assumed easy_first order).
    class_labels : NDArray[np.integer]
        Class label for each sample in the original dataset.

    Returns
    -------
    NDArray[np.intp]
        Reordered indices in easy_first order with class balance applied.
    """
    indices = result["indices"]
    num_classes = len(np.unique(class_labels))

    _logger.debug(
        "Computing class_balanced indices: %d classes, %d samples",
        num_classes,
        len(indices),
    )

    # Select indices based on class labels, respecting the original (easy-first) order
    indices_reordered = _select_ordered_by_label(class_labels[indices])

    return indices[indices_reordered]
