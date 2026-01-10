"""
Core re-ranking algorithms for embeddings.

This module provides pure algorithmic implementations that operate on
numpy arrays, independent of dataset abstractions.
"""

__all__ = []

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.core._rank import RankResult

_logger = logging.getLogger(__name__)


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

        - indices: NDArray[np.intp] - Reversed indices (hard samples first)
        - scores: NDArray[np.float32] | None - Scores in original order (unchanged if present)
        - method: str - Same as input
        - policy: str - "hard_first"

    Examples
    --------
    >>> from dataeval.core import rank_knn, rerank_hard_first
    >>> import numpy as np
    >>> embeddings = np.random.rand(100, 64).astype(np.float32)
    >>> result = rank_knn(embeddings, k=5)
    >>> result = rerank_hard_first(result)
    """
    _logger.info(
        "Reranking to hard_first: method=%s, current_policy=%s, %d samples",
        result["method"],
        result["policy"],
        len(result["indices"]),
    )
    return RankResult(
        indices=result["indices"][::-1],
        scores=result["scores"],
        method=result["method"],
        policy="hard_first",
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
    _logger.debug("Stratified reranking with num_bins=%d", num_bins)
    mn, mx = _compute_bin_extents(scores)
    _logger.debug("Score range: min=%.4f, max=%.4f", mn, mx)
    bin_edges = np.linspace(mn, mx, num=num_bins + 1, endpoint=True)
    bin_label = np.digitize(scores, bin_edges)
    unique_bins = len(np.unique(bin_label))
    _logger.debug("Samples distributed across %d unique bins (out of %d)", unique_bins, num_bins)
    srt_inds = _select_ordered_by_label(bin_label)
    return indices[srt_inds].astype(np.intp)


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

        - indices: NDArray[np.intp] - Reranked indices in hard_first order
        - scores: NDArray[np.float32] | None - Scores in original order (unchanged)
        - method: str - Same as input
        - policy: str - "stratified"

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
    _logger.info(
        "Starting stratified reranking: method=%s, current_policy=%s, num_bins=%d, %d samples",
        result["method"],
        result["policy"],
        num_bins,
        len(result["indices"]),
    )

    if result["scores"] is None:
        raise ValueError(
            "Ranking scores are necessary for stratified reranking. Use rank_knn or rank_kmeans_distance methods."
        )

    # Reverse to hard_first order for reranking (as expected by _stratified_rerank)
    indices_hard = result["indices"][::-1]
    scores_hard = result["scores"][::-1]

    # Apply stratified reranking
    reranked_indices = _stratified_rerank(scores_hard, indices_hard, num_bins)

    _logger.info("Stratified reranking complete: %d samples reranked", len(reranked_indices))

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

        - indices: NDArray[np.intp] - Reranked indices in hard_first order with class balance
        - scores: NDArray[np.float32] | None - Scores in original order (unchanged if present)
        - method: str - Same as input
        - policy: str - "class_balance"

    Examples
    --------
    >>> from dataeval.core import rank_knn, rerank_class_balance
    >>> import numpy as np
    >>> embeddings = np.random.rand(100, 64).astype(np.float32)
    >>> labels = np.random.randint(0, 3, size=100)
    >>> result = rank_knn(embeddings, k=5)
    >>> result = rerank_class_balance(result, class_labels=labels)
    """
    num_classes = len(np.unique(class_labels))
    _logger.info(
        "Starting class balance reranking: method=%s, current_policy=%s, %d classes, %d samples",
        result["method"],
        result["policy"],
        num_classes,
        len(result["indices"]),
    )

    # Reverse to hard_first order for reranking (as expected by _select_ordered_by_label)
    indices_hard = result["indices"][::-1]

    # Apply class balance reranking
    indices_reversed = _select_ordered_by_label(class_labels[indices_hard]).astype(np.int32)
    n = len(indices_reversed)
    reranked_indices = (n - 1 - indices_reversed).astype(np.intp)

    _logger.info("Class balance reranking complete: %d samples reranked", len(reranked_indices))

    return RankResult(
        indices=reranked_indices,
        scores=result["scores"],
        method=result["method"],
        policy="class_balance",
    )
