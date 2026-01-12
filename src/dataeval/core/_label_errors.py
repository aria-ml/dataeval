__all__ = []

import logging
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from dataeval.core._mst import compute_neighbor_distances
from dataeval.utils.arrays import flatten_samples

_logger = logging.getLogger(__name__)


class LabelErrorResult(TypedDict):
    """
    Type definition for label error output.

    Attributes
    ----------
    errors : dict[int, tuple[int, list[int]]]
        Potentially mislabeled samples. Dictionary keys are sample index and
        values are a tuple (assigned_label, suggested_new_label).
    error_rank : NDArray[np.int64]
        Array of samples where the index corresponds to sample index and
        the value is the rank of the sample's score
    scores : NDArray[np.float32]
        Sample label scores
    """

    errors: dict[int, tuple[int, list[int]]]
    error_rank: NDArray[np.int64]
    scores: NDArray[np.float32]


def _compute_label_scores(
    embeddings: NDArray[Any], labels: NDArray[np.int64], k: int = 50
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """
    Computes label quality scores based on the ratio of intra-class to extra-class distances.

    For each sample, this calculates the ratio between the average distance to its nearest
    neighbors within the same class (intra) and the average distance to its nearest neighbors
    in any other class (extra). High scores indicate a sample is far from its class center
    and close to another class, suggesting a potential label error.

    Parameters
    ----------
    embeddings : NDArray
        The dataset embeddings with shape (n_samples, n_features).
    labels : NDArray[np.int64]
        Array of ground truth labels for the data with shape (n_samples,).
    k : int, optional
        Number of nearest neighbors to consider for distance calculations.
        Will be automatically adjusted to `min(k, min_class_count - 1)` if classes
        are small. Default is 50.

    Returns
    -------
    label_scores : NDArray[np.float32]
        Array of shape (n_samples,) containing the computed distance ratio for each sample.
        Values > 1.0 suggest the sample is closer to a different class than its own.
    other_potential_labels : NDArray[np.int64]
        Array of shape (n_samples, k) containing the labels of the nearest neighbors
        from different classes (extra-class neighbors) for each sample.
    """
    n_labels, label_counts = np.unique(labels, return_counts=True)
    k = min(k, label_counts.min() - 1)
    n_labels = n_labels.tolist()

    other_potential_labels = np.full((labels.size, k), -1, dtype=np.int64)
    label_scores = np.zeros(labels.size, dtype=np.float32)

    for lbl in n_labels:
        # Get current label points
        current_label_idx = np.nonzero(labels == lbl)[0]
        current_label_data = embeddings[current_label_idx]

        # Get distance to similar labels
        _, similar_distances = compute_neighbor_distances(current_label_data, k=k)

        # Get all other label points
        other_label_groups = [np.nonzero(labels == lbl2)[0] for lbl2 in n_labels if lbl2 != lbl]
        if not other_label_groups:
            continue

        other_label_idx = np.concatenate(other_label_groups)
        other_label_data = embeddings[other_label_idx]

        # Find nearest neighbors
        other_neighbors, other_distances = compute_neighbor_distances(
            other_label_data,
            current_label_data,
            k=k,
        )

        other_potential_labels[current_label_idx] = labels[other_label_idx[other_neighbors]]

        label_scores[current_label_idx] = similar_distances.mean(axis=1) / other_distances.mean(axis=1)

    return label_scores, other_potential_labels


def _suggest_labels(
    neighbor_labels: NDArray[np.int64], num_classes: int, min_confidence: float = 0.4, ambiguity_threshold: float = 0.2
) -> list[list[int]]:
    """
    Suggests alternative labels for samples based on rank-weighted voting of their neighbors.

    Uses a rank-weighted voting system where closer neighbors (lower index in `neighbor_labels`)
    have higher voting weight. If the highest confidence score is below `min_confidence`,
    no suggestion is made. If the margin between the top two candidates is smaller than
    `ambiguity_threshold`, both are suggested.

    Parameters
    ----------
    neighbor_labels : NDArray[np.int64]
        Array of shape (n_samples, k) containing the label indices of the nearest
        neighbors for each sample, sorted by distance (nearest first).
    num_classes : int
        The total number of unique classes in the dataset.
    min_confidence : float, optional
        Minimum vote share (0.0 to 1.0) required for the top candidate to be suggested.
        Default is 0.4.
    ambiguity_threshold : float, optional
        The confidence margin below which two labels are considered tied/ambiguous.
        Default is 0.2.

    Returns
    -------
    recommendations : list[list[int]]
        A list of length n_samples containing suggested labels.

        - `[label]`: Strong recommendation for a single label.
        - `[label_a, label_b]`: Ambiguous recommendation (tie).
        - `[]`: No recommendation (low confidence/noisy neighborhood).
    """
    n_samples, k = neighbor_labels.shape

    # Create distance weights (the closer, the higher the weight)
    weights = np.ceil(np.linspace(k, 1, k) / 3)

    # Label voting
    # Create one-hot matrix: (n_samples, k, num_classes)
    one_hot = np.zeros((n_samples, k, num_classes))

    sample_indices = np.arange(n_samples)[:, None]  # Column vector 0..N
    neighbor_indices = np.arange(k)[None, :]  # Row vector 0..k

    one_hot[sample_indices, neighbor_indices, neighbor_labels] = 1.0

    # Multiply by weights and sum across neighbors
    weighted_votes = (one_hot * weights[:, None]).sum(axis=1)

    # Normalize to get confidence scores
    confidence_scores = weighted_votes / weights.sum()

    # Sort the scores to find top 2 labels
    sorted_indices = np.argsort(confidence_scores, axis=1)

    recommendations = []
    for i in range(n_samples):
        # Get indices of the best and second best classes (argsort is ascending)
        top1_idx = sorted_indices[i, -1]
        top2_idx = sorted_indices[i, -2]

        top1_score = confidence_scores[i, top1_idx]
        top2_score = confidence_scores[i, top2_idx]

        # Decision logic
        # No clear winner, don't provide suggestion
        if top1_score < min_confidence:
            recommendations.append([])
            continue

        # Small difference between top 2, provide both
        if (top1_score - top2_score) < ambiguity_threshold:
            recommendations.append([int(top1_idx), int(top2_idx)])
        else:
            recommendations.append([int(top1_idx)])

    return recommendations


def label_errors(embeddings: NDArray[Any], labels: NDArray[np.int64], k: int = 50) -> LabelErrorResult:
    """
    Identifies potential label errors in a dataset using embedding geometry.

    Calculates an "Intra/Extra Class Distance Ratio" for every sample. Samples are flagged
    as errors if they are significantly closer to samples of a different class than to
    samples of their own class (score >= 1.0).

    Parameters
    ----------
    embeddings : NDArray
        Input feature embeddings (e.g., from DINO, ResNet) with shape (n_samples, n_features).
    labels : NDArray[np.int64]
        Ground truth labels corresponding to the embeddings, with shape (n_samples,).
    k : int, optional
        Number of neighbors to use for local density estimation. Default is 50.

    Returns
    -------
    LabelErrorResult
        A dictionary containing:

        - 'errors': Dict mapping sample indices to tuples of (original_label, [suggested_labels]).
          Only contains samples with a score >= 1.0.
        - 'error_rank': Array of sample indices sorted by likelihood of error (descending score).
        - 'scores': Array of raw distance ratio scores for all samples.
    """
    _logger.info("Starting label class distance ratio calculation with k=%d", k)

    embeddings_np = flatten_samples(embeddings)
    _logger.debug("Embeddings shape: %s", embeddings_np.shape)

    # Get label_scores and rank them
    label_scores, other_potential_labels = _compute_label_scores(embeddings_np, labels, k=k)
    error_rank = np.argsort(-label_scores)

    # Get label_scores >= 1
    errors = {}
    problem_labels = np.nonzero(label_scores >= 1)[0]
    potential_label = _suggest_labels(other_potential_labels, labels.max() + 1)
    for mislabel in problem_labels:
        errors[int(mislabel)] = (int(labels[mislabel]), potential_label[mislabel])

    return LabelErrorResult(errors=errors, error_rank=error_rank, scores=label_scores)
