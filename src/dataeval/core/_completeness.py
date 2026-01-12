__all__ = []

import logging
from collections.abc import Sequence
from typing import TypedDict

import numpy as np
from sklearn.neighbors import NearestNeighbors

from dataeval.protocols import Array
from dataeval.utils.arrays import ensure_embeddings

_logger = logging.getLogger(__name__)


class CompletenessResult(TypedDict):
    """
    Result mapping for :func:`.completeness` metric.

    Attributes
    ----------
    completeness : float
        Completeness score between 0 and 1, measuring dimensional utilization
    nearest_neighbor_pairs : Sequence[tuple[int, int]]
        Sequence of tuples (i, j) representing point indices and their nearest neighbors,
        sorted by decreasing nearest neighbor distance. Each pair appears only once.
    """

    completeness: float
    nearest_neighbor_pairs: Sequence[tuple[int, int]]


def completeness(embeddings: Array) -> CompletenessResult:
    """
    Measures the dimensional utilization of embeddings - how effectively the data
    explores all available dimensions in its embedding space.

    This implementation uses a directional diversity approach based on eigenvalue entropy,
    which is more robust for high-dimensional data than traditional box-counting or
    neighbor-distance-based methods.

    Parameters
    ----------
    embeddings : Array
        Array of embedding vectors, shape (n_samples, n_dimensions)

    Returns
    -------
    CompletenessResult
        Mapping with keys:

        - completeness: float - Completeness score between 0 and 1
        - nearest_neighbor_pairs: Sequence[tuple[int, int]] - Sequence of tuples (i, j)
          representing point indices and their nearest neighbors, sorted by decreasing
          nearest neighbor distance. Each pair appears only once.
    """
    _logger.info("Starting completeness calculation")

    # Ensure proper data format
    embeddings = ensure_embeddings(embeddings, dtype=np.float64, unit_interval=False)

    # Get data dimensions
    n, D = embeddings.shape
    _logger.debug("Embeddings shape: (%d samples, %d dimensions)", n, D)

    # Get normed ranks from 1/2n to (1-1/2n), then center them
    centered_normed_ranks = (np.argsort(np.argsort(embeddings, axis=0), axis=0) + 0.5) / n - 0.5

    # Use SVD directly on the centered normalized ranks
    # We only need singular values, not the full U and V matrices
    _, s, _ = np.linalg.svd(centered_normed_ranks, full_matrices=False)

    # Convert singular values to eigenvalues of the covariance matrix
    # The eigenvalues of the covariance matrix are related to singular values by: eigenvalues = s^2/(n-1)
    eigenvalues = (s**2) / (n - 1)

    # Filter negligible eigenvalues
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Calculate entropy of normalized eigenvalues
    normalized_eigs = eigenvalues / np.sum(eigenvalues)
    entropy = -np.sum(normalized_eigs * np.log(normalized_eigs))

    # Calculate effective dimensionality
    effective_dim = np.exp(entropy)

    # Calculate completeness as ratio of effective to total dimensions
    # This is equivalent to the Nth root of the "occupied fraction" in traditional approaches
    completeness_score = effective_dim / D

    _logger.debug("Effective dimensionality: %.2f, Completeness score: %.4f", effective_dim, completeness_score)

    # Compute nearest neighbor pairs using sklearn
    if n <= 1:
        _logger.warning("Only %d sample(s) provided, skipping nearest neighbor calculation", n)
        return CompletenessResult(
            completeness=float(completeness_score),
            nearest_neighbor_pairs=[],
        )

    # Fit nearest neighbors model (k=2 because first neighbor is always the point itself)
    nn_model = NearestNeighbors(n_neighbors=2, algorithm="auto")
    nn_model.fit(embeddings)

    # Get distances and indices for nearest neighbors
    distances, indices = nn_model.kneighbors(embeddings)

    # Extract the actual nearest neighbor (second column, index 1)
    nearest_neighbors = indices[:, 1]
    nearest_distances = distances[:, 1]

    # Create pairs, deduplicating so each pair appears only once
    seen_pairs = set()
    pairs_with_distances = []

    for i in range(n):
        j = nearest_neighbors[i]
        dist = nearest_distances[i]

        # Create canonical pair representation for deduplication (smaller index first)
        pair_key = tuple(sorted([i, j]))

        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            pairs_with_distances.append((i, j, dist))

    # Sort by distance descending and extract just the index pairs
    pairs_with_distances.sort(key=lambda x: x[2], reverse=True)
    nearest_neighbor_pairs = [(i, j) for i, j, _ in pairs_with_distances]

    _logger.info(
        "Completeness calculation complete: completeness=%.4f, %d nearest neighbor pairs identified",
        completeness_score,
        len(nearest_neighbor_pairs),
    )

    return CompletenessResult(
        completeness=float(completeness_score),
        nearest_neighbor_pairs=nearest_neighbor_pairs,
    )
