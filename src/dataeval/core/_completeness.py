__all__ = []

import logging
from collections.abc import Sequence
from typing import TypedDict

import numpy as np
from sklearn.neighbors import NearestNeighbors

from dataeval.protocols import Array
from dataeval.utils._internal import ensure_embeddings

_logger = logging.getLogger(__name__)


class CompletenessResult(TypedDict):
    """
    Type definition for completeness output.

    Attributes
    ----------
    completeness : float
        Completeness score between 0 and 1, measuring dimensional utilization relative
        to full space.
    isotropy : float
        Isotropy score between 0 and 1, measuring dimensional utilization relative to
        space actually spanned by embeddings.
    nearest_neighbor_pairs : Sequence[tuple[int, int]]
        Sequence of tuples (i, j) representing point indices and their nearest neighbors,
        sorted by decreasing nearest neighbor distance. Each pair appears only once.
    nearest_neighbor_distances : Sequence[float]
        Distance between each pair in ``nearest_neighbor_pairs``, aligned 1:1 and sharing
        the same decreasing order. The small-distance tail flags near-duplicate samples.
    """

    completeness: float
    isotropy: float
    nearest_neighbor_pairs: Sequence[tuple[int, int]]
    nearest_neighbor_distances: Sequence[float]


def _get_effective_dim(x: Array, do_ranks: bool = True) -> float:
    n, d = x.shape

    # Get singular values, either with ranks or with raw data.
    if do_ranks:
        # Get normed ranks from 1/2n to (1-1/2n), then center them
        ranks = np.argsort(np.argsort(x, axis=0), axis=0)
        centered_normed_ranks = (ranks + 0.5) / n - 0.5

        # Use SVD directly on the centered normalized ranks
        # We only need singular values, not the full U and V matrices
        _, s, _ = np.linalg.svd(centered_normed_ranks, full_matrices=False)
    else:
        _, s, _ = np.linalg.svd(x - np.mean(x, axis=0, keepdims=True), full_matrices=False)

    # Convert singular values to eigenvalues of the covariance matrix
    # The eigenvalues of the covariance matrix are related to singular values by: eigenvalues = s^2/(n-1)
    eigenvalues = (s**2) / (n - 1)

    # Filter negligible eigenvalues
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Calculate entropy of normalized eigenvalues
    normalized_eigs = eigenvalues / np.sum(eigenvalues)
    entropy = -np.sum(normalized_eigs * np.log(normalized_eigs))

    # return effective dimensionality
    return np.exp(entropy)


def completeness(embeddings: Array) -> CompletenessResult:
    """
    Measure the dimensional utilization of :term:`embeddings<Embeddings>`.

    Completeness measures how effectively the data explores all available dimensions in
    its embedding space. This implementation uses a directional diversity approach based
    on eigenvalue entropy, which is more robust for high-dimensional data than traditional
    box-counting or neighbor-distance-based methods. The isotropy measure is similar, but
    measures directional diversity relative to the actual space spanned by the embeddings,
    rather than to the entire ambient space.


    Parameters
    ----------
    embeddings : Array
        Array of image :term:`embeddings<Embeddings>`, shape (n_samples, n_dimensions).
        Can be a 2D list, array-like object, or tensor.

    Returns
    -------
    CompletenessResult
        Mapping with keys:

        - completeness: float - Completeness score between 0 and 1
        - isotropy: float - Isotropy score between 0 and 1
        - nearest_neighbor_pairs: Sequence[tuple[int, int]] - Pairs of point indices
          and their nearest neighbors, sorted by decreasing distance

    Raises
    ------
    ValueError
        If embeddings are not 2D
    ValueError
        If embeddings have a zero dimension

    Examples
    --------
    Well-spread data across 3 dimensions:

    >>> rng = np.random.default_rng(42)
    >>> embeddings = rng.random((50, 3))
    >>> result = completeness(embeddings)
    >>> result["completeness"]
    0.9963684026790749
    >>> result["isotropy"]
    0.9865994134108708

    Single plane data across 3 dimensions:

    >>> directions = rng.normal(size=(2, 3))  # 2 random lines
    >>> directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    >>> t = np.random.uniform(0, 0.5, (len(directions), 25, 1))
    >>> embeddings = ([0.5] * 3 + t * directions[:, np.newaxis, :]).reshape(-1, 3)
    >>> result = completeness(embeddings)
    >>> result["completeness"]
    0.6001089325287554
    >>> result["isotropy"]
    0.40470070513943307

    Completeness can be less than isotropy:

    >>> X_low = rng.normal(size=(50, 2))
    >>> Q, _ = np.linalg.qr(rng.normal(size=(3, 2)))
    >>> embeddings = X_low @ Q.T
    >>> result = completeness(embeddings)
    >>> result["completeness"]  # penalized by unused ambient dimension
    0.6844547029590969
    >>> result["isotropy"]  # close to 1, isotropic within 2D subspace
    0.9869106459012913
    """
    _logger.info("Starting completeness calculation")

    # Ensure proper data format
    embeddings = ensure_embeddings(embeddings, dtype=np.float64, unit_interval=False)

    # Get data dimensions
    n, d = embeddings.shape
    _logger.debug("Embeddings shape: (%d samples, %d dimensions)", n, d)

    rank_effective_dim = _get_effective_dim(embeddings)
    raw_effective_dim = _get_effective_dim(embeddings, do_ranks=False)

    manifold_dim = np.linalg.matrix_rank(embeddings)

    # Calculate completeness as ratio of effective to total dimensions
    # This is equivalent to the Nth root of the "occupied fraction" in traditional approaches
    completeness_score = rank_effective_dim / d

    isotropy_score = raw_effective_dim / manifold_dim

    _logger.debug(
        "Rank effective dimensionality: %.2f, Completeness score: %.4f", rank_effective_dim, completeness_score
    )
    _logger.debug("Raw effective dimensionality: %.2f, Isotropy score: %.4f", rank_effective_dim, isotropy_score)

    # Compute nearest neighbor pairs using sklearn
    if n <= 1:
        _logger.warning("Only %d sample(s) provided, skipping nearest neighbor calculation", n)
        return CompletenessResult(
            completeness=float(completeness_score),
            isotropy=float(isotropy_score),
            nearest_neighbor_pairs=[],
            nearest_neighbor_distances=[],
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

    # Sort by distance descending and split into aligned pair / distance sequences
    pairs_with_distances.sort(key=lambda x: x[2], reverse=True)
    nearest_neighbor_pairs = [(i, j) for i, j, _ in pairs_with_distances]
    nearest_neighbor_distances = [float(dist) for _, _, dist in pairs_with_distances]

    _logger.info(
        "Completeness calculation complete: completeness=%.4f, %d nearest neighbor pairs identified",
        completeness_score,
        len(nearest_neighbor_pairs),
    )

    return CompletenessResult(
        completeness=float(completeness_score),
        isotropy=float(isotropy_score),
        nearest_neighbor_pairs=nearest_neighbor_pairs,
        nearest_neighbor_distances=nearest_neighbor_distances,
    )
