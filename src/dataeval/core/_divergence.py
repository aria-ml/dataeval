"""
This module contains the implementation of HP :term:`divergence<Divergence>`
using the Fast Nearest Neighbor and Minimum Spanning Tree algorithms
"""

__all__ = []

import logging
from collections.abc import Callable
from typing import TypedDict

import numpy as np

from dataeval.types import ArrayND
from dataeval.utils.arrays import as_numpy

_logger = logging.getLogger(__name__)


class DivergenceResult(TypedDict):
    """
    Result mapping for :func:`.divergence` estimator metric.

    Attributes
    ----------
    divergence : float
        :term:`Divergence` value calculated between 2 datasets ranging between 0.0 and 1.0
    errors : int
        The number of errors between the datasets
    """

    divergence: float
    errors: int


def _compute_divergence(
    emb_a: ArrayND[float],
    emb_b: ArrayND[float],
    error_fn: Callable[[ArrayND[float], ArrayND[int]], int],
) -> DivergenceResult:
    """Generic divergence computation using a custom error function."""
    _logger.debug("Computing divergence using error function: %s", error_fn.__name__)

    a = as_numpy(emb_a, dtype=np.float64)
    b = as_numpy(emb_b, dtype=np.float64)
    N = a.shape[0]
    M = b.shape[0]

    _logger.debug("Dataset A shape: %s, Dataset B shape: %s", a.shape, b.shape)

    stacked_data = np.vstack((a, b))
    labels = np.zeros((N + M,), dtype=np.intp)
    labels[N:] = 1

    errors = error_fn(stacked_data, labels)
    # A MST between two completely separate distributions results in 1 error
    if error_fn == _compute_mst_errors:
        errors -= 1
    dp = max(0.0, 1 - ((N + M) / (2 * N * M)) * errors)

    _logger.info("Divergence computation complete: divergence=%.4f, errors=%d", dp, errors)

    return DivergenceResult(divergence=dp, errors=errors)


def _compute_mst_errors(embeddings: ArrayND[float], labels: ArrayND[int]) -> int:
    from dataeval.core._mst import minimum_spanning_tree

    mst_result = minimum_spanning_tree(embeddings)
    source, target = mst_result["source"], mst_result["target"]
    return np.sum(labels[source] != labels[target])


def _compute_fnn_errors(embeddings: ArrayND[float], labels: ArrayND[int]) -> int:
    from dataeval.core._mst import compute_neighbors

    nn_indices = compute_neighbors(embeddings)
    return np.sum(labels[nn_indices] != labels)


def divergence_mst(emb_a: ArrayND[float], emb_b: ArrayND[float]) -> DivergenceResult:
    """
    Calculates the :term:`divergence` by counting the number of "between dataset" edges in the
    minimum spanning tree.

    Parameters
    ----------
    emb_a : ArrayLike, shape - (N, P)
        Image embeddings in an ArrayLike format to compare.
        Function expects the data to have 2 dimensions, N number of observations in a P-dimensional space.
    emb_b : ArrayLike, shape - (N, P)
        Image embeddings in an ArrayLike format to compare.
        Function expects the data to have 2 dimensions, N number of observations in a P-dimensional space.

    Returns
    -------
    DivergenceResult
        Mapping with keys:

        - divergence: float - The divergence value between 0.0 and 1.0
        - errors: int - The number of cross-label edges

    Examples
    --------
    Return divergence of two datasets (0-no divergence, 1-complete divergence)

    >>> import sklearn.datasets as dsets
    >>> from dataeval.core import divergence_mst
    >>> datasetA = dsets.make_blobs(
    ...     n_samples=50, centers=np.array([(-1, -1), (1, 1)]), cluster_std=0.3, random_state=712
    ... )[0]
    >>> datasetB = (
    ...     dsets.make_blobs(n_samples=50, centers=np.array([(-0.5, -0.5), (1, 1)]), cluster_std=0.3, random_state=712)[
    ...         0
    ...     ]
    ...     + 0.05
    ... )
    >>> datasetC = dsets.make_blobs(
    ...     n_samples=50, centers=np.array([(-0.5, 0.5), (1, -1)]), cluster_std=0.3, random_state=712
    ... )[0]

    Overlapping datasets - divergence == 0:

    >>> divergence_mst(datasetA, datasetB)
    {'divergence': 0.040000000000000036, 'errors': 48}

    Completely separated datasets - divergence == 1:

    >>> divergence_mst(datasetA, datasetC)
    {'divergence': 0.96, 'errors': 2}
    """
    _logger.info("Starting divergence_mst calculation")
    return _compute_divergence(emb_a, emb_b, _compute_mst_errors)


def divergence_fnn(emb_a: ArrayND[float], emb_b: ArrayND[float]) -> DivergenceResult:
    """
    Calculates the :term:`divergence` by counting the label disagreements between nearest neighbors
    in the datasets.

    Parameters
    ----------
    emb_a : ArrayLike, shape - (N, P)
        Image embeddings in an ArrayLike format to compare.
        Function expects the data to have 2 dimensions, N number of observations in a P-dimensional space.
    emb_b : ArrayLike, shape - (N, P)
        Image embeddings in an ArrayLike format to compare.
        Function expects the data to have 2 dimensions, N number of observations in a P-dimensional space.

    Returns
    -------
    DivergenceResult
        Mapping with keys:

        - divergence: float - The divergence value between 0.0 and 1.0
        - errors: int - The number of label disagreements

    Examples
    --------
    Return divergence of two datasets (0-no divergence, 1-complete divergence)

    >>> import sklearn.datasets as dsets
    >>> from dataeval.core import divergence_fnn
    >>> datasetA = dsets.make_blobs(
    ...     n_samples=50, centers=np.array([(-1, -1), (1, 1)]), cluster_std=0.3, random_state=712
    ... )[0]
    >>> datasetB = (
    ...     dsets.make_blobs(n_samples=50, centers=np.array([(-0.5, -0.5), (1, 1)]), cluster_std=0.3, random_state=712)[
    ...         0
    ...     ]
    ...     + 0.05
    ... )
    >>> datasetC = dsets.make_blobs(
    ...     n_samples=50, centers=np.array([(-0.5, 0.5), (1, -1)]), cluster_std=0.3, random_state=712
    ... )[0]

    Overlapping datasets - divergence == 0:

    >>> divergence_fnn(datasetA, datasetB)
    {'divergence': 0.0, 'errors': 54}

    Completely separated datasets - divergence == 1:

    >>> divergence_fnn(datasetA, datasetC)
    {'divergence': 1.0, 'errors': 0}
    """
    _logger.info("Starting divergence_fnn calculation")
    return _compute_divergence(emb_a, emb_b, _compute_fnn_errors)
