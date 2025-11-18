"""
This module contains the implementation of HP :term:`divergence<Divergence>`
using the Fast Nearest Neighbor and Minimum Spanning Tree algorithms
"""

from __future__ import annotations

__all__ = []

from collections.abc import Callable
from typing import TypedDict

import numpy as np

from dataeval.types import ArrayND
from dataeval.utils._array import as_numpy


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
    a = as_numpy(emb_a, dtype=np.float64)
    b = as_numpy(emb_b, dtype=np.float64)
    N = a.shape[0]
    M = b.shape[0]

    stacked_data = np.vstack((a, b))
    labels = np.zeros((N + M,), dtype=np.intp)
    labels[N:] = 1

    errors = error_fn(stacked_data, labels)
    dp = max(0.0, 1 - ((N + M) / (2 * N * M)) * errors)

    return DivergenceResult(divergence=dp, errors=errors)


def _compute_mst_errors(embeddings: ArrayND[float], labels: ArrayND[int]) -> int:
    from dataeval.core._mst import minimum_spanning_tree

    mst_result = minimum_spanning_tree(embeddings)
    source, target = mst_result["source"], mst_result["target"]
    return np.sum(labels[source] != labels[target])


def _compute_fnn_errors(embeddings: ArrayND[float], labels: ArrayND[int]) -> int:
    from dataeval.core._mst import compute_neighbors

    nn_indices = compute_neighbors(embeddings, embeddings)
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
        Dictionary containing 'divergence' and 'errors' (number of cross-label edges)

    Examples
    --------
    >>> divergence_mst(datasetA, datasetB)
    {'divergence': 0.0, 'errors': 50}
    """
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
        Dictionary containing 'divergence' and 'errors' (number of label disagreements)

    Examples
    --------
    >>> divergence_fnn(datasetA, datasetB)
    {'divergence': 0.28, 'errors': 36}
    """
    return _compute_divergence(emb_a, emb_b, _compute_fnn_errors)
