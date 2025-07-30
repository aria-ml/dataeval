"""
This module contains the implementation of HP :term:`divergence<Divergence>`
using the Fast Nearest Neighbor and Minimum Spanning Tree algorithms
"""

from __future__ import annotations

__all__ = []

from typing import Literal

import numpy as np

from dataeval.core._divergence import divergence_fnn, divergence_mst
from dataeval.outputs import DivergenceOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import Array
from dataeval.utils._array import ensure_embeddings
from dataeval.utils._method import get_method

_DIVERGENCE_FN_MAP = {"FNN": divergence_fnn, "MST": divergence_mst}


@set_metadata
def divergence(emb_a: Array, emb_b: Array, method: Literal["FNN", "MST"] = "FNN") -> DivergenceOutput:
    """
    Calculates the :term:`divergence` by counting the number of "between dataset" edges in the
    minimum spanning tree.

    Parameters
    ----------
    emb_a : ArrayLike, shape - (N, P)
        Image embeddings in an ArrayLike format to compare.
        Function expects the data to have 2 dimensions, N number of observations in a P-dimensionial space.
    emb_b : ArrayLike, shape - (N, P)
        Image embeddings in an ArrayLike format to compare.
        Function expects the data to have 2 dimensions, N number of observations in a P-dimensionial space.
    method : Literal["MST, "FNN"], default "FNN"
        Method used to estimate dataset :term:`divergence<Divergence>`

    Returns
    -------
    DivergenceOutput
        The divergence value (0.0..1.0) and the number of differing edges between the datasets

    Note
    ----
    The divergence value indicates how similar the 2 datasets are
    with 0 indicating approximately identical data distributions.

    References
    ----------
    For more information about this divergence, its formal definition,
    and its associated estimators see https://arxiv.org/abs/1412.6534.

    Examples
    --------
    Evaluate the datasets:

    >>> divergence(datasetA, datasetB)
    DivergenceOutput(divergence=0.28, errors=36)
    """
    div_fn = get_method(_DIVERGENCE_FN_MAP, method)
    a = ensure_embeddings(emb_a, dtype=np.float64)
    b = ensure_embeddings(emb_b, dtype=np.float64)
    N = a.shape[0]
    M = b.shape[0]

    stacked_data = np.vstack((a, b))
    labels = np.vstack([np.zeros([N, 1], dtype=np.intp), np.ones([M, 1], dtype=np.intp)])

    errors = div_fn(stacked_data, labels)
    dp = max(0.0, 1 - ((M + N) / (2 * M * N)) * errors)
    return DivergenceOutput(dp, errors)
