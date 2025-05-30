"""
This module contains the implementation of HP :term:`divergence<Divergence>`
using the Fast Nearest Neighbor and Minimum Spanning Tree algorithms
"""

from __future__ import annotations

__all__ = []

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from dataeval.outputs import DivergenceOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import Array
from dataeval.utils._array import ensure_embeddings
from dataeval.utils._method import get_method
from dataeval.utils._mst import compute_neighbors, minimum_spanning_tree


def divergence_mst(data: NDArray[np.float64], labels: NDArray[np.int_]) -> int:
    """
    Calculates the estimated label errors based on the minimum spanning tree

    Parameters
    ----------
    data : NDArray, shape - (N, ... )
        Input images to be grouped
    labels : NDArray
        Corresponding labels for each data point

    Returns
    -------
    int
        Number of label errors when creating the minimum spanning tree
    """
    mst = minimum_spanning_tree(data).toarray()
    edgelist = np.transpose(np.nonzero(mst))
    return np.sum(labels[edgelist[:, 0]] != labels[edgelist[:, 1]])


def divergence_fnn(data: NDArray[np.float64], labels: NDArray[np.int_]) -> int:
    """
    Calculates the estimated label errors based on their nearest neighbors.

    Parameters
    ----------
    data : NDArray, shape - (N, ... )
        Input images to be grouped
    labels : NDArray
        Corresponding labels for each data point

    Returns
    -------
    int
        Number of label errors when finding nearest neighbors
    """
    nn_indices = compute_neighbors(data, data)
    return np.sum(np.abs(labels[nn_indices] - labels))


_DIVERGENCE_FN_MAP = {"FNN": divergence_fnn, "MST": divergence_mst}


@set_metadata
def divergence(emb_a: Array, emb_b: Array, method: Literal["FNN", "MST"] = "FNN") -> DivergenceOutput:
    """
    Calculates the :term:`divergence` and any errors between the datasets.

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

    Warning
    -------
        MST is very slow in this implementation, this is unlike matlab where
        they have comparable speeds
        Overall, MST takes ~25x LONGER!!
        Source of slowdown:
        conversion to and from CSR format adds ~10% of the time diff between
        1nn and scipy mst function the remaining 90%

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
    labels = np.vstack([np.zeros([N, 1], dtype=np.int_), np.ones([M, 1], dtype=np.int_)])

    errors = div_fn(stacked_data, labels)
    dp = max(0.0, 1 - ((M + N) / (2 * M * N)) * errors)
    return DivergenceOutput(dp, errors)
