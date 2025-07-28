"""
This module contains the implementation of HP :term:`divergence<Divergence>`
using the Fast Nearest Neighbor and Minimum Spanning Tree algorithms
"""

from __future__ import annotations

__all__ = []


import numpy as np
from numpy.typing import NDArray

from dataeval.utils._mst import compute_neighbors, minimum_spanning_tree_fast


def divergence_mst(data: NDArray[np.float64], labels: NDArray[np.int_]) -> int:
    """
    Counts the number of cross-label edges in the minimum spanning tree of
    data.

    Parameters
    ----------
    data : NDArray, shape - (N, ... )
        Input images to be grouped
    labels : NDArray
        Corresponding labels for each data point

    Returns
    -------
    int
        Number of cross-label edges in the minimum spanning tree of input data
    """

    rows, cols = minimum_spanning_tree_fast(data)  # get rows and cols directly
    return np.sum(labels[rows] != labels[cols])


def divergence_fnn(data: NDArray[np.float64], labels: NDArray[np.int_]) -> int:
    """
    Counts label disagreements between nearest neighbors in data.

    Parameters
    ----------
    data : NDArray, shape - (N, ... )
        Input images to be grouped
    labels : NDArray
        Corresponding labels for each data point

    Returns
    -------
    int
        Number of label disagreements between nearest neighbors
    """
    nn_indices = compute_neighbors(data, data)
    return np.sum(labels[nn_indices] != labels)
