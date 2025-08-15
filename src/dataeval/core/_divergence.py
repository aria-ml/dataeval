"""
This module contains the implementation of HP :term:`divergence<Divergence>`
using the Fast Nearest Neighbor and Minimum Spanning Tree algorithms
"""

from __future__ import annotations

__all__ = []


import numpy as np
from numpy.typing import NDArray


def divergence_mst(data: NDArray[np.float64], labels: NDArray[np.intp]) -> int:
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
    from dataeval.core._mst import minimum_spanning_tree

    rows, cols = minimum_spanning_tree(data)  # get rows and cols directly
    return np.sum(labels[rows] != labels[cols])


def divergence_fnn(data: NDArray[np.float64], labels: NDArray[np.intp]) -> int:
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
    from dataeval.core._mst import compute_neighbors

    nn_indices = compute_neighbors(data, data)
    return np.sum(labels[nn_indices] != labels)
