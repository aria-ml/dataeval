"""
This module contains the implementation of HP :term:`divergence<Divergence>`
using the Fast Nearest Neighbor and Minimum Spanning Tree algorithms
"""

from __future__ import annotations

__all__ = []


import numpy as np
from numpy.typing import NDArray

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
