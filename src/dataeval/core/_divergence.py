"""
This module contains the implementation of HP :term:`divergence<Divergence>`
using the Fast Nearest Neighbor and Minimum Spanning Tree algorithms
"""

from __future__ import annotations

__all__ = []


import numpy as np

from dataeval.protocols import _1DArray, _2DArray


def divergence_mst(embeddings: _2DArray[float], class_labels: _1DArray[int]) -> int:
    """
    Counts the number of cross-label edges in the minimum spanning tree of data.

    Parameters
    ----------
    embeddings : _2DArray[float]
        Input images/embeddings to be grouped. Can be a 2D list, or array-like object.
    class_labels : _1DArray[int]
        Corresponding class labels for each data point. Can be a 1D list, or array-like object.

    Returns
    -------
    int
        Number of cross-label edges in the minimum spanning tree of input data
    """
    from dataeval.core._mst import minimum_spanning_tree

    rows, cols = minimum_spanning_tree(embeddings)  # get rows and cols directly
    return np.sum(class_labels[rows] != class_labels[cols])


def divergence_fnn(embeddings: _2DArray[float], class_labels: _1DArray[int]) -> int:
    """
    Counts label disagreements between nearest neighbors in data.

    Parameters
    ----------
    embeddings : _2DArray[float]
        Input images/embeddings to be grouped. Can be a 2D list, or array-like object.
    class_labels : _1DArray[int]
        Corresponding class labels for each data point. Can be a 1D list, or array-like object.

    Returns
    -------
    int
        Number of label disagreements between nearest neighbors
    """
    from dataeval.core._mst import compute_neighbors

    nn_indices = compute_neighbors(embeddings, embeddings)
    return np.sum(class_labels[nn_indices] != class_labels)
