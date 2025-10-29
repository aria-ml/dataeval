"""
This module contains the implementation of HP :term:`divergence<Divergence>`
using the Fast Nearest Neighbor and Minimum Spanning Tree algorithms
"""

from __future__ import annotations

__all__ = []


import numpy as np

from dataeval.types import Array1D, ArrayND
from dataeval.utils._array import as_numpy


def divergence_mst(embeddings: ArrayND[float], class_labels: Array1D[int]) -> int:
    """
    Counts the number of cross-label edges in the minimum spanning tree of data.

    Parameters
    ----------
    embeddings : ArrayND[float]
        Input images/embeddings to be grouped. Can be an N dimensional list, or array-like object.
    class_labels : Array1D[int]
        Corresponding class labels for each data point. Can be a 1D list, or array-like object.

    Returns
    -------
    int
        Number of cross-label edges in the minimum spanning tree of input data
    """
    from dataeval.core._mst import minimum_spanning_tree

    mst_result = minimum_spanning_tree(embeddings)
    source, target = mst_result["source"], mst_result["target"]
    return np.sum(class_labels[source] != class_labels[target])


def divergence_fnn(embeddings: ArrayND[float], class_labels: Array1D[int]) -> int:
    """
    Counts label disagreements between nearest neighbors in data.

    Parameters
    ----------
    embeddings : ArrayND[float]
        Input images/embeddings to be grouped. Can be an N dimensional list, or array-like object.
    class_labels : Array1D[int]
        Corresponding class labels for each data point. Can be a 1D list, or array-like object.

    Returns
    -------
    int
        Number of label disagreements between nearest neighbors
    """
    from dataeval.core._mst import compute_neighbors

    embeddings_np = as_numpy(embeddings)
    class_labels_np = as_numpy(class_labels, dtype=np.intp, required_ndim=1)

    nn_indices = compute_neighbors(embeddings_np, embeddings_np)
    return np.sum(class_labels_np[nn_indices] != class_labels_np)
