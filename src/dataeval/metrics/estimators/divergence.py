"""
This module contains the implementation of HP :term:`divergence<Divergence>`
using the Fast Nearest Neighbor and Minimum Spanning Tree algorithms
"""

from __future__ import annotations

__all__ = ["DivergenceOutput", "divergence"]

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval.interop import as_numpy
from dataeval.output import OutputMetadata, set_metadata
from dataeval.utils.shared import compute_neighbors, get_method, minimum_spanning_tree


@dataclass(frozen=True)
class DivergenceOutput(OutputMetadata):
    """
    Output class for :func:`divergence` estimator metric

    Attributes
    ----------
    divergence : float
        :term:`Divergence` value calculated between 2 datasets ranging between 0.0 and 1.0
    errors : int
        The number of differing edges between the datasets
    """

    divergence: float
    errors: int


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
    errors = np.sum(labels[edgelist[:, 0]] != labels[edgelist[:, 1]])
    return errors


def divergence_fnn(data: NDArray[np.float64], labels: NDArray[np.int_]) -> int:
    """
    Calculates the estimated label errors based on their nearest neighbors

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
    errors = np.sum(np.abs(labels[nn_indices] - labels))
    return errors


@set_metadata()
def divergence(data_a: ArrayLike, data_b: ArrayLike, method: Literal["FNN", "MST"] = "FNN") -> DivergenceOutput:
    """
    Calculates the :term`divergence` and any errors between the datasets

    Parameters
    ----------
    data_a : ArrayLike, shape - (N, P)
        A dataset in an ArrayLike format to compare.
        Function expects the data to have 2 dimensions, N number of observations in a P-dimensionial space.
    data_b : ArrayLike, shape - (N, P)
        A dataset in an ArrayLike format to compare.
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
    div_fn = get_method({"FNN": divergence_fnn, "MST": divergence_mst}, method)
    a = as_numpy(data_a)
    b = as_numpy(data_b)
    N = a.shape[0]
    M = b.shape[0]

    stacked_data = np.vstack((a, b))
    labels = np.vstack([np.zeros([N, 1], dtype=np.int_), np.ones([M, 1], dtype=np.int_)])

    errors = div_fn(stacked_data, labels)
    dp = max(0.0, 1 - ((M + N) / (2 * M * N)) * errors)
    return DivergenceOutput(dp, errors)
