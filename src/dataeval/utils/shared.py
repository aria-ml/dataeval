from __future__ import annotations

__all__ = []

import sys
from typing import Any, Callable, Literal, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree as mst
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

from dataeval.interop import as_numpy

EPSILON = 1e-5
HASH_SIZE = 8
MAX_FACTOR = 4


P = ParamSpec("P")
R = TypeVar("R")


def get_method(method_map: dict[str, Callable[P, R]], method: str) -> Callable[P, R]:
    if method not in method_map:
        raise ValueError(f"Specified method {method} is not a valid method: {method_map}.")
    return method_map[method]


def flatten(array: ArrayLike) -> NDArray[Any]:
    """
    Flattens input array from (N, ... ) to (N, -1) where all samples N have all data in their last dimension

    Parameters
    ----------
    X : NDArray, shape - (N, ... )
        Input array

    Returns
    -------
    NDArray, shape - (N, -1)
    """
    nparr = as_numpy(array)
    return nparr.reshape((nparr.shape[0], -1))


def minimum_spanning_tree(X: NDArray[Any]) -> Any:
    """
    Returns the minimum spanning tree from a :term:`NumPy` image array.

    Parameters
    ----------
    X : NDArray
        Numpy image array

    Returns
    -------
        Data representing the minimum spanning tree
    """
    # All features belong on second dimension
    X = flatten(X)
    # We add a small constant to the distance matrix to ensure scipy interprets
    # the input graph as fully-connected.
    dense_eudist = squareform(pdist(X)) + EPSILON
    eudist_csr = csr_matrix(dense_eudist)
    return mst(eudist_csr)


def get_classes_counts(labels: NDArray[np.int_]) -> tuple[int, int]:
    """
    Returns the classes and counts of from an array of labels

    Parameters
    ----------
    label : NDArray
        Numpy labels array

    Returns
    -------
        Classes and counts

    Raises
    ------
    ValueError
        If the number of unique classes is less than 2
    """
    classes, counts = np.unique(labels, return_counts=True)
    M = len(classes)
    if M < 2:
        raise ValueError("Label vector contains less than 2 classes!")
    N = np.sum(counts).astype(int)
    return M, N


def compute_neighbors(
    A: NDArray[Any],
    B: NDArray[Any],
    k: int = 1,
    algorithm: Literal["auto", "ball_tree", "kd_tree"] = "auto",
) -> NDArray[Any]:
    """
    For each sample in A, compute the nearest neighbor in B

    Parameters
    ----------
    A, B : NDArray
        The n_samples and n_features respectively
    k : int
        The number of neighbors to find
    algorithm : Literal
        Tree method for nearest neighbor (auto, ball_tree or kd_tree)

    Note
    ----
        Do not use kd_tree if n_features > 20

    Returns
    -------
    List:
        Closest points to each point in A and B

    Raises
    ------
    ValueError
        If algorithm is not "auto", "ball_tree", or "kd_tree"

    See Also
    --------
    sklearn.neighbors.NearestNeighbors
    """

    if k < 1:
        raise ValueError("k must be >= 1")
    if algorithm not in ["auto", "ball_tree", "kd_tree"]:
        raise ValueError("Algorithm must be 'auto', 'ball_tree', or 'kd_tree'")

    A = flatten(A)
    B = flatten(B)

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm=algorithm).fit(B)
    nns = nbrs.kneighbors(A)[1]
    nns = nns[:, 1:].squeeze()

    return nns
