from __future__ import annotations

__all__ = []

from typing import Any, Literal

from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree as mst
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

from dataeval.config import EPSILON
from dataeval.utils._array import flatten


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
    return nns[:, 1:].squeeze()
