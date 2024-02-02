from typing import Any, Literal, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree as mst
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

EPSILON = 1e-5


def pytorch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor to a NumPy array
    """
    if isinstance(tensor, np.ndarray):  # Already array, return
        return tensor
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Tensor is not of type torch.Tensor")

    x: np.ndarray = tensor.detach().cpu().numpy()
    return x


def numpy_to_pytorch(array: np.ndarray) -> torch.Tensor:
    """
    Converts a NumPy array to a PyTorch tensor
    """
    if isinstance(array, torch.Tensor):  # Already tensor, return
        return array
    if not isinstance(array, np.ndarray):
        raise TypeError("Array is not of type numpy.ndarray")
    x: torch.Tensor = torch.from_numpy(array.astype(np.float32))
    return x


def permute_to_torch(array: np.ndarray) -> torch.Tensor:
    """
    Converts and permutes a NumPy image array into a PyTorch image tensor.

    Parameters
    ----------
    array: np.ndarray
        Array containing image data in the format NHWC

    Returns
    -------
    torch.Tensor
        Tensor containing image data in the format NCHW
    """
    x = numpy_to_pytorch(array)
    x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    return x


def permute_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts and permutes a PyTorch image tensor into a NumPy image array.

    Does not permute if given np.ndarray

    Parameters
    ----------
    tensor: torch.Tensor
        Tensor containing image data in the format NCHW

    Returns
    -------
    np.ndarray
        Array containing image data in the format NHWC
    """
    x = tensor.permute(0, 2, 3, 1)
    x = pytorch_to_numpy(x)  # NCHW -> NHWC
    return x


def minimum_spanning_tree(X: np.ndarray) -> Any:
    """
    Returns the minimum spanning tree from a NumPy image array.

    Parameters
    ----------
    X: np.ndarray
        Numpy image array

    Returns
    -------
        Data representing the minimum spanning tree
    """
    # All features belong on second dimension
    X = X.reshape((X.shape[0], -1))
    # We add a small constant to the distance matrix to ensure scipy interprets
    # the input graph as fully-connected.
    dense_eudist = squareform(pdist(X)) + EPSILON
    eudist_csr = csr_matrix(dense_eudist)
    return mst(eudist_csr)


def get_classes_counts(labels: np.ndarray) -> Tuple[int, int]:
    """
    Returns the classes and counts of from an array of labels

    Parameters
    ----------
    label: np.ndarray
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
    A: np.ndarray,
    B: np.ndarray,
    k: int = 1,
    algorithm: Literal["auto", "ball_tree", "kd_tree"] = "auto",
) -> np.ndarray:
    """
    For each sample in A, compute the nearest neighbor in B

    Parameters
    ----------
    A, B : np.ndarray
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

    See Also
    --------
    :func:`sklearn.neighbors.NearestNeighbors`
    """

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm=algorithm).fit(B)
    nns = nbrs.kneighbors(A)[1]
    nns = nns[:, 1]

    return nns
