from typing import Any, Literal, NamedTuple, Tuple, Union

import numpy as np
from scipy.signal import convolve2d
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree as mst
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

EPSILON = 1e-5
EDGE_KERNEL = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.int8)
BIT_DEPTH = (1, 8, 12, 16, 32)


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
    nns = nns[:, 1:].squeeze()

    return nns


class BitDepth(NamedTuple):
    depth: int
    pmin: Union[float, int]
    pmax: Union[float, int]


def get_bitdepth(image: np.ndarray) -> BitDepth:
    """
    Approximates the bit depth of the image using the
    min and max pixel values.
    """
    pmin, pmax = np.min(image), np.max(image)
    if pmin < 0:
        return BitDepth(0, pmin, pmax)
    else:
        depth = ([x for x in BIT_DEPTH if 2**x > pmax] or [max(BIT_DEPTH)])[0]
        return BitDepth(depth, 0, 2**depth - 1)


def rescale(image: np.ndarray, depth: int = 1) -> np.ndarray:
    """
    Rescales the image using the bit depth provided.
    """
    bitdepth = get_bitdepth(image)
    if bitdepth.depth == depth:
        return image
    else:
        normalized = (image + bitdepth.pmin) / (bitdepth.pmax - bitdepth.pmin)
        return normalized * (2**depth - 1)


def normalize_image_shape(image: np.ndarray) -> np.ndarray:
    """
    Normalizes the image shape into (C,H,W).
    """
    ndim = image.ndim
    if ndim == 2:
        return np.expand_dims(image, axis=0)
    elif ndim == 3:
        return image
    elif ndim > 3:
        # Slice all but the last 3 dimensions
        return image[(0,) * (ndim - 3)]
    else:
        raise ValueError("Images must have 2 or more dimensions.")


def edge_filter(image: np.ndarray, offset: float = 0.5) -> np.ndarray:
    """
    Returns the image filtered using a 3x3 edge detection kernel:
    [[ -1, -1, -1 ],
     [ -1,  8, -1 ],
     [ -1, -1, -1 ]]
    """
    edges = convolve2d(image, EDGE_KERNEL, mode="same", boundary="symm") + offset
    np.clip(edges, 0, 255, edges)
    return edges
