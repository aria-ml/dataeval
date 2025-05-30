"""
This module contains the implementation of the
FR Test Statistic based estimate and the
KNN based estimate for the :term:`Bayes error rate<Bayes Error Rate (BER)>`

Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4)
https://arxiv.org/abs/1811.06419

"""

from __future__ import annotations

__all__ = []

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.stats import mode

from dataeval.config import EPSILON
from dataeval.outputs import BEROutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import Array
from dataeval.utils._array import as_numpy, ensure_embeddings
from dataeval.utils._method import get_method
from dataeval.utils._mst import compute_neighbors, minimum_spanning_tree


def ber_mst(images: NDArray[np.float64], labels: NDArray[np.int_], k: int = 1) -> tuple[float, float]:
    """Calculates the :term:`Bayes error rate<Bayes Error Rate (BER)>` using a minimum spanning tree.

    Parameters
    ----------
    images : NDArray, shape - (N, ... )
        n_samples containing n_features
    labels : NDArray, shape - (N, 1)
        Labels corresponding to each sample
    k : int
        Unused

    Returns
    -------
    Tuple[float, float]
        The upper and lower bounds of the bayes error rate
    """
    M, N = get_classes_counts(labels)

    tree = coo_matrix(minimum_spanning_tree(images))
    matches = np.sum([labels[tree.row[i]] != labels[tree.col[i]] for i in range(N - 1)])
    deltas = matches / (2 * N)
    upper = 2 * deltas
    lower = ((M - 1) / (M)) * (1 - max(1 - 2 * ((M) / (M - 1)) * deltas, 0) ** 0.5)
    return upper, lower


def ber_knn(images: NDArray[np.float64], labels: NDArray[np.int_], k: int) -> tuple[float, float]:
    """Calculates the :term:`Bayes error rate<Bayes Error Rate (BER)>` using K-nearest neighbors

    Parameters
    ----------
    images : NDArray, shape - (N, ... )
        n_samples containing n_features
    labels : NDArray, shape - (N, 1)
        Labels corresponding to each sample
    k : int
        The number of neighbors to find

    Returns
    -------
    Tuple[float, float]
        The upper and lower bounds of the bayes error rate
    """
    M, N = get_classes_counts(labels)
    nn_indices = compute_neighbors(images, images, k=k)
    nn_indices = np.expand_dims(nn_indices, axis=1) if nn_indices.ndim == 1 else nn_indices
    modal_class = mode(labels[nn_indices], axis=1, keepdims=True).mode.squeeze()
    upper = float(np.count_nonzero(modal_class - labels) / N)
    lower = knn_lowerbound(upper, M, k)
    return upper, lower


def knn_lowerbound(value: float, classes: int, k: int) -> float:
    """Several cases for computing the BER lower bound"""
    if value <= EPSILON:
        return 0.0

    if classes == 2 and k != 1:
        if k > 5:
            # Property 2 (Devroye, 1981) cited in Snoopy paper, not in snoopy repo
            alpha = 0.3399
            beta = 0.9749
            a_k = alpha * np.sqrt(k) / (k - 3.25) * (1 + beta / (np.sqrt(k - 3)))
            return value / (1 + a_k)
        if k > 2:
            return value / (1 + (1 / np.sqrt(k)))
        # k == 2:
        return value / 2

    return ((classes - 1) / classes) * (1 - np.sqrt(max(0, 1 - ((classes / (classes - 1)) * value))))


_BER_FN_MAP = {"KNN": ber_knn, "MST": ber_mst}


@set_metadata
def ber(embeddings: Array, labels: Array, k: int = 1, method: Literal["KNN", "MST"] = "KNN") -> BEROutput:
    """
    An estimator for Multi-class :term:`Bayes error rate<Bayes Error Rate (BER)>` \
    using FR or KNN test statistic basis.

    Parameters
    ----------
    embeddings : ArrayLike (N, ... )
        Array of image :term:`embeddings<Embeddings>`
    labels : ArrayLike (N, 1)
        Array of labels for each image
    k : int, default 1
        Number of nearest neighbors for KNN estimator -- ignored by MST estimator
    method : Literal["KNN", "MST"], default "KNN"
        Method to use when estimating the Bayes error rate

    Returns
    -------
    BEROutput
        The upper and lower bounds of the Bayes Error Rate

    References
    ----------
    [1] `Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4) <https://arxiv.org/abs/1811.06419>`_

    Examples
    --------
    >>> import sklearn.datasets as dsets
    >>> from dataeval.metrics.estimators import ber

    >>> images, labels = dsets.make_blobs(n_samples=50, centers=2, n_features=2, random_state=0)

    >>> ber(images, labels)
    BEROutput(ber=0.04, ber_lower=0.020416847668728033)
    """
    ber_fn = get_method(_BER_FN_MAP, method)
    X = ensure_embeddings(embeddings, dtype=np.float64)
    y = as_numpy(labels)
    upper, lower = ber_fn(X, y, k)
    return BEROutput(upper, lower)


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
    N = int(np.sum(counts))
    return M, N
