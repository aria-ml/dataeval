from __future__ import annotations

__all__ = []


import numpy as np
from numpy.typing import NDArray
from scipy.stats import mode

from dataeval.config import EPSILON
from dataeval.protocols import _1DArray, _2DArray
from dataeval.utils._array import as_numpy


def ber_mst(embeddings: _2DArray[float], class_labels: _1DArray[int]) -> tuple[float, float]:
    """
    An estimator for Multi-class :term:`Bayes error rate<Bayes Error Rate (BER)>` \
    using FR with a minimum spanning tree (MST) test statistic basis.

    Parameters
    ----------
    embeddings : _2DArray[float]
        Array of image :term:`embeddings<Embeddings>`. Can be a 2D list, or array-like object.
    class_labels : _1DArray[int]
        Array of class labels for each image. Can be a 1D list, or array-like object.

    Returns
    -------
    tuple[float, float]
        The upper and lower bounds, respectively, of the Bayes Error Rate

    References
    ----------
    [1] `Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4) <https://arxiv.org/abs/1811.06419>`_

    Examples
    --------
    >>> import sklearn.datasets as dsets
    >>> from dataeval.core._ber import ber_mst

    >>> images, class_labels = dsets.make_blobs(n_samples=50, centers=2, n_features=2, random_state=0)
    >>> ber_mst(images, class_labels)
    (0.04, 0.020416847668728033)
    """
    from dataeval.core._mst import minimum_spanning_tree

    data_np = as_numpy(embeddings, dtype=np.float32)
    labels_np = as_numpy(class_labels, dtype=np.intp)

    M, N = _get_classes_counts(labels_np)

    rows, cols = minimum_spanning_tree(data_np)  # get rows and cols directly
    mismatches = np.sum(class_labels[rows] != class_labels[cols])
    deltas = mismatches / (2 * N)
    upper = float(2 * deltas)
    lower = float(((M - 1) / (M)) * (1 - max(1 - 2 * ((M) / (M - 1)) * deltas, 0) ** 0.5))
    return upper, lower


def ber_knn(embeddings: _2DArray[float], class_labels: _1DArray[int], k: int) -> tuple[float, float]:
    """
    An estimator for Multi-class :term:`Bayes error rate<Bayes Error Rate (BER)>` \
    using KNN test statistic basis.

    Parameters
    ----------
    embeddings : _2DArray[float]
        Array of image :term:`embeddings<Embeddings>`. Can be a 2D list, or array-like object.
    class_labels : _1DArray[int]
        Array of class labels for each image. Can be a 1D list, or array-like object.
    k : int
        Number of nearest neighbors for KNN estimator

    Returns
    -------
    tuple[float, float]
        The upper and lower bounds, respectively, of the Bayes Error Rate

    References
    ----------
    [1] `Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4) <https://arxiv.org/abs/1811.06419>`_

    Examples
    --------
    >>> import sklearn.datasets as dsets
    >>> from dataeval.core._ber import ber_knn

    >>> images, class_labels = dsets.make_blobs(n_samples=50, centers=2, n_features=2, random_state=0)
    >>> ber_knn(images, class_labels, 1)
    (0.04, 0.020416847668728033)
    """
    from dataeval.core._mst import compute_neighbors

    data_np = as_numpy(embeddings, dtype=np.float32)
    labels_np = as_numpy(class_labels, dtype=np.intp)

    M, N = _get_classes_counts(labels_np)
    nn_indices = compute_neighbors(data_np, data_np, k=k)
    nn_indices = np.expand_dims(nn_indices, axis=1) if nn_indices.ndim == 1 else nn_indices
    modal_class = mode(class_labels[nn_indices], axis=1, keepdims=True).mode.squeeze()
    upper = float(np.count_nonzero(modal_class - class_labels) / N)
    lower = _knn_lowerbound(upper, M, k)
    return upper, lower


def _knn_lowerbound(value: float, classes: int, k: int) -> float:
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

    return float(((classes - 1) / classes) * (1 - np.sqrt(max(0, 1 - ((classes / (classes - 1)) * value)))))


def _get_classes_counts(labels: NDArray[np.intp]) -> tuple[int, int]:
    classes, counts = np.unique(labels, return_counts=True)
    M = len(classes)
    if M < 2:
        raise ValueError("Label vector contains less than 2 classes!")
    N = int(np.sum(counts))
    return M, N
