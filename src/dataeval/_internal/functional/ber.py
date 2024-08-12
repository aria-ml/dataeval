from typing import Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.stats import mode

from dataeval._internal.functional.utils import compute_neighbors, get_classes_counts, minimum_spanning_tree


def ber_mst(X: np.ndarray, y: np.ndarray, _: int) -> Tuple[float, float]:
    """Calculates the Bayes Error Rate using a minimum spanning tree

    Parameters
    ----------
    X : np.ndarray (N, :)
        Data points with arbitrary dimensionality
    y : np.ndarray (N, 1)
        Labels for each data point
    """

    M, N = get_classes_counts(y)

    tree = coo_matrix(minimum_spanning_tree(X))
    matches = np.sum([y[tree.row[i]] != y[tree.col[i]] for i in range(N - 1)])
    deltas = matches / (2 * N)
    upper = 2 * deltas
    lower = ((M - 1) / (M)) * (1 - max(1 - 2 * ((M) / (M - 1)) * deltas, 0) ** 0.5)
    return upper, lower


def ber_knn(X: np.ndarray, y: np.ndarray, k: int) -> Tuple[float, float]:
    """Calculates the Bayes Error Rate using K-nearest neighbors"""

    M, N = get_classes_counts(y)

    # All features belong on second dimension
    X = X.reshape((X.shape[0], -1))
    nn_indices = compute_neighbors(X, X, k=k)
    nn_indices = np.expand_dims(nn_indices, axis=1) if nn_indices.ndim == 1 else nn_indices
    modal_class = mode(y[nn_indices], axis=1, keepdims=True).mode.squeeze()
    upper = float(np.count_nonzero(modal_class - y) / N)
    lower = _knn_lowerbound(upper, M, k)
    return upper, lower


def _knn_lowerbound(value: float, classes: int, k: int) -> float:
    """Several cases for computing the BER lower bound"""
    if value <= 1e-10:
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
