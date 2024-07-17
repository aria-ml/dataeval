"""
This module contains the implementation of the
FR Test Statistic based estimate and the
KNN based estimate for the Bayes Error Rate

Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4)
https://arxiv.org/abs/1811.06419
"""

from typing import Callable, Dict, Literal, Tuple

import numpy as np
from maite.protocols import ArrayLike
from scipy.sparse import coo_matrix
from scipy.stats import mode

from daml._internal.metrics.base import EvaluateMixin, MethodsMixin

from .utils import compute_neighbors, get_classes_counts, minimum_spanning_tree


def _mst(X: np.ndarray, y: np.ndarray, _: int) -> Tuple[float, float]:
    M, N = get_classes_counts(y)

    tree = coo_matrix(minimum_spanning_tree(X))
    matches = np.sum([y[tree.row[i]] != y[tree.col[i]] for i in range(N - 1)])
    deltas = matches / (2 * N)
    upper = 2 * deltas
    lower = ((M - 1) / (M)) * (1 - max(1 - 2 * ((M) / (M - 1)) * deltas, 0) ** 0.5)
    return upper, lower


def _knn(X: np.ndarray, y: np.ndarray, k: int) -> Tuple[float, float]:
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
    "Several cases for computing the BER lower bound"
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


_METHODS = Literal["MST", "KNN"]
_FUNCTION = Callable[[np.ndarray, np.ndarray, int], Tuple[float, float]]


class BER(EvaluateMixin, MethodsMixin[_METHODS, _FUNCTION]):
    """
    An estimator for Multi-class Bayes Error Rate using FR or KNN test statistic basis

    Parameters
    ----------
    data : np.ndarray
        Array of images or image embeddings
    labels : np.ndarray
        Array of labels for each image or image embedding
    method : Literal["MST", "KNN"], default "KNN"
        Method to use when estimating the Bayes error rate
    k : int, default 1
        number of nearest neighbors for KNN estimator -- ignored by MST estimator


    See Also
    --------
    `Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4) <https://arxiv.org/abs/1811.06419>`_

    """

    def __init__(self, data: ArrayLike, labels: ArrayLike, method: _METHODS = "KNN", k: int = 1) -> None:
        self.data = data
        self.labels = labels
        self.k = k
        self._set_method(method)

    @classmethod
    def _methods(
        cls,
    ) -> Dict[str, _FUNCTION]:
        return {"MST": _mst, "KNN": _knn}

    def evaluate(self) -> Dict[str, float]:
        """
        Calculates the Bayes Error Rate estimate using the provided method

        Returns
        -------
        Dict[str, float]
            ber : float
                The estimated lower bounds of the Bayes Error Rate
            ber_lower : float
                The estimated upper bounds of the Bayes Error Rate

        Raises
        ------
        ValueError
            If unique classes M < 2
        """
        data = np.asarray(self.data)
        labels = np.asarray(self.labels)
        upper, lower = self._method(data, labels, self.k)
        return {"ber": upper, "ber_lower": lower}
