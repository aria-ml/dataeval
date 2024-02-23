"""
This module contains the implementation of the
FR Test Statistic based estimate and the
KNN based estimate for the Bayes Error Rate

Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4)
https://arxiv.org/abs/1811.06419
"""

from typing import Callable, Dict, Literal, Tuple

import numpy as np
from scipy.sparse import coo_matrix

from daml._internal.metrics.base import EvaluateMixin, MethodsMixin
from daml._internal.metrics.outputs import BEROutput

from .utils import compute_neighbors, get_classes_counts, minimum_spanning_tree


def _mst(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    M, N = get_classes_counts(y)

    tree = coo_matrix(minimum_spanning_tree(X))
    matches = np.sum([y[tree.row[i]] != y[tree.col[i]] for i in range(N - 1)])
    deltas = matches / (2 * N)
    upper = 2 * deltas
    lower = ((M - 1) / (M)) * (1 - (1 - 2 * ((M) / (M - 1)) * deltas) ** 0.5)
    return upper, lower


def _knn(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    M, N = get_classes_counts(y)

    # All features belong on second dimension
    X = X.reshape((X.shape[0], -1))
    nn_indices = compute_neighbors(X, X)
    deltas = float(np.count_nonzero(y[nn_indices] - y) / (2 * N))
    upper = 2 * deltas
    lower = ((M - 1) / (M)) * (1 - (1 - 2 * ((M) / (M - 1)) * deltas) ** 0.5)
    return upper, lower


class BER(
    EvaluateMixin, MethodsMixin[Callable[[np.ndarray, np.ndarray], Tuple[float, float]]]
):
    """
    An estimator for Multi-class Bayes Error Rate using FR or KNN test statistic basis

    Parameters
    ----------
    data : np.ndarray
        Array of images or image embeddings
    labels : np.ndarray
        Array of labels for each image or image embedding
    method : Literal["MST", "KNN"], default "MST"
        Method to use when estimating the Bayes error rate

    See Also
    --------
    `Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4) <https://arxiv.org/abs/1811.06419>`_

    """  # noqa F401

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        method: Literal["MST", "KNN"] = "MST",
    ) -> None:
        self.data = data
        self.labels = labels
        self.method = method

    @classmethod
    def _methods(
        cls,
    ) -> Dict[str, Callable[[np.ndarray, np.ndarray], Tuple[float, float]]]:
        return {"MST": _mst, "KNN": _knn}

    def evaluate(self) -> BEROutput:
        """
        Calculates the Bayes Error Rate estimate using the provided method

        Returns
        -------
        BEROutput
            The estimated upper and lower bounds of the Bayes Error Rate

        Raises
        ------
        ValueError
            If unique classes M < 2
        """
        upper, lower = self.method(self.data, self.labels)
        return BEROutput(ber=upper, ber_lower=lower)
