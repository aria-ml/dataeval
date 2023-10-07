"""
This module contains the implementation of the
FR Test Statistic based estimate for the Bayes Error Rate
"""
from typing import Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from daml._internal.metrics.base import Metric
from daml._internal.metrics.outputs import BEROutput


class MultiClassBER(Metric):
    """
    Implements the FR Test Statistic based estimator for the Bayes Error Rate

    Note
    ----
    `Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4) <https://arxiv.org/abs/1811.06419>`_
    """  # noqa F401

    def __init__(self) -> None:
        """Constructor method"""

        super().__init__()

    def _multiclass_ber(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Calculates the Bayes Error Rate estimate

        Parameters
        ----------
        X : np.ndarray
            (n_samples x n_features) array of covariates (or image embeddings)
        y : np.ndarray
            n_samples vector of class labels with M unique classes. 2 <= M <= 10

        Returns
        -------
        float
            Estimate of the Bayes Error Rate

        Raises
        ------
        ValueError
            If unique classes M < 2 or M > 10
        """
        classes, counts = np.unique(y, return_counts=True)
        M = len(classes)
        N = np.sum(counts)
        if M < 2:
            raise ValueError("Label vector contains less than 2 classes!")
        if M > 10:
            raise ValueError("Label vector contains more than 10 classes!")
        # All features belong on second dimension
        X = X.reshape((X.shape[0], -1))
        # We add a small constant to the distance matrix to ensure scipy interprets
        # the input graph as fully-connected.
        dense_eudist = squareform(pdist(X)) + 1e-4
        eudist_csr = csr_matrix(dense_eudist)
        tree = coo_matrix(minimum_spanning_tree(eudist_csr))
        deltas = np.sum([y[tree.row[i]] != y[tree.col[i]] for i in range(N - 1)]) / (
            2 * N
        )
        upper = 2 * deltas
        lower = ((M - 1) / (M)) * (1 - (1 - 2 * ((M) / (M - 1)) * deltas) ** 0.5)
        return upper, lower

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> BEROutput:
        """
        Returns the Bayes Error Rate estimate

        Parameters
        ----------
        X : np.ndarray
            (n_samples x n_features) array of covariates (or image embeddings)
        y : np.ndarray
            n_samples vector of class labels with M unique classes. 2 <= M <= 10

        Returns
        -------
        BEROutput
            Dataclass containing the estimate of the Bayes Error Rate
        """
        ber, ber_lower = self._multiclass_ber(X, y)
        return BEROutput(ber=ber, ber_lower=ber_lower)
