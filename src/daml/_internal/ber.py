from abc import ABC
from typing import Any, Dict

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from daml._internal.MetricClasses import BER, Metrics


class MultiClassBER(BER, ABC):
    def __init__(self) -> None:
        super().__init__()

    def _multiclass_ber(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Implements the FR Test Statistic based estimator for the Bayes Error Rate
        derived here: https://arxiv.org/abs/1811.06419 (Th. 3 and Th. 4)
        :inputs:
        X - (n_samples x n_features) array of covariates (or image embeddings)
        y - n_samples vector of class labels with M unique classes. 2 <= M <= 10
        :return:
        Estimate of the Bayes Error Rate
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
        # Sparse matrix of pairwise distances between each feature vector
        Xdist = csr_matrix(np.triu(squareform(pdist(X)), 1))
        tree = coo_matrix(minimum_spanning_tree(Xdist, overwrite=True))
        deltas = np.sum([y[tree.row[i]] != y[tree.col[i]] for i in range(N - 1)])

        return deltas / N

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        TODO: Add Metric description for documentation.
        https://gitlab.jatic.net/jatic/aria/daml/-/issues/83
        """
        return {
            Metrics.BER: self._multiclass_ber(X, y),
        }
