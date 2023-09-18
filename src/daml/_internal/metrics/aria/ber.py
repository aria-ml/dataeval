"""
This module contains the implementation of the
FR Test Statistic based estimate for the Bayes Error Rate
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from daml._internal.metrics.base import Metric
from daml.metrics.outputs import BEROutput


class MultiClassBER(Metric):
    def __init__(self) -> None:
        """Constructor method"""

        super().__init__()

    def _multiclass_ber(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Implements the FR Test Statistic based estimator for the Bayes Error Rate

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

        See Also
        --------
        https://arxiv.org/abs/1811.06419 (Th. 3 and Th. 4)
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
    ) -> BEROutput:
        """
        Return the Bayes Error Rate estimate

        Parameters
        ----------
        X : np.ndarray
            (n_samples x n_features) array of covariates (or image embeddings)
        y : np.ndarray
            n_samples vector of class labels with M unique classes. 2 <= M <= 10

        Returns
        -------
        Dict[str, float]
            "ber": Estimate of the Bayes Error Rate

        .. todo:: Add Metric description for documentation.

        See Also
        --------
        https://gitlab.jatic.net/jatic/aria/daml/-/issues/83
        """

        return BEROutput(ber=self._multiclass_ber(X, y))
