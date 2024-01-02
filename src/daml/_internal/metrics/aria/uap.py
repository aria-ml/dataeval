"""
This module contains the implementation of the
FR Test Statistic based estimate for the upperbound
average precision (UAP)
"""
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import average_precision_score

from daml._internal.datasets.datasets import DamlDataset
from daml._internal.metrics.aria.base import _AriaMetric
from daml._internal.metrics.outputs import UAPOutput


class UAP(_AriaMetric):
    def __init__(self, encode: bool = False) -> None:
        """Constructor method"""

        super().__init__(encode)

    def _get_classes_counts(self, labels: np.ndarray) -> Tuple[int, np.intp]:
        classes, counts = np.unique(labels, return_counts=True)
        M = len(classes)
        N = np.sum(counts)
        return M, N

    def evaluate(self, dataset: DamlDataset) -> UAPOutput:
        """
        Return the Upperbound Average Precision estimate

        Parameters
        ----------
        dataset : DamlDataset
            Dataset containing (n_samples x n_features) array of (padded) instance
            embeddings and n_samples vector of class labels with M unique classes.

        Returns
        -------
        UAPOutput
            The estimated UAP
        """
        X: np.ndarray = dataset.images
        y: np.ndarray = dataset.labels

        # If self.encode == True, pass X through an autoencoder before evaluating BER
        if self.encode:
            if not self._is_trained or self.autoencoder is None:
                raise TypeError(
                    "Tried to encode data without fitting a model.\
                    Try calling Metric.fit_dataset(dataset) first."
                )
            else:
                X = self.autoencoder.encoder.predict(X)

        uap = self._uap(X, y)
        return UAPOutput(uap=uap)

    def _uap(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """
        Calculates the UAP estimate

        Parameters
        ----------
        X : np.ndarray
            (n_samples x n_features) array of (padded) instance embeddings
        y : np.ndarray
            n_samples vector of class labels with M unique classes. 2 <= M <= 10

        Returns
        -------
        float
            Estimate of the UAP

        Raises
        ------
        ValueError
            If unique classes M < 2 or M > 10
        """
        M, N = self._get_classes_counts(y)

        # All features belong on second dimension
        X = X.reshape((X.shape[0], -1))
        # We add a small constant to the distance matrix to ensure scipy interprets
        # the input graph as fully-connected.
        dense_eudist = squareform(pdist(X)) + 1e-4
        eudist_csr = csr_matrix(dense_eudist)
        tree = minimum_spanning_tree(eudist_csr).todense()
        conf_mat = np.zeros((M, M))
        for i in range(len(tree)):
            edges = np.where(tree[i, :] != 0)[1]
            for j in range(len(edges)):
                conf_mat[int(y[i]), int(y[edges[j]])] = (
                    conf_mat[int(y[i]), int(y[edges[j]])] + 1
                )
        probs = np.zeros((M, M))
        for i in range(M):
            probs[i, :] = conf_mat[i, :] / np.sum(conf_mat[i, :])
        lab_inds = np.zeros((N, M))
        prob_inds = np.zeros((N, M))
        for i in range(N):
            lab_inds[i, int(y[i])] = 1
            prob_inds[i, :] = probs[int(y[i]), :]
        uap = average_precision_score(lab_inds, prob_inds)
        return float(uap)
