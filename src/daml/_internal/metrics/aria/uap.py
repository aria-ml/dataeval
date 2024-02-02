"""
This module contains the implementation of the
FR Test Statistic based estimate for the upperbound
average precision (UAP_MST) and empirical mean
precision (UAP_EMP)
"""

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from daml._internal.datasets.datasets import DamlDataset
from daml._internal.metrics.aria.base import _BaseMetric
from daml._internal.metrics.outputs import UAPOutput

from .utils import get_classes_counts, minimum_spanning_tree


class UAP_MST(_BaseMetric):
    def __init__(self, dataset: DamlDataset) -> None:
        """
        Parameters
        ----------
        dataset : DamlDataset
            Dataset containing (n_samples x n_features) array of (padded) instance
            embeddings and n_samples vector of class labels with M unique classes.
        """
        super().__init__(dataset, encode=False, device=torch.device("cpu"))

    def _evaluate(self) -> UAPOutput:
        """
        Upperbound Average Precision estimate

        Returns
        -------
        UAPOutput
            The estimated UAP
        """
        uap = self._uap(self.dataset.images, self.dataset.labels)
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
        M, N = get_classes_counts(y)

        tree = minimum_spanning_tree(X).todense()
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


class UAP_EMP(_BaseMetric):
    def __init__(self, dataset: DamlDataset, scores: np.ndarray) -> None:
        """
        Parameters
        ----------
        dataset : DamlDataset
            Dataset containing (n_samples x n_features) array of (padded) instance
            embeddings and n_samples vector of class labels with M unique classes.

        scores : np.ndarray
            A 2D array of class probabilities per image
        """
        self.scores = scores
        super().__init__(dataset, encode=False, device=torch.device("cpu"))

    def _evaluate(self) -> UAPOutput:
        """
        Returns
        -------
        UAPOutput
            The estimated UAP
        """
        y: np.ndarray = self.dataset.labels

        uap = float(average_precision_score(y, self.scores, average="weighted"))
        return UAPOutput(uap=uap)
