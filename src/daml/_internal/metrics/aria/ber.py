"""
This module contains the implementation of the
FR Test Statistic based estimate and the
FNN based estimate for the Bayes Error Rate
"""
from abc import abstractmethod
from typing import Tuple

import numpy as np
import torch
from scipy.sparse import coo_matrix

from daml._internal.metrics.aria.base import _BaseMetric
from daml._internal.metrics.outputs import BEROutput

from .utils import compute_neighbors, get_classes_counts, minimum_spanning_tree


class _MultiClassBer(_BaseMetric):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        encode: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Constructor method"""
        super().__init__(images, encode, device=device)
        self.images = images
        self.labels = labels

    @abstractmethod
    def _multiclass_ber(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, float]:
        """Abstract method for the implementation of multiclass BER calculation"""

    def _evaluate(self) -> BEROutput:
        """
        Return the Bayes Error Rate estimate

        Returns
        -------
        BEROutput
            The estimated upper and lower bounds of the Bayes Error Rate

        Raises
        ------
        ValueError
            If unique classes M < 2
        """
        # Pass X through an autoencoder before evaluating BER
        embeddings = self._encode(self.images)
        ber, ber_lower = self._multiclass_ber(embeddings, self.labels)
        return BEROutput(ber=ber, ber_lower=ber_lower)


class MultiClassBerMST(_MultiClassBer):
    """
    Implements the FR Test Statistic based estimator for the Bayes Error Rate

    Note
    ----
    `Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4) <https://arxiv.org/abs/1811.06419>`_
    """  # noqa F401

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
        M, N = get_classes_counts(y)

        tree = coo_matrix(minimum_spanning_tree(X))
        matches = np.sum([y[tree.row[i]] != y[tree.col[i]] for i in range(N - 1)])
        deltas = matches / (2 * N)
        upper = 2 * deltas
        lower = ((M - 1) / (M)) * (1 - (1 - 2 * ((M) / (M - 1)) * deltas) ** 0.5)
        return upper, lower


class MultiClassBerFNN(_MultiClassBer):
    """
    Implements the KNN Test Statistic based estimator for the Bayes Error Rate

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
    `Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4) <https://arxiv.org/abs/1811.06419>`_
    """  # noqa F401

    def _multiclass_ber(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, float]:
        M, N = get_classes_counts(y)

        # All features belong on second dimension
        X = X.reshape((X.shape[0], -1))
        nn_indices = compute_neighbors(X, X)
        deltas = float(np.count_nonzero(y[nn_indices] - y) / (2 * N))
        upper = 2 * deltas
        lower = ((M - 1) / (M)) * (1 - (1 - 2 * ((M) / (M - 1)) * deltas) ** 0.5)
        return upper, lower
