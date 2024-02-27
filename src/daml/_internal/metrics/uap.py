"""
This module contains the implementation of the
FR Test Statistic based estimate for the upperbound
average precision using empirical mean precision
"""

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score

from daml._internal.metrics.base import EvaluateMixin


class UAP(EvaluateMixin):
    """
    FR Test Statistic based estimate of the empirical mean precision

    Parameters
    ----------
    labels : np.ndarray
        A numpy array of n_samples of class labels with M unique classes.

    scores : np.ndarray
        A 2D array of class probabilities per image
    """

    def __init__(self, labels: np.ndarray, scores: np.ndarray) -> None:
        self.labels = labels
        self.scores = scores

    def evaluate(self) -> Dict[str, float]:
        """
        Returns
        -------
        Dict[str, float]
            uap : The empirical mean precision estimate

        Raises
        ------
        ValueError
            If unique classes M < 2
        """
        uap = float(
            average_precision_score(self.labels, self.scores, average="weighted")
        )
        return {"uap": uap}
