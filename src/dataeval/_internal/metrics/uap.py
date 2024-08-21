"""
This module contains the implementation of the
FR Test Statistic based estimate for the upperbound
average precision using empirical mean precision
"""

from typing import Dict

from dataeval._internal.functional.uap import uap
from dataeval._internal.interop import ArrayLike, to_numpy
from dataeval._internal.metrics.base import EvaluateMixin


class UAP(EvaluateMixin):
    """
    FR Test Statistic based estimate of the empirical mean precision

    """

    def evaluate(self, labels: ArrayLike, scores: ArrayLike) -> Dict[str, float]:
        """
        Estimates the upperbound average precision

        Parameters
        ----------
        labels : ArrayLike
            A numpy array of n_samples of class labels with M unique classes.
        scores : ArrayLike
            A 2D array of class probabilities per image

        Returns
        -------
        Dict[str, float]
            uap : The empirical mean precision estimate

        Raises
        ------
        ValueError
            If unique classes M < 2
        """

        return {"uap": uap(to_numpy(labels), to_numpy(scores))}
