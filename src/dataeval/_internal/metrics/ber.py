"""
This module contains the implementation of the
FR Test Statistic based estimate and the
KNN based estimate for the Bayes Error Rate

Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4)
https://arxiv.org/abs/1811.06419
"""

from typing import Callable, Dict, Literal, Tuple

import numpy as np

from dataeval._internal.functional.ber import ber_knn, ber_mst
from dataeval._internal.metrics.base import EvaluateMixin, MethodsMixin

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

    def __init__(self, data: np.ndarray, labels: np.ndarray, method: _METHODS = "KNN", k: int = 1) -> None:
        self.data = data
        self.labels = labels
        self.k = k
        self._set_method(method)

    @classmethod
    def _methods(
        cls,
    ) -> Dict[str, _FUNCTION]:
        return {"KNN": ber_knn, "MST": ber_mst}

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

        upper, lower = self._method(np.asarray(self.data), np.asarray(self.labels), self.k)
        return {"ber": upper, "ber_lower": lower}
