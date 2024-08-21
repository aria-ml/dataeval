"""
This module contains the implementation of HP Divergence
using the Fast Nearest Neighbor and Minimum Spanning Tree algorithms
"""

from typing import Any, Callable, Dict, Literal

import numpy as np

from dataeval._internal.functional.divergence import divergence_fnn, divergence_mst
from dataeval._internal.interop import ArrayLike, to_numpy
from dataeval._internal.metrics.base import EvaluateMixin, MethodsMixin

_METHODS = Literal["MST", "FNN"]
_FUNCTION = Callable[[np.ndarray, np.ndarray], int]


class Divergence(EvaluateMixin, MethodsMixin[_METHODS, _FUNCTION]):
    """
    Calculates the estimated divergence between two datasets

    Parameters
    ----------
    method : Literal["MST, "FNN"], default "MST"
        Method used to estimate dataset divergence

    See Also
    --------
        For more information about this divergence, its formal definition,
        and its associated estimators see https://arxiv.org/abs/1412.6534.

    Warning
    -------
        MST is very slow in this implementation, this is unlike matlab where
        they have comparable speeds
        Overall, MST takes ~25x LONGER!!
        Source of slowdown:
        conversion to and from CSR format adds ~10% of the time diff between
        1nn and scipy mst function the remaining 90%
    """

    def __init__(self, method: _METHODS = "MST") -> None:
        self._set_method(method)

    @classmethod
    def _methods(cls) -> Dict[str, _FUNCTION]:
        return {"FNN": divergence_fnn, "MST": divergence_mst}

    def evaluate(self, data_a: ArrayLike, data_b: ArrayLike) -> Dict[str, Any]:
        """
        Calculates the divergence and any errors between the datasets

        Parameters
        ----------
        data_a : ArrayLike
            Array of images or image embeddings to compare
        data_b : ArrayLike
            Array of images or image embeddings to compare

        Returns
        -------
        Dict[str, Any]
            dp : float
                divergence value between 0.0 and 1.0
            errors : int
                the number of differing edges
        """
        a = to_numpy(data_a)
        b = to_numpy(data_b)
        N = a.shape[0]
        M = b.shape[0]

        stacked_data = np.vstack((a, b))
        labels = np.vstack([np.zeros([N, 1]), np.ones([M, 1])])

        errors = self._method(stacked_data, labels)
        dp = max(0.0, 1 - ((M + N) / (2 * M * N)) * errors)
        return {"divergence": dp, "error": errors}
