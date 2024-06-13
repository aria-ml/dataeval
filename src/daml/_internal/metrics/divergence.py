"""
This module contains the implementation of HP Divergence
using the Fast Nearest Neighbor and Minimum Spanning Tree algorithms
"""

from typing import Any, Callable, Dict, Literal

import numpy as np

from daml._internal.metrics.base import EvaluateMixin, MethodsMixin

from .utils import compute_neighbors, minimum_spanning_tree


def _mst(data: np.ndarray, labels: np.ndarray) -> int:
    mst = minimum_spanning_tree(data).toarray()
    edgelist = np.transpose(np.nonzero(mst))
    errors = np.sum(labels[edgelist[:, 0]] != labels[edgelist[:, 1]])
    return errors


def _fnn(data: np.ndarray, labels: np.ndarray) -> int:
    nn_indices = compute_neighbors(data, data)
    errors = np.sum(np.abs(labels[nn_indices] - labels))
    return errors


_METHODS = Literal["MST", "FNN"]
_FUNCTION = Callable[[np.ndarray, np.ndarray], int]


class Divergence(EvaluateMixin, MethodsMixin[_METHODS, _FUNCTION]):
    """
    Calculates the estimated divergence between two datasets

    Parameters
    ----------
    data_a : np.ndarray
        Array of images or image embeddings to compare
    data_b : np.ndarray
        Array of images or image embeddings to compare
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

    def __init__(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        method: _METHODS = "MST",
    ) -> None:
        self.data_a = data_a
        self.data_b = data_b
        self._set_method(method)

    @classmethod
    def _methods(cls) -> Dict[str, _FUNCTION]:
        return {"FNN": _fnn, "MST": _mst}

    def evaluate(self) -> Dict[str, Any]:
        """
        Calculates the divergence and any errors between the datasets

        Returns
        -------
        Dict[str, Any]
            dp : float
                divergence value between 0.0 and 1.0
            errors : int
                the number of differing edges
        """
        N = self.data_a.shape[0]
        M = self.data_b.shape[0]

        stacked_data = np.vstack((self.data_a, self.data_b))
        labels = np.vstack([np.zeros([N, 1]), np.ones([M, 1])])

        errors = self._method(stacked_data, labels)
        dp = max(0.0, 1 - ((M + N) / (2 * M * N)) * errors)
        return {"divergence": dp, "error": errors}
