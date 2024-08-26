"""
This module contains the implementation of the
FR Test Statistic based estimate and the
KNN based estimate for the Bayes Error Rate

Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4)
https://arxiv.org/abs/1811.06419
"""

from typing import Callable, Dict, Literal, Tuple

from numpy.typing import ArrayLike, NDArray

from dataeval._internal.functional.ber import ber_knn, ber_mst
from dataeval._internal.interop import to_numpy
from dataeval._internal.metrics.base import EvaluateMixin, MethodsMixin

_METHODS = Literal["KNN", "MST"]
_FUNCTION = Callable[[NDArray, NDArray, int], Tuple[float, float]]


class BER(EvaluateMixin, MethodsMixin[_METHODS, _FUNCTION]):
    """
    An estimator for Multi-class Bayes Error Rate using FR or KNN test statistic basis

    Parameters
    ----------
    method : Literal["KNN", "MST"], default "KNN"
        Method to use when estimating the Bayes error rate
    k : int, default 1
        Number of nearest neighbors for KNN estimator -- ignored by MST estimator

    References
    ----------
    [1] `Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4) <https://arxiv.org/abs/1811.06419>`_

    Examples
    --------
    >>> import sklearn.datasets as dsets
    >>> from dataeval.metrics import BER

    >>> images, labels = dsets.make_blobs(n_samples=50, centers=2, n_features=2, random_state=0)

    >>> ber = BER()
    >>> ber.evaluate(images, labels)
    {'ber': 0.04, 'ber_lower': 0.020416847668728033}
    """

    def __init__(self, method: _METHODS = "KNN", k: int = 1) -> None:
        self.k: int = k
        self._set_method(method)

    @classmethod
    def _methods(cls) -> Dict[str, _FUNCTION]:
        return {"KNN": ber_knn, "MST": ber_mst}

    def evaluate(self, images: ArrayLike, labels: ArrayLike) -> Dict[str, float]:
        """
        Calculates the Bayes Error Rate estimate using the provided method

        Parameters
        ----------
        images : ArrayLike (N, ... )
            Array of images or image embeddings
        labels : ArrayLike (N, 1)
            Array of labels for each image or image embedding

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

        upper, lower = self._method(to_numpy(images), to_numpy(labels), self.k)  # type: ignore
        return {"ber": upper, "ber_lower": lower}
