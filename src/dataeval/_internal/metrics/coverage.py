from typing import Literal, Tuple

import numpy as np

from dataeval._internal.functional.coverage import coverage
from dataeval._internal.interop import ArrayLike, to_numpy
from dataeval._internal.metrics.base import EvaluateMixin


class Coverage(EvaluateMixin):
    """
    Class for evaluating coverage and identifying images/samples that are in undercovered regions.

    This implementation is based on https://dl.acm.org/doi/abs/10.1145/3448016.3457315.

    Parameters
    ----------
    radius_type : Literal["adaptive", "naive"], default "adaptive"
        The function used to determine radius.
    k: int, default 20
        Number of observations required in order to be covered.
    percent: np.float64, default np.float(0.01)
        Percent of observations to be considered uncovered. Only applies to adaptive radius.
    """

    def __init__(
        self,
        radius_type: Literal["adaptive", "naive"] = "adaptive",
        k: int = 20,
        percent: np.float64 = np.float64(0.01),
    ):
        self.radius_type: Literal["adaptive", "naive"] = radius_type
        self.k: int = k
        self.percent: np.float64 = percent

    def evaluate(self, embeddings: ArrayLike) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform a one-way chi-squared test between observation frequencies and expected frequencies that
        tests the null hypothesis that the observed data has the expected frequencies.

        Parameters
        ----------
        embeddings : ArrayLike
            n x p array of image embeddings from the dataset.

        Returns
        -------
        np.ndarray
            Array of uncovered indices
        np.ndarray
            Array of critical value radii
        float
            Radius for coverage

        Raises
        ------
        ValueError
            If length of embeddings is less than or equal to k
        ValueError
            If radius_type is unknown

        Note
        ----
        Embeddings should be on the unit interval.
        """

        return coverage(to_numpy(embeddings), self.radius_type, self.k, self.percent)
