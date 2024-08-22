from typing import Literal, Tuple

import numpy as np

from dataeval._internal.functional.coverage import coverage
from dataeval._internal.interop import ArrayLike, to_numpy
from dataeval._internal.metrics.base import EvaluateMixin


class Coverage(EvaluateMixin):
    """
    Class for evaluating coverage and identifying images/samples that are in undercovered regions.

    Parameters
    ----------
    radius_type : Literal["adaptive", "naive"], default "adaptive"
        The function used to determine radius.
    k: int, default 20
        Number of observations required in order to be covered.
        [1] suggests that a minimum of 20-50 samples is necessary.
    percent: np.float64, default np.float(0.01)
        Percent of observations to be considered uncovered. Only applies to adaptive radius.

    Reference
    ---------
    This implementation is based on https://dl.acm.org/doi/abs/10.1145/3448016.3457315.
    [1] Seymour Sudman. 1976. Applied sampling. Academic Press New York (1976).

    Examples
    --------
    Initialize the Coverage class:

    >>> cover = Coverage()

    Adjusting parameters:

    >>> cover = Coverage(k=5, percent=0.1)
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
        embeddings : ArrayLike, shape - (N, P)
            A dataset in an ArrayLike format.
            Function expects the data to have 2 dimensions, N number of observations in a P-dimesionial space.

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

        Example
        -------
        >>> cover.evaluate(embeddings)
        (array([31,  7, 22, 37, 11]), array([0.35938604, 0.26462789, 0.20319609, 0.34140912, 0.31069921,
               0.2308378 , 0.33300179, 0.69881025, 0.53587532, 0.35689803,
               0.39333634, 0.67497874, 0.21788128, 0.43510162, 0.38601861,
               0.34171868, 0.16941337, 0.66438044, 0.20319609, 0.19732733,
               0.48660288, 0.5135814 , 0.69352653, 0.26946943, 0.31120605,
               0.33067705, 0.30508271, 0.32802489, 0.51805702, 0.31120605,
               0.40843265, 0.74996768, 0.31069921, 0.52263763, 0.26654013,
               0.33113507, 0.40814838, 0.67723008, 0.48124375, 0.37243185,
               0.29760001, 0.30907904, 0.59023236, 0.57778087, 0.21839853,
               0.46067782, 0.31078966, 0.65199049, 0.26410603, 0.19542706]))
        """

        return coverage(to_numpy(embeddings), self.radius_type, self.k, self.percent)
