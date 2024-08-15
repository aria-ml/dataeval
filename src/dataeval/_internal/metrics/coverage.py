import math
from typing import Literal, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform

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
        self.radius_type = radius_type
        self.k = k
        self.percent = percent

    def evaluate(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a one-way chi-squared test between observation frequencies and expected frequencies that
        tests the null hypothesis that the observed data has the expected frequencies.

        Parameters
        ----------
        embeddings : np.ndarray
            n x p array of image embeddings from the dataset.

        Returns
        -------
        np.ndarray
            Array of uncovered indices
        np.ndarray
            Array of critical value radii

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

        # Calculate distance matrix, look at the (k+1)th farthest neighbor for each image.
        n = len(embeddings)
        if n <= self.k:
            raise ValueError("Number of observations less than or equal to the specified number of neighbors.")
        mat = squareform(pdist(embeddings))
        sorted_dists = np.sort(mat, axis=1)
        crit = sorted_dists[:, self.k + 1]

        d = np.shape(embeddings)[1]
        if self.radius_type == "naive":
            self.rho = (1 / math.sqrt(math.pi)) * ((2 * self.k * math.gamma(d / 2 + 1)) / (n)) ** (1 / d)
            pvals = np.where(crit > self.rho)[0]
        elif self.radius_type == "adaptive":
            # Use data adaptive cutoff
            cutoff = int(n * self.percent)
            pvals = np.argsort(crit)[::-1][:cutoff]
        else:
            raise ValueError("Invalid radius type.")
        return pvals, crit
