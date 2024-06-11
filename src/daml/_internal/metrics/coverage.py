import math
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform


class Coverage:
    """
     Class for evaluating coverage and identifying images/samples which are in undercovered regions.

     Idea behind this implementation comes from https://dl.acm.org/doi/abs/10.1145/3448016.3457315.

     Parameters
     ----------
     embeddings : np.ndarray
         n x p array of image embeddings from the dataset (reduced in dimension with an autoencoder or similar).
        note: Embeddings should be on the unit interval.
     radius_type : Optional[str]
         Either "naive" or "adaptive", denoting the type of radius to consider for coverage (default = adaptive).
    k: int
        Number of observations required in order to be covered. Default is 20 (based on the above paper and
          the central limit theorem).
    percent: np.float64
         Percent of observations to be considered uncovered. Only applies to adaptive radius (default is 0.01).

    Raises
    ------
    ValueError
         If radius type is not one of the two accepted values.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        radius_type: Optional[str] = "adaptive",
        k: int = 20,
        percent: np.float64 = np.float64(0.01),
    ):
        self.embeddings = embeddings
        self.radius_type = radius_type
        self.k = k
        self.percent = percent

    def evaluate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a one-way chi-squared test between observation frequencies and expected frequencies that
        tests the null hypothesis that the observed data has the expected frequencies.

        This function acts as an interface to the scipy.stats.chisquare method, which is documented at
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html

        Returns
        -------
        np.ndarray
            Array of uncovered indices
        np.ndarray
            Array of critical value radii
        """

        # Calculate distance matrix, look at the (k+1)st farthest neighbor for each image.
        n = len(self.embeddings)
        if n <= self.k:
            raise ValueError("Number of observations less than specified number of neighbors.")
        mat = squareform(pdist(self.embeddings))
        sorted_dists = np.sort(mat, axis=1)
        crit = sorted_dists[:, self.k + 1]

        d = np.shape(self.embeddings)[1]
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
