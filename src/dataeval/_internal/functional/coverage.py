import math
from typing import Literal, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform


def coverage(
    embeddings: np.ndarray,
    radius_type: Literal["adaptive", "naive"] = "adaptive",
    k: int = 20,
    percent: np.float64 = np.float64(0.01),
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Perform a one-way chi-squared test between observation frequencies and expected frequencies that
    tests the null hypothesis that the observed data has the expected frequencies.

    Parameters
    ----------
    embeddings : ArrayLike, shape - (N, P)
        A dataset in an ArrayLike format.
        Function expects the data to have 2 dimensions, N number of observations in a P-dimesionial space.
    radius_type : Literal["adaptive", "naive"], default "adaptive"
        The function used to determine radius.
    k: int, default 20
        Number of observations required in order to be covered.
        [1] suggests that a minimum of 20-50 samples is necessary.
    percent: np.float64, default np.float(0.01)
        Percent of observations to be considered uncovered. Only applies to adaptive radius.

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

    Reference
    ---------
    This implementation is based on https://dl.acm.org/doi/abs/10.1145/3448016.3457315.
    [1] Seymour Sudman. 1976. Applied sampling. Academic Press New York (1976).
    """

    # Calculate distance matrix, look at the (k+1)th farthest neighbor for each image.
    n = len(embeddings)
    if n <= k:
        raise ValueError("Number of observations less than or equal to the specified number of neighbors.")
    mat = squareform(pdist(embeddings))
    sorted_dists = np.sort(mat, axis=1)
    crit = sorted_dists[:, k + 1]

    d = np.shape(embeddings)[1]
    if radius_type == "naive":
        rho = (1 / math.sqrt(math.pi)) * ((2 * k * math.gamma(d / 2 + 1)) / (n)) ** (1 / d)
        pvals = np.where(crit > rho)[0]
    elif radius_type == "adaptive":
        # Use data adaptive cutoff as rho
        rho = int(n * percent)
        pvals = np.argsort(crit)[::-1][:rho]
    else:
        raise ValueError("Invalid radius type.")
    return pvals, crit, rho
