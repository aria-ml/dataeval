import math
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import pdist, squareform

from dataeval._internal.interop import to_numpy


class CoverageOutput(NamedTuple):
    """
    Attributes
    ----------
    indices : np.ndarray
        Array of uncovered indices
    radii : np.ndarray
        Array of critical value radii
    critical_value : float
        Radius for coverage
    """

    indices: NDArray[np.intp]
    radii: NDArray[np.float64]
    critical_value: float


def coverage(
    embeddings: ArrayLike,
    radius_type: Literal["adaptive", "naive"] = "adaptive",
    k: int = 20,
    percent: np.float64 = np.float64(0.01),
) -> CoverageOutput:
    """
    Class for evaluating coverage and identifying images/samples that are in undercovered regions.

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
    CoverageOutput
        Array of uncovered indices, critical value radii, and the radius for coverage

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
    >>> coverage(embeddings)
    CoverageOutput(indices=array([], dtype=int64), radii=array([0.59307666, 0.56956307, 0.56328616, 0.70660265, 0.57778087,
        0.53738624, 0.58968217, 1.27721334, 0.84378694, 0.67767021,
        0.69680335, 1.35532621, 0.59764166, 0.8691945 , 0.83627602,
        0.84187303, 0.62212358, 1.09039732, 0.67956797, 0.60134383,
        0.83713908, 0.91784263, 1.12901193, 0.73907618, 0.63943983,
        0.61188447, 0.47872713, 0.57207771, 0.92885883, 0.54750511,
        0.83015726, 1.20721778, 0.50421928, 0.98312246, 0.59764166,
        0.61009202, 0.73864073, 1.0381061 , 0.77598609, 0.72984036,
        0.67573006, 0.48056064, 1.00050879, 0.89532971, 0.58395529,
        0.95954793, 0.60134383, 1.10096454, 0.51955314, 0.73038702]), critical_value=0)

    Reference
    ---------
    This implementation is based on https://dl.acm.org/doi/abs/10.1145/3448016.3457315.
    [1] Seymour Sudman. 1976. Applied sampling. Academic Press New York (1976).
    """  # noqa: E501

    # Calculate distance matrix, look at the (k+1)th farthest neighbor for each image.
    embeddings = to_numpy(embeddings)
    n = len(embeddings)
    if n <= k:
        raise ValueError("Number of observations less than or equal to the specified number of neighbors.")
    mat = squareform(pdist(embeddings)).astype(np.float64)
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
    return CoverageOutput(pvals, crit, rho)
