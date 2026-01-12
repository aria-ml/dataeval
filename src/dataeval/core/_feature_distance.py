__all__ = []

import logging
from collections.abc import Sequence
from typing import NamedTuple, TypedDict, cast

import numpy as np
from scipy.stats import iqr, ks_2samp, wasserstein_distance

from dataeval.types import Array1D, Array2D
from dataeval.utils.arrays import as_numpy

_logger = logging.getLogger(__name__)


class KSType(NamedTuple):
    """Used to typehint scipy's internal hidden ks_2samp output"""

    statistic: float
    statistic_location: float
    pvalue: float


class FeatureDistanceResult(TypedDict):
    """
    Type definition for a single feature distance test result.

    Attributes
    ----------
    statistic : float
        The Kolmogorov-Smirnov test statistic
    location : float
        The normalized location where the KS statistic was achieved
    dist : float
        The Earth Mover's Distance (Wasserstein distance) between distributions
    p_value : float
        The p-value from the KS test
    """

    statistic: float
    location: float
    dist: float
    p_value: float


def _calculate_drift(x1: Array1D[float] | Array2D[float], x2: Array1D[float] | Array2D[float]) -> float:
    """Calculates the shift magnitude between x1 and x2 scaled by x1"""

    distance = wasserstein_distance(x1, x2)

    X = iqr(x1)

    # Preferred scaling of x1
    if X:
        return distance / X

    # Return if single-valued, else scale
    xmin, xmax = np.min(as_numpy(x1)), np.max(as_numpy(x1))
    return distance if xmin == xmax else distance / (xmax - xmin)


def feature_distance(
    continuous_data_1: Array1D[float] | Array2D[float],
    continuous_data_2: Array1D[float] | Array2D[float],
) -> Sequence[FeatureDistanceResult]:
    """
    Measures the feature-wise distance between two continuous distributions and computes a
    p-value to evaluate its significance.

    Uses the Earth Mover's Distance and the Kolmogorov-Smirnov two-sample test, featurewise.

    Parameters
    ----------
    continuous_data_1 : Array1D[float] | Array2D[float]
        Array of values to be used as reference. Can be a 1D or 2D list, or array-like object.
    continuous_data_2 : Array1D[float] | Array2D[float]
        Array of values to be compare with the reference. Can be a 1D or 2D list, or array-like object.

    Returns
    -------
    Sequence[FeatureDistanceResult]
        Sequence of mappings, one per feature, each with keys:

        - statistic: float - The Kolmogorov-Smirnov test statistic
        - location: float - The normalized location where the KS statistic was achieved
        - dist: float - The Earth Mover's Distance between distributions
        - p_value: float - The p-value from the KS test

    See Also
    --------
    Earth mover's distance

    Kolmogorov-Smirnov two-sample test
    """
    _logger.info("Starting feature_distance calculation")

    cont1 = np.atleast_2d(as_numpy(continuous_data_1, dtype=np.float64, required_ndim=(1, 2)))  # (S, F)
    cont2 = np.atleast_2d(as_numpy(continuous_data_2, dtype=np.float64, required_ndim=(1, 2)))  # (S, F)

    _logger.debug("Data 1 shape: %s, Data 2 shape: %s", cont1.shape, cont2.shape)

    if len(cont1.T) != len(cont2.T):
        raise ValueError(f"Data must have the same numbers of features. ({len(cont1.T)} != {len(cont2.T)})")

    N = len(cont1)
    M = len(cont2)

    # This is a simplified version of sqrt(N*M / N+M) < 4
    if (N - 16) * (M - 16) < 256:
        _logger.warning(
            f"Sample sizes of {N}, {M} will yield unreliable p-values from the KS test. "
            f"Recommended 32 samples per factor or at least 16 if one set has many more.",
        )

    # Set default for statistic, location, and magnitude to zero and pvalue to one
    results: list[FeatureDistanceResult] = []

    # Per factor
    for i in range(len(cont1.T)):
        fdata1 = cont1[:, i]  # (S, 1)
        fdata2 = cont2[:, i]  # (S, 1)

        # Min and max over both distributions
        xmin = min(min(fdata1), min(fdata2))
        xmax = max(max(fdata1), max(fdata2))

        # Default case
        if xmin == xmax:
            results.append({"statistic": 0.0, "location": 0.0, "dist": 0.0, "p_value": 1.0})
            continue

        ks_result = cast(KSType, ks_2samp(fdata1, fdata2, method="asymp"))

        # Normalized location
        loc = float((ks_result.statistic_location - xmin) / (xmax - xmin))

        drift = _calculate_drift(fdata1, fdata2)

        results.append({"statistic": ks_result.statistic, "location": loc, "dist": drift, "p_value": ks_result.pvalue})

    _logger.info("Feature distance calculation complete: analyzed %d features", len(results))

    return results
