from __future__ import annotations

__all__ = []

import warnings
from collections.abc import Sequence
from typing import NamedTuple, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import iqr, ks_2samp, wasserstein_distance

from dataeval.typing import ArrayLike


class KSType(NamedTuple):
    """Used to typehint scipy's internal hidden ks_2samp output"""

    statistic: float
    statistic_location: float
    pvalue: float


def _calculate_drift(x1: ArrayLike, x2: ArrayLike) -> float:
    """Calculates the shift magnitude between x1 and x2 scaled by x1"""

    distance = wasserstein_distance(x1, x2)

    X = iqr(x1)

    # Preferred scaling of x1
    if X:
        return distance / X

    # Return if single-valued, else scale
    xmin, xmax = np.min(x1), np.max(x1)
    return distance if xmin == xmax else distance / (xmax - xmin)


def feature_distance(
    continuous_data_1: NDArray[np.float64],
    continuous_data_2: NDArray[np.float64],
) -> Sequence[tuple[float, float, float, float]]:
    """
    Measures the feature-wise distance between two continuous distributions and computes a
    p-value to evaluate its significance.

    Uses the Earth Mover's Distance and the Kolmogorov-Smirnov two-sample test, featurewise.

    Parameters
    ----------
    continuous_data_1 : NDArray[np.float64]
        Array of values to be used as reference.
    continuous_data_2 : NDArray[np.float64]
        Array of values to be compare with the reference.

    Returns
    -------
    Sequence[tuple[float, float, float, float]]
        A sequence of KSTestResult tuples as defined by scipy.stats.ks_2samp.

    See Also
    --------
    Earth mover's distance

    Kolmogorov-Smirnov two-sample test
    """

    cont1 = np.atleast_2d(continuous_data_1)  # (S, F)
    cont2 = np.atleast_2d(continuous_data_2)  # (S, F)

    if len(cont1.T) != len(cont2.T):
        raise ValueError(f"Data must have the same numbers of features. ({len(cont1.T)} != {len(cont2.T)})")

    N = len(cont1)
    M = len(cont2)

    # This is a simplified version of sqrt(N*M / N+M) < 4
    if (N - 16) * (M - 16) < 256:
        warnings.warn(
            f"Sample sizes of {N}, {M} will yield unreliable p-values from the KS test. "
            f"Recommended 32 samples per factor or at least 16 if one set has many more.",
            UserWarning,
        )

    # Set default for statistic, location, and magnitude to zero and pvalue to one
    results: list[tuple[float, float, float, float]] = []

    # Per factor
    for i in range(len(cont1.T)):
        fdata1 = cont1[:, i]  # (S, 1)
        fdata2 = cont2[:, i]  # (S, 1)

        # Min and max over both distributions
        xmin = min(min(fdata1), min(fdata2))
        xmax = max(max(fdata1), max(fdata2))

        # Default case
        if xmin == xmax:
            results.append((0.0, 0.0, 0.0, 1.0))
            continue

        ks_result = cast(KSType, ks_2samp(fdata1, fdata2, method="asymp"))

        # Normalized location
        loc = float((ks_result.statistic_location - xmin) / (xmax - xmin))

        drift = _calculate_drift(fdata1, fdata2)

        results.append((ks_result.statistic, loc, drift, ks_result.pvalue))

    return results
