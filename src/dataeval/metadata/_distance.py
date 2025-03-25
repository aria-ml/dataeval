from __future__ import annotations

__all__ = []

import warnings
from typing import NamedTuple, cast

import numpy as np
from scipy.stats import iqr, ks_2samp
from scipy.stats import wasserstein_distance as emd

from dataeval.metadata._utils import _compare_keys, _validate_factors_and_data
from dataeval.outputs._base import MappingOutput
from dataeval.typing import ArrayLike
from dataeval.utils.data import Metadata


class KSType(NamedTuple):
    """Used to typehint scipy's internal hidden ks_2samp output"""

    statistic: float
    statistic_location: float
    pvalue: float


class MetadataKSResult(NamedTuple):
    """
    Attributes
    ----------
    statistic : float
        the KS statistic
    location : float
        The value at which the KS statistic has its maximum, measured in IQR-normalized units relative
        to the median of the reference distribution.
    dist : float
        The Earth Mover's Distance normalized by the interquartile range (IQR) of the reference
    pvalue : float
        The p-value from the KS two-sample test
    """

    statistic: float
    location: float
    dist: float
    pvalue: float


class KSOutput(MappingOutput[str, MetadataKSResult]):
    """
    Output class for results of ks_2samp featurewise comparisons of new metadata to reference metadata.


    Attributes
    ----------
    key: str
        Metadata feature names
    value: :class:`MetadataKSResult`
        Output per feature name containing the statistic, statistic location, distance, and pvalue.
    """


def _calculate_drift(x1: ArrayLike, x2: ArrayLike) -> float:
    """Calculates the shift magnitude between x1 and x2 scaled by x1"""

    distance = emd(x1, x2)

    X = iqr(x1)

    # Preferred scaling of x1
    if X:
        return distance / X

    # Return if single-valued, else scale
    xmin, xmax = np.min(x1), np.max(x1)
    return distance if xmin == xmax else distance / (xmax - xmin)


def metadata_distance(metadata1: Metadata, metadata2: Metadata) -> KSOutput:
    """
    Measures the feature-wise distance between two continuous metadata distributions and
    computes a p-value to evaluate its significance.

    Uses the Earth Mover's Distance and the Kolmogorov-Smirnov two-sample test, featurewise.

    Parameters
    ----------
    metadata1 : Metadata
        Class containing continuous factor names and values to be used as reference
    metadata2 : Metadata
        Class containing continuous factor names and values to be compare with the reference

    Returns
    -------
    dict[str, KstestResult]
        A dictionary with keys corresponding to metadata feature names, and values that are KstestResult objects, as
        defined by scipy.stats.ks_2samp.

    See Also
    --------
    Earth mover's distance

    Kolmogorov-Smirnov two-sample test

    Note
    ----
    This function only applies to the continuous data

    Examples
    --------
    >>> output = metadata_distance(metadata1, metadata2)
    >>> list(output)
    ['time', 'altitude']
    >>> output["time"]
    MetadataKSResult(statistic=1.0, location=0.44354838709677413, dist=2.7, pvalue=0.0)
    """

    _compare_keys(metadata1.continuous_factor_names, metadata2.continuous_factor_names)
    fnames = metadata1.continuous_factor_names

    cont1 = np.atleast_2d(metadata1.continuous_data)  # (S, F)
    cont2 = np.atleast_2d(metadata2.continuous_data)  # (S, F)

    _validate_factors_and_data(fnames, cont1)
    _validate_factors_and_data(fnames, cont2)

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
    results: dict[str, MetadataKSResult] = {}

    # Per factor
    for i, fname in enumerate(fnames):
        fdata1 = cont1[:, i]  # (S, 1)
        fdata2 = cont2[:, i]  # (S, 1)

        # Min and max over both distributions
        xmin = min(np.min(fdata1), np.min(fdata2))
        xmax = max(np.max(fdata1), np.max(fdata2))

        # Default case
        if xmin == xmax:
            results[fname] = MetadataKSResult(statistic=0.0, location=0.0, dist=0.0, pvalue=1.0)
            continue

        ks_result = cast(KSType, ks_2samp(fdata1, fdata2, method="asymp"))

        # Normalized location
        loc = float((ks_result.statistic_location - xmin) / (xmax - xmin))

        drift = _calculate_drift(fdata1, fdata2)

        results[fname] = MetadataKSResult(
            statistic=ks_result.statistic,
            location=loc,
            dist=drift,
            pvalue=ks_result.pvalue,
        )

    return KSOutput(results)
