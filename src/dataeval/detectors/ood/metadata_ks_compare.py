from __future__ import annotations

import numbers
import warnings
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.stats import iqr, ks_2samp
from scipy.stats import wasserstein_distance as emd


def meta_distribution_compare(
    md0: Mapping[str, list[Any] | NDArray[Any]], md1: Mapping[str, list[Any] | NDArray[Any]]
) -> dict[str, dict[str, float]]:
    """Measures the featurewise distance between two metadata distributions, and computes a p-value to evaluate its
        significance.

        Uses the Earth Mover's Distance and the Kolmogorov-Smirnov two-sample test, featurewise.

        Parameters
        ----------
        md0 : Mapping[str, list[Any] | NDArray[Any]]
            A set of arrays of values, indexed by metadata feature names, with one value per data example per feature.
        md1 : Mapping[str, list[Any] | NDArray[Any]]
            Another set of arrays of values, indexed by metadata feature names, with one value per data example per
            feature.

        Returns
        -------
        dict[str, KstestResult]
            A dictionary with keys corresponding to metadata feature names, and values that are KstestResult objects, as
            defined by scipy.stats.ks_2samp. These values also have two additional attributes: shift_magnitude and
            statistic_location. The first is the Earth Mover's Distance normalized by the interquartile range (IQR) of
            the reference, while the second is the value at which the KS statistic has its maximum, measured in
            IQR-normalized units relative to the median of the reference distribution.

        Examples
        --------
        Imagine we have 3 data examples, and that the corresponding metadata contains 2 features called time and
        altitude.

    >>> import numpy
    >>> md0 = {"time": [1.2, 3.4, 5.6], "altitude": [235, 6789, 101112]}
    >>> md1 = {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, 211101]}
    >>> md_out = meta_distribution_compare(md0, md1)
    >>> for k, v in md_out.items():
    >>>     print(k)
    >>>     for kv in v:
    >>>         print("\t", f"{kv}: {v[kv]:.3f}")
    time
             statistic_location: 0.444
             shift_magnitude: 2.700
             pvalue: 0.000
    altitude
             statistic_location: 0.478
             shift_magnitude: 0.749
             pvalue: 0.944
    """

    if (metadata_keys := md0.keys()) != md1.keys():
        raise ValueError(f"Both sets of metadata keys must be identical: {list(md0)}, {list(md1)}")

    mdc_dict = {}  # output dict
    for k in metadata_keys:
        mdc_dict.update({k: {}})

        x0, x1 = list(md0[k]), list(md1[k])

        allx = x0 + x1  # "+" sign concatenates lists.

        if not all(isinstance(allxi, numbers.Number) for allxi in allx):  # NB: np.nan *is* a number in this context.
            continue  # non-numeric features will return an empty dict for feature k

        # from Numerical Recipes in C, 3rd ed. p. 737. If too few points, warn and keep going.
        if np.sqrt(((N := len(x0)) * (M := len(x1))) / (N + M)) < 4:
            warnings.warn(
                f"Sample sizes of {N}, {M} for feature {k} will yield unreliable p-values from the KS test.",
                UserWarning,
            )

        xmin, xmax = min(allx), max(allx)
        if xmin == xmax:  # only one value in this feature, so fill in the obvious results for feature k
            mdc_dict[k].update({"statistic_location": 0.0, "shift_magnitude": 0.0, "pvalue": 1.0})
            continue

        ks_result = ks_2samp(x0, x1, method="asymp")
        dev = ks_result.statistic_location - xmin  #  pyright: ignore  (KSresult type)
        loc = dev / (xmax - xmin) if xmax > xmin else dev

        dX = iqr(x0)  # preferred value of dX, which is the scale of the the md0 values for feature k
        dX = (max(x0) - min(x0)) / 2.0 if dX == 0 else dX  # reasonable alternative value of dX, when iqr is zero.
        dX = 1.0 if dX == 0 else dX  # if dX is *still* zero, just avoid division by zero this way

        drift = emd(x0, x1) / dX

        mdc_dict[k].update({"statistic_location": loc, "shift_magnitude": drift, "pvalue": ks_result.pvalue})  #  pyright: ignore

    return mdc_dict
