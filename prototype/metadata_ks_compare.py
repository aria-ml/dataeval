from __future__ import annotations

from typing import Any, Mapping

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
        Another set of arrays of values, indexed by metadata feature names, with one value per data example per feature.

    Returns
    -------
    dict[str, KstestResult]
        A dictionary with keys corresponding to metadata feature names, and values that are KstestResult objects, as
        defined by scipy.stats.ks_2samp. These values also have two additional attributes: shift_magnitude and
        statistic_location. The first is the Earth Mover's Distance normalized by the interquartile range (IQR) of the
        reference, while the second is the value at which the KS statistic has its maximum, measured in IQR-normalized
        units relative to the median of the reference distribution.

    Examples
    --------
    Imagine we have 3 data examples, and that the corresponding metadata contains 2 features called time and altitude.

    >>> import numpy
    >>> md0 = {"time": [1.2, 3.4, 5.6], "altitude": [235, 6789, 101112]}
    >>> md1 = {"time": [7.8, 9.10, 11.12], "altitude": [532, 9876, 211101]}
    >>> md_out = meta_distribution_compare(md0, md1)
    >>> for k, v in md_out.items():
    >>>     print(k)
    >>>     for f in v._fields:
    >>>         print('\t',f, getattr(v, f))
    time
         statistic 1.0
         pvalue 0.0
    altitude
         statistic 0.33333333333333337
         pvalue 0.9444444444444444
    """

    metadata_keys = md0.keys()
    mdc_dict = {}
    for k in metadata_keys:
        mdc_dict.update({k: {}})

    for k in md0:
        x0, x1 = md0[k], md1[k]
        allx = list(x0) + list(x1)  # "+" sign concatenates lists.
        xmin = min(allx)
        xmax = max(allx)

        res = ks_2samp(x0, x1, method="asymp")
        dev = res.statistic_location - xmin  #  pyright: ignore
        loc = dev / (xmax - xmin) if xmax > xmin else dev

        dX = iqr(x0)  # preferred value of dX
        dX = (max(x0) - min(x0)) / 2.0 if dX == 0 else dX  # reasonable alternative value
        dX = 1.0 if dX == 0 else dX  # if alternative is still zero, just avoid division by zero this way

        drift = emd(x0, x1) / dX

        mdc_dict[k].update({"statistic_location": loc, "shift_magnitude": drift, "pvalue": res.pvalue})  #  pyright: ignore

    return mdc_dict
