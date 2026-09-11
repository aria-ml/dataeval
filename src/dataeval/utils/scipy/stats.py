"""SciPy stats wrappers with explicit return typing."""

__all__ = [
    "AndersonKsampResult",
    "BwsTestResult",
    "Chi2ContingencyResult",
    "CramerVonMisesResult",
    "CrosstabResult",
    "KstestResult",
    "MannwhitneyuResult",
    "SignificanceResult",
    "anderson_ksamp",
    "bws_test",
    "chi2_contingency",
    "cramervonmises_2samp",
    "crosstab",
    "ks_2samp",
    "mannwhitneyu",
]

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import anderson_ksamp as _scipy_anderson_ksamp
from scipy.stats import bws_test as _scipy_bws_test
from scipy.stats import cramervonmises_2samp as _scipy_cramervonmises_2samp
from scipy.stats import ks_2samp as _scipy_ks_2samp
from scipy.stats import mannwhitneyu as _scipy_mannwhitneyu
from scipy.stats.contingency import chi2_contingency as _scipy_chi2_contingency
from scipy.stats.contingency import crosstab as _scipy_crosstab


class Chi2ContingencyResult(NamedTuple):
    """Result of chi2_contingency with explicit typing.

    Attributes
    ----------
    statistic : float
        The test statistic.
    pvalue : float
        The p-value of the test.
    dof : int
        Degrees of freedom.
    expected_freq : NDArray[np.float64]
        The expected frequencies based on marginal sums.
    """

    statistic: float
    pvalue: float
    dof: int
    expected_freq: NDArray[np.float64]


class CrosstabResult(NamedTuple):
    """Result of crosstab with explicit typing.

    Attributes
    ----------
    elements : tuple[NDArray[Any], ...]
        Tuple of unique values for each input array.
    count : NDArray[np.intp]
        Contingency table of counts.
    """

    elements: tuple[NDArray[Any], ...]
    count: NDArray[np.intp]  # pyright: ignore[reportIncompatibleMethodOverride]


class KstestResult(NamedTuple):
    """Result of Kolmogorov-Smirnov test with explicit typing.

    Attributes
    ----------
    statistic : float
        The KS test statistic.
    pvalue : float
        The associated p-value.
    statistic_location : float
        Value corresponding with the KS statistic.
    statistic_sign : int
        Sign of the difference at statistic_location (+1 or -1).
    """

    statistic: float
    pvalue: float
    statistic_location: float
    statistic_sign: int


class SignificanceResult(NamedTuple):
    """Result of hypothesis test returning a test statistic and p-value.

    Attributes
    ----------
    statistic : float
        The test statistic.
    pvalue : float
        The associated p-value.
    """

    statistic: float
    pvalue: float


MannwhitneyuResult = SignificanceResult
CramerVonMisesResult = SignificanceResult
BwsTestResult = SignificanceResult
AndersonKsampResult = SignificanceResult


def chi2_contingency(
    observed: ArrayLike,
    *,
    correction: bool = True,
    lambda_: float | str | None = None,
) -> Chi2ContingencyResult:
    """Perform chi-square test of independence with typed result.

    Wrapper for :func:`scipy.stats.contingency.chi2_contingency`.

    Parameters
    ----------
    observed : ArrayLike
        Contingency table.
    correction : bool, default True
        Whether to apply Yates' correction for continuity (for 2x2 tables).
    lambda_ : float or str or None, default None
        Power-divergence statistic parameter. None uses Pearson's chi-square.

    Returns
    -------
    Chi2ContingencyResult
        NamedTuple with fields ``statistic``, ``pvalue``, ``dof``, and ``expected_freq``.

    Raises
    ------
    TypeError
        If any returned field does not match the expected type.
    """
    raw = _scipy_chi2_contingency(observed, correction=correction, lambda_=lambda_)

    stat = getattr(raw, "statistic", raw[0])
    pval = getattr(raw, "pvalue", raw[1])
    dof = getattr(raw, "dof", raw[2])
    expected = getattr(raw, "expected_freq", raw[3])

    if not isinstance(stat, (float, np.floating, int, np.integer)):
        raise TypeError(f"Expected statistic to be numeric, got {type(stat).__name__}")
    if not isinstance(pval, (float, np.floating, int, np.integer)):
        raise TypeError(f"Expected pvalue to be numeric, got {type(pval).__name__}")
    if not isinstance(dof, (int, np.integer)):
        raise TypeError(f"Expected dof to be an integer, got {type(dof).__name__}")

    expected_arr = np.asarray(expected, dtype=np.float64)

    return Chi2ContingencyResult(
        statistic=float(stat),
        pvalue=float(pval),
        dof=int(dof),
        expected_freq=expected_arr,
    )


def crosstab(
    *args: ArrayLike,
    levels: Sequence[Any] | None = None,
    sparse: bool = False,
) -> CrosstabResult:
    """Compute contingency table of counts with typed result.

    Wrapper for :func:`scipy.stats.contingency.crosstab`.

    Parameters
    ----------
    *args : ArrayLike
        Input data arrays.
    levels : Sequence or None, default None
        User-specified category levels for each input.
    sparse : bool, default False
        Whether to return a sparse array (only False is supported for typed array output).

    Returns
    -------
    CrosstabResult
        NamedTuple with fields ``elements`` and ``count``.

    Raises
    ------
    TypeError
        If returned elements or counts do not match expected types.
    """
    raw = _scipy_crosstab(*args, levels=levels, sparse=sparse)

    elements = getattr(raw, "elements", raw[0])
    count = getattr(raw, "count", raw[1])

    if not isinstance(elements, (tuple, list)):
        raise TypeError(f"Expected elements to be sequence of arrays, got {type(elements).__name__}")

    elements_tuple = tuple(np.asarray(elem) for elem in elements)
    count_arr = np.asarray(count, dtype=np.intp)

    return CrosstabResult(elements=elements_tuple, count=count_arr)


def ks_2samp(
    data1: ArrayLike,
    data2: ArrayLike,
    alternative: str = "two-sided",
    method: str = "auto",
    *,
    axis: int = 0,
) -> KstestResult:
    """Perform Kolmogorov-Smirnov test on two samples with typed result.

    Wrapper for :func:`scipy.stats.ks_2samp`.

    Parameters
    ----------
    data1, data2 : ArrayLike
        Input sample data.
    alternative : {"two-sided", "less", "greater"}, default "two-sided"
        Defines the null and alternative hypotheses.
    method : {"auto", "exact", "asymp"}, default "auto"
        Method used for calculating the p-value.
    axis : int, default 0
        Axis along which to compute the test.

    Returns
    -------
    KstestResult
        NamedTuple with fields ``statistic``, ``pvalue``, ``statistic_location``,
        and ``statistic_sign``.

    Raises
    ------
    TypeError
        If any returned field does not match the expected type.
    """
    raw = _scipy_ks_2samp(data1, data2, alternative=alternative, method=method, axis=axis)

    stat = getattr(raw, "statistic", raw[0])
    pval = getattr(raw, "pvalue", raw[1])
    stat_loc = getattr(raw, "statistic_location", raw[2] if len(raw) > 2 else 0.0)
    stat_sign = getattr(raw, "statistic_sign", raw[3] if len(raw) > 3 else 1)

    if not isinstance(stat, (float, np.floating, int, np.integer)):
        raise TypeError(f"Expected statistic to be numeric, got {type(stat).__name__}")
    if not isinstance(pval, (float, np.floating, int, np.integer)):
        raise TypeError(f"Expected pvalue to be numeric, got {type(pval).__name__}")
    if not isinstance(stat_loc, (float, np.floating, int, np.integer)):
        raise TypeError(f"Expected statistic_location to be numeric, got {type(stat_loc).__name__}")
    if not isinstance(stat_sign, (int, np.integer)):
        raise TypeError(f"Expected statistic_sign to be an integer, got {type(stat_sign).__name__}")

    return KstestResult(
        statistic=float(stat),
        pvalue=float(pval),
        statistic_location=float(stat_loc),
        statistic_sign=int(stat_sign),
    )


def mannwhitneyu(
    x: ArrayLike,
    y: ArrayLike,
    *,
    use_continuity: bool = True,
    alternative: str = "two-sided",
    axis: int = 0,
    method: str = "auto",
) -> MannwhitneyuResult:
    """Perform Mann-Whitney U rank test on two samples with typed result.

    Wrapper for :func:`scipy.stats.mannwhitneyu`.

    Parameters
    ----------
    x, y : ArrayLike
        Input sample data.
    use_continuity : bool, default True
        Whether to apply continuity correction.
    alternative : {"two-sided", "less", "greater"}, default "two-sided"
        Defines the alternative hypothesis.
    axis : int, default 0
        Axis along which to compute the test.
    method : {"auto", "asymptotic", "exact"}, default "auto"
        Method used for calculating the p-value.

    Returns
    -------
    MannwhitneyuResult
        NamedTuple with fields ``statistic`` and ``pvalue``.

    Raises
    ------
    TypeError
        If any returned field does not match the expected type.
    """
    raw = _scipy_mannwhitneyu(
        x,
        y,
        use_continuity=use_continuity,
        alternative=alternative,
        axis=axis,
        method=method,
    )

    stat = getattr(raw, "statistic", raw[0])
    pval = getattr(raw, "pvalue", raw[1])

    if not isinstance(stat, (float, np.floating, int, np.integer)):
        raise TypeError(f"Expected statistic to be numeric, got {type(stat).__name__}")
    if not isinstance(pval, (float, np.floating, int, np.integer)):
        raise TypeError(f"Expected pvalue to be numeric, got {type(pval).__name__}")

    return MannwhitneyuResult(
        statistic=float(stat),
        pvalue=float(pval),
    )


def cramervonmises_2samp(
    x: ArrayLike,
    y: ArrayLike,
    *,
    method: str = "auto",
    axis: int = 0,
) -> CramerVonMisesResult:
    """Perform Cramér-von Mises two-sample test with typed result.

    Wrapper for :func:`scipy.stats.cramervonmises_2samp`.
    """
    raw = _scipy_cramervonmises_2samp(x, y, method=method, axis=axis)

    stat = getattr(raw, "statistic", raw[0] if isinstance(raw, tuple) else None)
    pval = getattr(raw, "pvalue", raw[1] if isinstance(raw, tuple) else None)

    if not isinstance(stat, (float, np.floating, int, np.integer)):
        raise TypeError(f"Expected statistic to be numeric, got {type(stat).__name__}")
    if not isinstance(pval, (float, np.floating, int, np.integer)):
        raise TypeError(f"Expected pvalue to be numeric, got {type(pval).__name__}")

    return CramerVonMisesResult(statistic=float(stat), pvalue=float(pval))


def bws_test(
    x: ArrayLike,
    y: ArrayLike,
    *,
    alternative: str = "two-sided",
    method: Any = None,
    axis: int = 0,
) -> BwsTestResult:
    """Perform Baumgartner-Weiss-Schindler test on two samples with typed result.

    Wrapper for :func:`scipy.stats.bws_test`.
    """
    kwargs: dict[str, Any] = {}
    if method is not None:
        kwargs["method"] = method
    raw = _scipy_bws_test(x, y, alternative=alternative, axis=axis, **kwargs)

    stat = getattr(raw, "statistic", None)
    pval = getattr(raw, "pvalue", None)

    if not isinstance(stat, (float, np.floating, int, np.integer)):
        raise TypeError(f"Expected statistic to be numeric, got {type(stat).__name__}")
    if not isinstance(pval, (float, np.floating, int, np.integer)):
        raise TypeError(f"Expected pvalue to be numeric, got {type(pval).__name__}")

    return BwsTestResult(statistic=float(stat), pvalue=float(pval))


def anderson_ksamp(
    samples: Sequence[ArrayLike],
    **kwargs: Any,
) -> AndersonKsampResult:
    """Perform Anderson-Darling test for k-samples with typed result.

    Wrapper for :func:`scipy.stats.anderson_ksamp`.
    """
    raw = _scipy_anderson_ksamp(samples, **kwargs)

    stat = getattr(raw, "statistic", raw[0] if isinstance(raw, tuple) else None)
    pval = getattr(raw, "pvalue", raw[1] if isinstance(raw, tuple) else None)

    if not isinstance(stat, (float, np.floating, int, np.integer)):
        raise TypeError(f"Expected statistic to be numeric, got {type(stat).__name__}")
    if not isinstance(pval, (float, np.floating, int, np.integer)):
        raise TypeError(f"Expected pvalue to be numeric, got {type(pval).__name__}")

    return AndersonKsampResult(statistic=float(stat), pvalue=float(pval))
