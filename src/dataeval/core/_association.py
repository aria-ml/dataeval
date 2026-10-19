"""
Shared building blocks for association statistics on contingency tables.

These helpers are used by :func:`~dataeval.core.parity` and by the label co-occurrence
statistics, so that Cochran's sufficiency criterion, the Bergsma bias correction and the
multiple-testing adjustment are each defined exactly once.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Cochran's criterion for chi-square validity: no expected count below 1,
# and no more than 20% of expected counts below 5.
MIN_EXPECTED = 5.0
FLOOR_EXPECTED = 1.0
MAX_SPARSE_SHARE = 0.2


def cochran_breach(expected: NDArray[np.float64]) -> bool:
    """
    Whether a table is too thin for the chi-square approximation to be trusted.

    Parameters
    ----------
    expected : NDArray[np.float64]
        Expected cell counts under independence, of any shape.

    Returns
    -------
    bool
        True if any expected count is below 1, or more than 20% of expected counts are
        below 5. An empty table is never a breach.

    References
    ----------
    Cochran, W. G. (1952). The chi-square test of goodness of fit. Annals of Mathematical
    Statistics, 23(3), 315-345.
    Cochran, W. G. (1954). Some methods for strengthening the common chi-square tests.
    Biometrics, 10(4), 417-451.

    Examples
    --------
    >>> cochran_breach(np.array([[10.0, 12.0], [11.0, 9.0]]))
    False
    >>> cochran_breach(np.array([[10.0, 0.5], [11.0, 9.0]]))
    True
    """
    if expected.size == 0:
        return False
    return bool((expected < FLOOR_EXPECTED).any() or (expected < MIN_EXPECTED).mean() > MAX_SPARSE_SHARE)


def cochran_breach_2x2(n_a: ArrayLike, n_b: ArrayLike, n_total: ArrayLike) -> NDArray[np.bool_]:
    """
    Evaluate Cochran's criterion for 2x2 presence tables, vectorized over many pairs.

    For two binary indicators present in ``n_a`` and ``n_b`` of ``n_total`` samples, the
    four expected cell counts under independence follow from the margins alone, so no
    table has to be built. The result matches :func:`cochran_breach` applied to each
    pair's expected table. With four cells, a single expected count below 5 already
    exceeds the 20% share, so the criterion reduces to "any expected count below 5".

    A table with no samples, or with a zero margin, has zero or undefined expected counts
    and is reported as a breach.

    Parameters
    ----------
    n_a, n_b : ArrayLike
        Number of samples in which each indicator is present.
    n_total : ArrayLike
        Total number of samples. Broadcast against ``n_a`` and ``n_b``.

    Returns
    -------
    NDArray[np.bool_]
        True where the 2x2 table breaches Cochran's criterion.

    Examples
    --------
    >>> cochran_breach_2x2([50, 4], [40, 30], 100).tolist()
    [False, True]
    """
    a = np.asarray(n_a, dtype=np.float64)
    b = np.asarray(n_b, dtype=np.float64)
    n = np.asarray(n_total, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.stack(
            np.broadcast_arrays(
                a * b / n,
                a * (n - b) / n,
                (n - a) * b / n,
                (n - a) * (n - b) / n,
            ),
        )
    # Negated comparisons, so that undefined (NaN) expected counts are treated as breaches
    below_floor = ~(expected >= FLOOR_EXPECTED)
    below_min = ~(expected >= MIN_EXPECTED)
    return np.asarray(below_floor.any(axis=0) | (below_min.mean(axis=0) > MAX_SPARSE_SHARE))


def bias_corrected_cramers_v(
    chi_stat: ArrayLike,
    n: ArrayLike,
    r: ArrayLike,
    k: ArrayLike,
) -> NDArray[np.float64] | np.float64:
    """
    Compute Cramér's V with the Bergsma (2013) bias correction.

    Works on scalars or arrays (broadcast together), so a single contingency table and
    thousands of 2x2 tables share one definition. Where the sample is too small for the
    correction to be defined (``n <= 1``, or the corrected dimension is not positive),
    the score is 0.

    Parameters
    ----------
    chi_stat : ArrayLike
        Pearson chi-square statistic of the table.
    n : ArrayLike
        Total count of the table (the effective count, if the table was rescaled).
    r, k : ArrayLike
        Number of rows and columns of the table.

    Returns
    -------
    NDArray[np.float64] or np.float64
        The corrected score, 0 for independence and approaching 1 for perfect
        association. A scalar for scalar inputs, an array otherwise.

    References
    ----------
    Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T.
    Journal of the Korean Statistical Society, 42(3), 323-328.

    Examples
    --------
    >>> round(float(bias_corrected_cramers_v(100.0, 100, 2, 2)), 6)
    1.0
    >>> float(bias_corrected_cramers_v(0.0, 100, 2, 2))
    0.0
    >>> float(bias_corrected_cramers_v(5.0, 1, 2, 2))
    0.0
    """
    chi = np.asarray(chi_stat, dtype=np.float64)
    n_ = np.asarray(n, dtype=np.float64)
    r_ = np.asarray(r, dtype=np.float64)
    k_ = np.asarray(k, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        n_less_one = n_ - 1
        # 1. phi-squared, corrected for the expected value under independence
        phi2_corr = np.maximum(0.0, chi / n_ - ((k_ - 1) * (r_ - 1)) / n_less_one)
        # 2. dimensions, corrected for the same bias
        r_corr = r_ - ((r_ - 1) ** 2) / n_less_one
        k_corr = k_ - ((k_ - 1) ** 2) / n_less_one
        min_dim_corr = np.minimum(k_corr - 1, r_corr - 1)
        score = np.sqrt(phi2_corr / min_dim_corr)

    defined = (n_ > 1) & (min_dim_corr > 0)
    # ``[()]`` turns a 0-d result into a scalar and leaves arrays as they are
    return np.where(defined, score, 0.0)[()]


def benjamini_hochberg(p_values: ArrayLike, n_tests: int | None = None) -> NDArray[np.float64]:
    """
    Adjust p-values to control the false discovery rate (Benjamini-Hochberg).

    ``n_tests`` lets the correction account for tests that were run but whose p-values
    are not passed in. Omitted tests are treated as having p = 1: they can never be
    rejected, but they still count toward the number of comparisons. This is valid and
    slightly conservative, and it keeps the cost proportional to the p-values supplied.

    Parameters
    ----------
    p_values : ArrayLike
        One-dimensional p-values in [0, 1].
    n_tests : int or None, default None
        Total number of tests performed. Defaults to the number of p-values supplied.

    Returns
    -------
    NDArray[np.float64]
        Adjusted p-values, in the order of the input, each at least its raw p-value
        and at most 1.

    Raises
    ------
    ValueError
        If the p-values are not one-dimensional or not within [0, 1] (NaN included), or
        if ``n_tests`` is smaller than the number of p-values.

    References
    ----------
    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a
    practical and powerful approach to multiple testing. Journal of the Royal
    Statistical Society: Series B, 57(1), 289-300.

    Examples
    --------
    >>> np.round(benjamini_hochberg([0.01, 0.04, 0.03]), 6).tolist()
    [0.03, 0.04, 0.04]

    Two p-values out of four tests, the other two not reported:

    >>> np.round(benjamini_hochberg([0.01, 0.04], n_tests=4), 6).tolist()
    [0.04, 0.08]
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"p_values must be one-dimensional, got shape {p.shape}")
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError("p_values must all lie within [0, 1]")
    m = p.size if n_tests is None else int(n_tests)
    if m < p.size:
        raise ValueError(f"n_tests ({m}) is smaller than the number of p-values ({p.size})")
    if p.size == 0:
        return p.copy()

    order = np.argsort(p, kind="stable")
    scaled = p[order] * m / np.arange(1, p.size + 1)
    # Enforce monotonicity from the largest p-value down, then cap at 1
    adjusted_sorted = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    adjusted = np.empty_like(p)
    adjusted[order] = adjusted_sorted
    return adjusted
