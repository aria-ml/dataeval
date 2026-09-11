__all__ = []

from collections import defaultdict
from collections.abc import Mapping
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats.contingency import chi2_contingency, crosstab

from dataeval._experimental import experimental
from dataeval._log import get_logger
from dataeval.core._effective_n import pair_n, rescaled, validate_effective_n
from dataeval.types import Array1D, Array2D
from dataeval.utils._array import as_numpy

_logger = get_logger(__name__)


class ParityResult(TypedDict):
    """
    Type definition for parity output.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    Attributes
    ----------
    scores : NDArray[np.float64]
        Array of Bias-Corrected Cramér's V statistics for each factor.
        Values range from 0.0 (independent/high parity) to 1.0 (perfect association).
    p_values : NDArray[np.float64]
        Array of p-values calculated via the G-test (Log-Likelihood Ratio).
        Indicates the statistical significance of the calculated association.
    insufficient_data : Mapping[int, Mapping[int, Mapping[int, float]]]
        Dictionary flagging specific data subsets with low sample counts (< 5).
        Structure: {factor_index: {factor_category_value: {class_label: count}}}.
        The count is fractional where ``effective_n`` scaled the table, since what is
        counted there is entities rather than rows; see :func:`parity`.
    """

    scores: NDArray[np.float64]
    p_values: NDArray[np.float64]
    insufficient_data: Mapping[int, Mapping[int, Mapping[int, float]]]


# Cochran's criterion for trusting the chi-square approximation, as he stated it: no expected
# count below 1, and no more than a fifth of them below 5. Both are properties of the
# *expected* counts, which is what makes them a statement about the approximation rather than
# about the sample. Observed counts answer a different question and answer it in both
# directions -- a table of 47/3/3/47 has two observed counts under five and expected counts of
# twenty-five throughout, where a perfectly associated 100/100/6 table has no observed count
# under five outside its structural zeros and a smallest expected count of 0.175.
_MIN_EXPECTED = 5.0
_FLOOR_EXPECTED = 1.0
_MAX_SPARSE_SHARE = 0.2


def _cochran_breach(expected: NDArray[np.float64]) -> bool:
    """Whether a table is too thin for the chi-square approximation to be trusted.

    References
    ----------
    Cochran, W. G. (1952). The chi-square test of goodness of fit. Annals of Mathematical
    Statistics, 23(3), 315-345.
    Cochran, W. G. (1954). Some methods for strengthening the common chi-square tests.
    Biometrics, 10(4), 417-451.
    """
    if expected.size == 0:
        return False
    return bool((expected < _FLOOR_EXPECTED).any() or (expected < _MIN_EXPECTED).mean() > _MAX_SPARSE_SHARE)


@experimental
def parity(  # noqa: C901
    factor_data: Array2D[int],
    class_labels: Array1D[int],
    effective_n: Array1D[int] | None = None,
) -> ParityResult:
    """
    Compute statistical parity using Bias-Corrected Cramér's V.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    This function measures the association between metadata factors and class labels
    to identify potential bias or spurious correlations. Both margins are conditioned on,
    so no distribution is assumed for either the factor or the class labels.

    The calculation uses Pearson's chi-square for the statistical test and applies the
    Bergsma (2013) bias correction to the Cramér's V statistic. This correction provides a
    more accurate estimate of association strength than standard Cramér's V, particularly
    for finite samples or large contingency tables.

    Parameters
    ----------
    factor_data : Array2D[int]
        Binned metadata factor values. Shape should be (n_samples, n_factors).
    class_labels : Array1D[int]
        Observed class labels. Shape should be (n_samples,).
    effective_n : Array1D[int] or None, default None
        How many distinct entities stand behind each column — the class labels at index 0
        and factor ``i`` of ``factor_data`` at index ``i+1``. None counts every row as its
        own observation, which is right whenever the rows are one level of one dataset. It
        is wrong where a column was replicated onto finer rows than it was measured at: a
        per-image factor read on detection rows takes one value per image, and counting a
        detection apiece multiplies the G-test's evidence by the fan-out. See Notes.

        .. versionadded:: 1.2

    Returns
    -------
    ParityResult
        A dictionary containing:

        - scores: NDArray[np.float64] - Array of bias-corrected Cramér's V statistics ranging from
          0 (independence) to 1 (perfect association).
        - p_values: NDArray[np.float64] - Array of p-values from the G-test. Low p-values (< 0.05) indicate
          statistical significance.
        - insufficient_data: Mapping[int, Mapping[int, Mapping[int, float]]] - Nested dictionary naming,
          for each factor whose table breaches Cochran's criterion, the cells whose *expected* count
          falls below 5. Empty for a factor whose table the approximation holds for, however few
          observations any one cell happens to hold.

          Sample structure: `{factor_index: {factor_category: {class_label: expected_count}}}`.

    See Also
    --------
    :class:`~dataeval.bias.Balance`

    Notes
    -----
    **Interpretation:**
    - **0.0 - 0.1:** Negligible association (High Parity)
    - **0.1 - 0.3:** Weak association
    - **0.3 - 0.5:** Moderate association
    - **> 0.5:** Strong association (Potential Bias)

    **Methodology:**
    1. Constructs a contingency matrix for each factor against class labels.
    2. Scales that matrix to ``effective_n`` where one is given.
    3. Performs Pearson's chi-square test of independence.
    4. Flags the factor where Cochran's criterion on the expected counts is breached.
    5. Computes Cramér's V with Bergsma's bias correction.

    **Why Pearson's chi-square rather than the G-test.** Both are asymptotically chi-square
    under independence and they disagree in finite samples, in one direction: the G-test is
    anti-conservative once cells thin out. Across 3000 simulations of independent data at a
    nominal 5%, it rejects 7.5% of the time at five observations per cell, 15.6% at two and a
    half, and 26.5% at two -- while Pearson holds 3.9-5.1% throughout. That is the regime a
    continuous factor auto-binned into sixteen or thirty-two levels lands in, so it is the
    ordinary case here and not an edge one. Larntz (1978) and Koehler & Larntz (1980) report
    the same ordering. Pearson is also the statistic Cramér's V is *defined* on -- phi-squared
    is ``X^2 / n`` -- so one choice settles both outputs; on a thin table the two differ by
    much more than their asymptotic equivalence suggests. The G statistic is not bounded by
    ``n`` the way ``X^2`` is, so a V read off it can exceed the 1.0 this documents as its
    maximum: a perfectly associated 5x5 table at n=20 reaches 1.06.

    **Why the continuity correction is off.** ``chi2_contingency`` applies Yates' correction to
    2x2 tables by default, and neither output wants it. Cramér's V is defined on the
    uncorrected statistic, and a perfect 2x2 association reads 0.77 with Yates where it must
    read 1.0. For the test it is the wrong correction as well: Yates approximates Fisher's
    exact test, which conditions on both margins being fixed, and a test of independence has
    neither fixed. Measured under independence at a nominal 5%, it rejects 2.4% of the time at
    n=40 and 4.0% at n=400, against 5.0-5.6% uncorrected -- power given away for a
    conservatism the design does not call for.

    **Why the scaling comes second.** All three outputs read the table's counts, and a
    replicated column inflates every one of them. The G-test statistic is linear in the
    total, so a fan-out of forty turns a p-value of 0.3 into 1e-50 and the test rejects
    independence on repetition alone. Cramér's V divides by ``n`` and so survives the
    scaling, but its Bergsma correction subtracts a term in ``1/(n-1)`` and under-corrects
    against an inflated ``n``. The insufficient-data flag compares raw counts against 5 and
    is *suppressed* by replication, which is the reverse of what it exists for: two entities
    seen forty times each reads as eighty observations. Scaling the table once, before any
    of them, puts all three on the same honest footing — and leaves the association itself
    untouched, since scaling every cell alike does not move the table's proportions.

    Flagged counts are fractional once scaled, because what they count is entities rather
    than rows.

    References
    ----------
    Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T.
    Journal of the Korean Statistical Society, 42(3), 323-328.

    Cochran, W. G. (1954). Some methods for strengthening the common chi-square tests.
    Biometrics, 10(4), 417-451.

    Larntz, K. (1978). Small-sample comparisons of exact levels for chi-squared goodness-of-fit
    statistics. Journal of the American Statistical Association, 73(362), 253-263.

    Koehler, K. J., & Larntz, K. (1980). An empirical investigation of goodness-of-fit statistics
    for sparse multinomials. Journal of the American Statistical Association, 75(370), 336-344.
    """
    _logger.info("Starting parity calculation")

    factor_data_np = as_numpy(factor_data, dtype=np.intp, required_ndim=2)
    class_labels_np = as_numpy(class_labels, dtype=np.intp, required_ndim=1)

    _logger.debug("Factor data shape: %s, Class labels shape: %s", factor_data_np.shape, class_labels_np.shape)

    chi_scores = np.zeros(factor_data_np.shape[1])
    p_values = np.zeros_like(chi_scores)
    insufficient_ddict: defaultdict[int, defaultdict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    unique_class_labels = np.unique(class_labels_np)
    # One entry per factor plus one for the class labels, which is the column list the pairs
    # below index into.
    effective_n_np = validate_effective_n(effective_n, factor_data_np.shape[1] + 1, factor_data_np.shape[0])

    for i, col_data in enumerate(factor_data_np.T):
        # Builds a contingency matrix where entry at index (r,c) represents
        # the frequency of current_factor_name achieving value unique_factor_values[r]
        # at a data point with class c.
        results = crosstab(col_data, class_labels_np)
        contingency_matrix = as_numpy(results.count)  # type: ignore

        # Scaled here, ahead of both readers below, so the sufficiency check and the test
        # agree about how much evidence this table holds. The pair is this factor against
        # the class labels, and takes the larger of their two counts.
        contingency_matrix = rescaled(contingency_matrix, pair_n(effective_n_np, 0, i + 1))

        # Pearson's chi-square, which is the statistic both outputs are defined on: the
        # p-value holds its nominal level in thin tables where the G-test does not, and
        # Cramér's V is a function of phi-squared = X^2 / n. Uncorrected, because Yates
        # adjusts the *test* and not phi-squared, and both readers below are better without
        # it -- see Notes.
        chi_results = chi2_contingency(contingency_matrix, correction=False)
        chi_stat, p_val = cast(tuple[np.float64, np.float64], chi_results[:2])
        expected = as_numpy(chi_results[3])

        # Cochran's criterion, read off the *expected* counts rather than the observed ones,
        # and reported only where it is breached -- see ``_cochran_breach``.
        unique_factor_values = np.unique(col_data)
        if _cochran_breach(expected):
            for _factor, _class in zip(*np.nonzero(expected < _MIN_EXPECTED), strict=False):
                int_factor, int_class = int(_factor), int(_class)
                factor_category = unique_factor_values[int_factor].item()
                class_label = int(unique_class_labels[int_class])
                insufficient_ddict[i][factor_category][class_label] = float(expected[int_factor, int_class])

        # Calculate Bias-Corrected Cramér's V
        # Based on Bergsma (2013)
        n = contingency_matrix.sum()
        r, k = contingency_matrix.shape

        if n > 1:
            # 1. Calculate phi-squared
            phi2 = chi_stat / n

            # 2. Correct phi-squared
            phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))

            # 3. Correct dimensions
            r_corr = r - ((r - 1) ** 2) / (n - 1)
            k_corr = k - ((k - 1) ** 2) / (n - 1)

            # 4. Calculate corrected score
            min_dim_corr = min((k_corr - 1), (r_corr - 1))

            # Avoid division by zero if corrected dimensions are too small
            if min_dim_corr > 0:
                chi_scores[i] = np.sqrt(phi2_corr / min_dim_corr)
            else:
                chi_scores[i] = 0.0
        else:
            chi_scores[i] = 0.0

        p_values[i] = p_val

    insufficient_data = {k: dict(v) for k, v in insufficient_ddict.items()}

    _logger.info(
        "Parity calculation complete: %d factors analyzed, mean Bias-Corrected Cramér's V=%.4f",
        factor_data_np.shape[1],
        np.mean(chi_scores),
    )
    _logger.debug("P-values: %s", p_values)
    if insufficient_data:
        _logger.warning("Found insufficient data for %d factor(s)", len(insufficient_data))

    return ParityResult(scores=chi_scores, p_values=p_values, insufficient_data=insufficient_data)
