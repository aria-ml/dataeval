__all__ = []

from collections import defaultdict
from collections.abc import Mapping
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from dataeval._experimental import experimental
from dataeval._log import get_logger
from dataeval.core._effective_n import pair_n, rescaled, validate_effective_n
from dataeval.types import Array1D, Array2D
from dataeval.utils._array import as_numpy
from dataeval.utils.scipy.stats import chi2_contingency, crosstab

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
        Array of p-values from Pearson's chi-square test of independence.
        Indicates the statistical significance of the calculated association.
    insufficient_data : Mapping[int, Mapping[int, Mapping[int, float]]]
        Dictionary flagging specific data subsets with low expected counts (< 5).
        Structure: {factor_index: {factor_category_value: {class_label: count}}}.
    """

    scores: NDArray[np.float64]
    p_values: NDArray[np.float64]
    insufficient_data: Mapping[int, Mapping[int, Mapping[int, float]]]


# Cochran's criterion for chi-square validity: no expected count below 1,
# and no more than 20% of expected counts below 5.
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
        Number of distinct entities for each column (class labels at index 0,
        factor columns at indices 1 to n). Used to scale contingency tables when
        factors are evaluated across hierarchical levels. If None, row counts are used.
        See Notes.

        .. versionadded:: 1.2

    Returns
    -------
    ParityResult
        A dictionary containing:

        - scores: NDArray[np.float64] - Array of bias-corrected Cramér's V statistics ranging from
          0 (independence) to 1 (perfect association).
        - p_values: NDArray[np.float64] - Array of p-values from Pearson's chi-square test of
          independence. Low p-values (< 0.05) indicate statistical significance.
        - insufficient_data: Mapping[int, Mapping[int, Mapping[int, float]]] - Nested dictionary
          containing expected cell counts (< 5) for factors breaching Cochran's criterion.
          Empty if the contingency table satisfies the criterion.

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
    2. Rescales the matrix to ``effective_n`` if provided.
    3. Performs Pearson's chi-square test of independence without continuity correction.
    4. Evaluates Cochran's criterion on expected cell counts (< 5).
    5. Computes Cramér's V with Bergsma's bias correction.

    Pearson's chi-square is used instead of the G-test because it maintains nominal
    type I error rates on sparse contingency tables (Larntz 1978; Koehler & Larntz 1980)
    and directly bounds Cramér's V within [0, 1]. Yates' continuity correction is
    disabled to avoid over-conservatism in test power and Cramér's V estimation.

    When ``effective_n`` is provided, the contingency table is rescaled prior to testing
    so that statistical significance and sufficiency checks reflect independent entity counts
    rather than repeated rows.

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
        contingency_matrix = results.count

        # Rescale contingency table to effective entity counts before statistical tests.
        contingency_matrix = rescaled(contingency_matrix, pair_n(effective_n_np, 0, i + 1))

        # Pearson's chi-square test of independence without Yates' continuity correction.
        chi_results = chi2_contingency(contingency_matrix, correction=False)
        chi_stat = chi_results.statistic
        p_val = chi_results.pvalue
        expected = chi_results.expected_freq

        # Flag factors breaching Cochran's sufficiency criterion.
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
