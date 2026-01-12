__all__ = []

import logging
from collections import defaultdict
from collections.abc import Mapping
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats.contingency import chi2_contingency, crosstab

from dataeval.types import Array1D, Array2D
from dataeval.utils.arrays import as_numpy

_logger = logging.getLogger(__name__)


class ParityResult(TypedDict):
    """
    Type definition for the output of the parity function.

    Attributes
    ----------
    scores : NDArray[np.float64]
        Array of Bias-Corrected Cramér's V statistics for each factor.
        Values range from 0.0 (independent/high parity) to 1.0 (perfect association).
    p_values : NDArray[np.float64]
        Array of p-values calculated via the G-test (Log-Likelihood Ratio).
        Indicates the statistical significance of the calculated association.
    insufficient_data : Mapping[int, Mapping[int, Mapping[int, int]]]
        Dictionary flagging specific data subsets with low sample counts (< 5).
        Structure: {factor_index: {factor_category_value: {class_label: count}}}.
    """

    scores: NDArray[np.float64]
    p_values: NDArray[np.float64]
    insufficient_data: Mapping[int, Mapping[int, Mapping[int, int]]]


def parity(
    factor_data: Array2D[int],
    class_labels: Array1D[int],
) -> ParityResult:
    """
    Calculate statistical parity using Bias-Corrected Cramér's V.

    This function measures the association between metadata factors and class labels
    to identify potential bias or spurious correlations. It assumes an equal distribution
    of metadata factors within the dataset.

    The calculation uses the G-test (Log-Likelihood Ratio) for the statistical test
    and applies the Bergsma (2013) bias correction to the Cramér's V statistic.
    This correction provides a more accurate estimate of association strength than
    standard Cramér's V, particularly for finite samples or large contingency tables.

    Parameters
    ----------
    factor_data : Array2D[int]
        Binned metadata factor values. Shape should be (n_samples, n_factors).
    class_labels : Array1D[int]
        Observed class labels. Shape should be (n_samples,).

    Returns
    -------
    ParityResult
        A dictionary containing:

        - scores: NDArray[np.float64] - Array of bias-corrected Cramér's V statistics ranging from
          0 (independence) to 1 (perfect association).
        - p_values: NDArray[np.float64] - Array of p-values from the G-test. Low p-values (< 0.05) indicate
          statistical significance.
        - insufficient_data: Mapping[int, Mapping[int, Mapping[int, int]]] - Nested dictionary flagging
          specific combinations with low sample counts (< 5).

          Sample structure: `{factor_index: {factor_category: {class_label: count}}}`.

    Notes
    -----
    **Interpretation:**
    - **0.0 - 0.1:** Negligible association (High Parity)
    - **0.1 - 0.3:** Weak association
    - **0.3 - 0.5:** Moderate association
    - **> 0.5:** Strong association (Potential Bias)

    **Methodology:**
    1. Constructs a contingency matrix for each factor against class labels.
    2. Identifies and flags cells with counts < 5 (insufficient data).
    3. Removes rows with zero sums to prevent calculation errors.
    4. Performs a G-test (Log-Likelihood Ratio) instead of Pearson's Chi-Squared.
    5. Computes Cramér's V with Bergsma's bias correction.

    References
    ----------
    Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T.
    Journal of the Korean Statistical Society, 42(3), 323-328.

    See Also
    --------
    balance
    """
    _logger.info("Starting parity calculation")

    factor_data_np = as_numpy(factor_data, dtype=np.intp, required_ndim=2)
    class_labels_np = as_numpy(class_labels, dtype=np.intp, required_ndim=1)

    _logger.debug("Factor data shape: %s, Class labels shape: %s", factor_data_np.shape, class_labels_np.shape)

    chi_scores = np.zeros(factor_data_np.shape[1])
    p_values = np.zeros_like(chi_scores)
    insufficient_ddict: defaultdict[int, defaultdict[int, dict[int, int]]] = defaultdict(lambda: defaultdict(dict))

    for i, col_data in enumerate(factor_data_np.T):
        # Builds a contingency matrix where entry at index (r,c) represents
        # the frequency of current_factor_name achieving value unique_factor_values[r]
        # at a data point with class c.
        results = crosstab(col_data, class_labels_np)
        contingency_matrix = as_numpy(results.count)  # type: ignore

        # Determines if any frequencies are too low
        counts = np.nonzero(contingency_matrix < 5)
        unique_factor_values = np.unique(col_data)
        for _factor, _class in zip(counts[0], counts[1]):
            int_factor, int_class = int(_factor), int(_class)
            if contingency_matrix[int_factor, int_class] > 0:
                factor_category = unique_factor_values[int_factor].item()
                class_count = contingency_matrix[int_factor, int_class].item()
                insufficient_ddict[i][factor_category][int_class] = class_count

        # This deletes rows containing only zeros,
        # because scipy.stats.chi2_contingency fails when there are rows containing only zeros.
        contingency_matrix = contingency_matrix[np.any(contingency_matrix, axis=1)]

        # Perform chi-square test using log-likelihood ratio (G-test)
        # https://en.wikipedia.org/wiki/G-test
        chi_results = chi2_contingency(contingency_matrix, lambda_="log-likelihood")
        chi_stat, p_val = cast(tuple[np.float64, np.float64], chi_results[:2])

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
