from __future__ import annotations

from typing import TypedDict

__all__ = []

import logging
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.stats.contingency import chi2_contingency, crosstab

from dataeval.types import Array1D, Array2D
from dataeval.utils._array import as_numpy

_logger = logging.getLogger(__name__)


class ParityResult(TypedDict):
    """
    Type definition for parity output.

    Attributes
    ----------
    chi_scores : NDArray[np.float64]
        Array of chi-squared statistics for each factor
    p_values : NDArray[np.float64]
        Array of p-values for each factor
    insufficient_data : Mapping[int, Mapping[int, Mapping[int, int]]]
        Mapping of factors to categories to classes with insufficient data counts
    """

    chi_scores: NDArray[np.float64]
    p_values: NDArray[np.float64]
    insufficient_data: Mapping[int, Mapping[int, Mapping[int, int]]]


def parity(
    factor_data: Array2D[int],
    class_labels: Array1D[int],
    *,
    return_insufficient_data: bool = False,
) -> ParityResult:
    """
    Calculate chi-square statistics to assess the linear relationship \
    between multiple factors and class labels.

    This function computes the chi-square statistic for each metadata factor to determine if there is
    a significant relationship between the factor values and class labels. The chi-square statistic is
    only valid for linear relationships. If non-linear relationships exist, use `balance`.

    Parameters
    ----------
    factor_data: Array2D[int]
        Binned metadata factor values. Can be a 2D list, or array-like object.
    class_labels: Array1D[int]
        Observed class labels. Can be a 1D list, or array-like object.

    Returns
    -------
    ParityResult
        Mapping with keys:
        - chi_scores : NDArray[np.float64] - Array of chi-squared statistics for each factor
        - p_values : NDArray[np.float64] - Array of p-values for each factor
        - insufficient_data : Mapping[int, Mapping[int, Mapping[int, int]]] - Mapping of factors to categories to
        classes with insufficient data counts

    Raises
    ------
    Warning
        If any cell in the contingency matrix has a value between 0 and 5, a warning is issued because this can
        lead to inaccurate chi-square calculations. It is recommended to ensure that each label co-occurs with
        factor values either 0 times or at least 5 times.

    Notes
    -----
    - A high score with a low p-value suggests that a metadata factor is strongly correlated with a class label.
    - The function creates a contingency matrix for each factor, where each entry represents the frequency of a
      specific factor value co-occurring with a particular class label.
    - Rows containing only zeros in the contingency matrix are removed before performing the chi-square test
      to prevent errors in the calculation.

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

        chi_scores[i], p_values[i] = chi2_contingency(contingency_matrix)[:2]  # type: ignore

    insufficient_data = {k: dict(v) for k, v in insufficient_ddict.items()}

    _logger.info(
        "Parity calculation complete: %d factors analyzed, mean chi-score=%.4f",
        factor_data_np.shape[1],
        np.mean(chi_scores),
    )
    _logger.debug("P-values: %s", p_values)
    if insufficient_data:
        _logger.warning("Found insufficient data for %d factor(s)", len(insufficient_data))

    return ParityResult(chi_scores=chi_scores, p_values=p_values, insufficient_data=insufficient_data)
