from __future__ import annotations

from typing import Literal, overload

__all__ = []

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats.contingency import chi2_contingency, crosstab

from dataeval.utils._array import as_numpy


@overload
def parity(
    binned_data: NDArray[np.intp],
    class_labels: Sequence[int],
    *,
    return_insufficient_data: Literal[False] = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


@overload
def parity(
    binned_data: NDArray[np.intp],
    class_labels: Sequence[int],
    *,
    return_insufficient_data: Literal[True],
) -> tuple[NDArray[np.float64], NDArray[np.float64], Mapping[int, Mapping[int, Mapping[int, int]]]]: ...


def parity(
    binned_data: NDArray[np.intp],
    class_labels: Sequence[int],
    *,
    return_insufficient_data: bool = False,
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64]]
    | tuple[NDArray[np.float64], NDArray[np.float64], Mapping[int, Mapping[int, Mapping[int, int]]]]
):
    """
    Calculate chi-square statistics to assess the linear relationship \
    between multiple factors and class labels.

    This function computes the chi-square statistic for each metadata factor to determine if there is
    a significant relationship between the factor values and class labels. The chi-square statistic is
    only valid for linear relationships. If non-linear relationships exist, use `balance`.

    Parameters
    ----------
    binned_data: NDArray[np.intp]
        Binned metadata factor values
    class_labels: Sequence[int]
        Observed class labels

    Returns
    -------
    ParityOutput[NDArray[np.float64]]
        Arrays of length (num_factors) whose (i)th element corresponds to the
        chi-square score and :term:`p-value<P-Value>` for the relationship between factor i and
        the class labels in the dataset.

    Raises
    ------
    Warning
        If any cell in the contingency matrix has a value between 0 and 5, a warning is issued because this can
        lead to inaccurate chi-square calculations. It is recommended to ensure that each label co-occurs with
        factor values either 0 times or at least 5 times.

    Note
    ----
    - A high score with a low p-value suggests that a metadata factor is strongly correlated with a class label.
    - The function creates a contingency matrix for each factor, where each entry represents the frequency of a
      specific factor value co-occurring with a particular class label.
    - Rows containing only zeros in the contingency matrix are removed before performing the chi-square test
      to prevent errors in the calculation.

    See Also
    --------
    balance
    """
    chi_scores = np.zeros(binned_data.shape[1])
    p_values = np.zeros_like(chi_scores)
    insufficient_data: defaultdict[int, defaultdict[int, dict[int, int]]] = defaultdict(lambda: defaultdict(dict))
    for i, col_data in enumerate(binned_data.T):
        # Builds a contingency matrix where entry at index (r,c) represents
        # the frequency of current_factor_name achieving value unique_factor_values[r]
        # at a data point with class c.
        results = crosstab(col_data, class_labels)
        contingency_matrix = as_numpy(results.count)  # type: ignore

        # Determines if any frequencies are too low
        counts = np.nonzero(contingency_matrix < 5)
        unique_factor_values = np.unique(col_data)
        for _factor, _class in zip(counts[0], counts[1]):
            int_factor, int_class = int(_factor), int(_class)
            if contingency_matrix[int_factor, int_class] > 0:
                factor_category = unique_factor_values[int_factor].item()
                class_count = contingency_matrix[int_factor, int_class].item()
                insufficient_data[i][factor_category][int_class] = class_count

        # This deletes rows containing only zeros,
        # because scipy.stats.chi2_contingency fails when there are rows containing only zeros.
        contingency_matrix = contingency_matrix[np.any(contingency_matrix, axis=1)]

        chi_scores[i], p_values[i] = chi2_contingency(contingency_matrix)[:2]  # type: ignore

    if return_insufficient_data:
        return chi_scores, p_values, insufficient_data

    return chi_scores, p_values
