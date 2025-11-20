from __future__ import annotations

__all__ = []

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from dataeval.core._parity import parity as _parity
from dataeval.data import Metadata
from dataeval.types import DictOutput, set_metadata


@dataclass(frozen=True)
class ParityOutput(DictOutput):
    """
    Output class for :func:`.parity` :term:`bias<Bias>` metrics.

    Attributes
    ----------
    score : NDArray[np.float64]
        chi-squared score(s) of the test
    p_value : NDArray[np.float64]
        p-value(s) of the test
    factor_names : Sequence[str]
        Names of each metadata factor
    insufficient_data: Mapping[str, Mapping[int, Mapping[str, int]]]
        Mapping of metadata factors with less than 5 class occurrences per value
    """

    score: NDArray[np.float64]
    p_value: NDArray[np.float64]
    factor_names: Sequence[str]
    insufficient_data: Mapping[str, Mapping[int, Mapping[str, int]]]

    def to_dataframe(self) -> pd.DataFrame:
        """Export results to pandas dataframe."""
        import pandas as pd

        data = {
            "factor_name": self.factor_names,
            "chi_square_score": self.score,
            "p_value": self.p_value,
        }
        return pd.DataFrame(data)


@set_metadata
def parity(metadata: Metadata) -> ParityOutput:
    """
    Calculate chi-square statistics to assess the linear relationship \
    between multiple factors and class labels.

    This function computes the chi-square statistic for each metadata factor to determine if there is
    a significant relationship between the factor values and class labels. The chi-square statistic is
    only valid for linear relationships. If non-linear relationships exist, use `balance`.

    Parameters
    ----------
    metadata : Metadata
        Preprocessed metadata

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

    Examples
    --------
    Randomly creating some "continuous" and categorical variables using ``np.random.default_rng``

    >>> metadata = generate_random_metadata(
    ...     labels=["doctor", "artist", "teacher"],
    ...     factors={
    ...         "age": [25, 30, 35, 45],
    ...         "income": [50000, 65000, 80000],
    ...         "gender": ["M", "F"]},
    ...     length=100,
    ...     random_seed=175)

    >>> parity(metadata)
    ParityOutput(score=array([7.357, 5.467, 0.515]), p_value=array([0.289, 0.243, 0.773]), factor_names=['age', 'income', 'gender'], insufficient_data={'age': {35: {'artist': 4}, 45: {'artist': 4, 'teacher': 3}}, 'income': {50000: {'artist': 3}}})
    """  # noqa: E501
    factor_names = metadata.factor_names
    index2label = metadata.index2label

    if not factor_names:
        raise ValueError("No factors found in provided metadata.")

    output = _parity(metadata.binned_data, metadata.class_labels.tolist(), return_insufficient_data=True)

    insufficient_data = {
        factor_names[k]: {vk: {index2label[vvk]: vvv for vvk, vvv in vv.items()} for vk, vv in v.items()}
        for k, v in output["insufficient_data"].items()
    }

    if insufficient_data:
        warnings.warn(
            f"Factors {list(insufficient_data)} did not meet the recommended "
            "5 occurrences for each value-label combination."
        )

    return ParityOutput(
        score=output["chi_scores"],
        p_value=output["p_values"],
        factor_names=metadata.factor_names,
        insufficient_data=insufficient_data,
    )
