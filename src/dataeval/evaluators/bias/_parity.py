from __future__ import annotations

__all__ = []

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from dataeval.core._parity import parity as _parity
from dataeval.data import Metadata
from dataeval.types import DictOutput, set_metadata

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParityOutput(DictOutput):
    """
    Output class for :func:`.parity` :term:`bias<Bias>` metrics.

    Attributes
    ----------
    score : NDArray[np.float64]
        Array of Bias-Corrected Cramér's V statistics for each factor.
        Values range from 0.0 (independent/high parity) to 1.0 (perfect association).
    p_value : NDArray[np.float64]
        Array of p-values calculated via the G-test (Log-Likelihood Ratio).
        Indicates the statistical significance of the calculated association.
    factor_names : Sequence[str]
        Names of each metadata factor
    insufficient_data: Mapping[str, Mapping[int, Mapping[str, int]]]
        Dictionary flagging specific data subsets with low sample counts (< 5).
        Structure: {factor_name: {factor_category_value: {class_label: count}}}.
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
            "cramers_v": self.score,
            "p_value": self.p_value,
        }
        return pd.DataFrame(data)


@set_metadata
def parity(metadata: Metadata) -> ParityOutput:
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
    metadata : Metadata
        Preprocessed metadata

    Returns
    -------
    ParityOutput
        Output dataclass containing:
        - score: Array of Bias-Corrected Cramér's V statistics (range 0.0 to 1.0).
        0 indicates independence (parity), 1 indicates perfect association.
        - p_value: Array of p-values from the G-test. Low p-values (< 0.05) indicate
        statistical significance.
        - factor_names: Names of the metadata factors analyzed.
        - insufficient_data: Nested dictionary flagging specific combinations with low sample counts (< 5).
        Structure: {factor_name: {factor_category: {class_label: count}}}.

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

    Examples
    --------
    Randomly creating some "continuous" and categorical variables using ``np.random.default_rng``

    >>> metadata = generate_random_metadata(
    ...     labels=["doctor", "artist", "teacher"],
    ...     factors={"age": [25, 30, 35, 45], "income": [50000, 65000, 80000], "gender": ["M", "F"]},
    ...     length=100,
    ...     random_seed=175,
    ... )

    >>> parity(metadata)
    ParityOutput(score=array([0.081, 0.086, 0.   ]), p_value=array([0.29 , 0.239, 0.773]), factor_names=['age', 'income', 'gender'], insufficient_data={'age': {35: {'artist': 4}, 45: {'artist': 4, 'teacher': 3}}, 'income': {50000: {'artist': 3}}})
    """  # noqa: E501
    factor_names = metadata.factor_names
    index2label = metadata.index2label

    if not factor_names:
        raise ValueError("No factors found in provided metadata.")

    output = _parity(metadata.binned_data, metadata.class_labels.tolist())

    insufficient_data = {
        factor_names[k]: {vk: {index2label[vvk]: vvv for vvk, vvv in vv.items()} for vk, vv in v.items()}
        for k, v in output["insufficient_data"].items()
    }

    if insufficient_data:
        _logger.warning(
            f"Factors {list(insufficient_data)} did not meet the recommended "
            "5 occurrences for each value-label combination."
        )

    return ParityOutput(
        score=output["scores"],
        p_value=output["p_values"],
        factor_names=metadata.factor_names,
        insufficient_data=insufficient_data,
    )
