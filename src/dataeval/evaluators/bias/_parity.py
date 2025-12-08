from __future__ import annotations

__all__ = []

import logging
from dataclasses import dataclass
from typing import Any

import polars as pl

from dataeval.core import parity
from dataeval.data import Metadata
from dataeval.protocols import AnnotatedDataset
from dataeval.types import DictOutput, set_metadata

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParityOutput(DictOutput):
    """
    Output class for the :class:`.Parity` :term:`bias<Bias>` evaluator.

    Contains a polars DataFrame with Cramér's V scores and threshold flags.

    Attributes
    ----------
    factors : pl.DataFrame
        DataFrame with columns:
        - factor_name: str - Name of the metadata factor
        - score: float - Bias-Corrected Cramér's V statistic
        - p_value: float - P-value from G-test (Log-Likelihood Ratio)
        - is_correlated: bool - True if score >= score_threshold AND p_value <= p_value_threshold
        - has_insufficient_data: bool - True if any cells have < 5 samples
    insufficient_data : dict[str, dict[int, dict[str, int]]]
        Dictionary flagging specific data subsets with low sample counts (< 5).
        Structure: {factor_name: {factor_category_value: {class_label: count}}}.
    """

    factors: pl.DataFrame
    insufficient_data: dict[str, dict[int, dict[str, int]]]


class Parity:
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
    score_threshold : float, default 0.3
        Threshold for identifying highly correlated factors. Factors with Cramér's V
        above this threshold and p-value below p_value_threshold are considered
        highly correlated with class labels.
    p_value_threshold : float, default 0.05
        P-value threshold for statistical significance. Only factors with p-value
        below this threshold are considered for correlation flagging.

    Attributes
    ----------
    metadata : Metadata
        Preprocessed metadata
    score_threshold : float
        Threshold for identifying highly correlated factors
    p_value_threshold : float
        P-value threshold for statistical significance

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

    Examples
    --------
    Initialize the Parity class:

    >>> parity = Parity()

    Specifying custom thresholds:

    >>> parity = Parity(score_threshold=0.4, p_value_threshold=0.01)

    output = parity(metadata.binned_data, metadata.class_labels.tolist())
    """

    def __init__(
        self,
        score_threshold: float = 0.3,
        p_value_threshold: float = 0.05,
    ) -> None:
        """Initialize Parity evaluator."""
        self.metadata: Metadata
        self.score_threshold = score_threshold
        self.p_value_threshold = p_value_threshold

    @set_metadata(state=["score_threshold", "p_value_threshold"])
    def evaluate(self, data: AnnotatedDataset[Any] | Metadata) -> ParityOutput:
        """
        Calculate chi-square statistics for the dataset.

        Parameters
        ----------
        data : AnnotatedDataset[Any] or Metadata
            Either an annotated dataset (which will be converted to Metadata) or preprocessed Metadata directly.

        Returns
        -------
        ParityOutput
            DataFrame containing score, p_value, and correlation flags for each factor,
            along with insufficient data details.

        Examples
        --------
        Randomly creating some "continuous" and categorical variables using ``np.random.default_rng``

        >>> metadata = generate_random_metadata(
        ...     labels=["doctor", "artist", "teacher"],
        ...     factors={"age": [25, 30, 35, 45], "income": [50000, 65000, 80000], "gender": ["M", "F"]},
        ...     length=100,
        ...     random_seed=175,
        ... )

        >>> parity = Parity()
        >>> result = parity.evaluate(metadata)
        >>> result.factors
        shape: (3, 5)
        ┌─────────────┬──────────┬────────────┬───────────────┬───────────────────────┐
        │ factor_name ┆ score    ┆ p_value    ┆ is_correlated ┆ has_insufficient_data │
        │ ---         ┆ ---      ┆ ---        ┆ ---           ┆ ---                   │
        │ cat         ┆ f64      ┆ f64        ┆ bool          ┆ bool                  │
        ╞═════════════╪══════════╪════════════╪═══════════════╪═══════════════════════╡
        │ age         ┆ 0.445379 ┆ 4.8290e-8  ┆ true          ┆ true                  │
        │ income      ┆ 0.568195 ┆ 8.4062e-14 ┆ true          ┆ true                  │
        │ gender      ┆ 0.291057 ┆ 0.0055     ┆ false         ┆ false                 │
        └─────────────┴──────────┴────────────┴───────────────┴───────────────────────┘
        """
        # Convert AnnotatedDataset to Metadata if needed
        if isinstance(data, Metadata):
            self.metadata = data
        else:
            self.metadata = Metadata(data)

        factor_names = self.metadata.factor_names
        index2label = self.metadata.index2label

        if not factor_names:
            raise ValueError("No factors found in provided metadata.")

        output = parity(self.metadata.binned_data, self.metadata.class_labels.tolist())

        insufficient_data = {
            factor_names[k]: {vk: {index2label[vvk]: vvv for vvk, vvv in vv.items()} for vk, vv in v.items()}
            for k, v in output["insufficient_data"].items()
        }

        if insufficient_data:
            _logger.warning(
                f"Factors {list(insufficient_data)} did not meet the recommended "
                "5 occurrences for each value-label combination."
            )

        # Create factors DataFrame - build as columnar data
        factor_name_col: list[str] = []
        score_col: list[float] = []
        p_value_col: list[float] = []
        is_correlated_col: list[bool] = []
        has_insufficient_data_col: list[bool] = []

        for i, factor_name in enumerate(factor_names):
            score = float(output["scores"][i])
            p_value = float(output["p_values"][i])
            is_correlated = bool(score >= self.score_threshold and p_value <= self.p_value_threshold)
            has_insufficient_data_flag = bool(factor_name in insufficient_data)

            factor_name_col.append(factor_name)
            score_col.append(score)
            p_value_col.append(p_value)
            is_correlated_col.append(is_correlated)
            has_insufficient_data_col.append(has_insufficient_data_flag)

        factors_df = pl.DataFrame(
            {
                "factor_name": factor_name_col,
                "score": score_col,
                "p_value": p_value_col,
                "is_correlated": is_correlated_col,
                "has_insufficient_data": has_insufficient_data_col,
            }
        ).with_columns(
            pl.col("factor_name").cast(pl.Categorical),
        )

        return ParityOutput(factors=factors_df, insufficient_data=insufficient_data)
