__all__ = []

import logging
from dataclasses import dataclass
from typing import Any

import polars as pl

from dataeval import Metadata as _Metadata
from dataeval._helpers import _get_index2label
from dataeval.core import parity
from dataeval.protocols import AnnotatedDataset, Metadata
from dataeval.types import DictOutput, Evaluator, EvaluatorConfig, set_metadata

_logger = logging.getLogger(__name__)

DEFAULT_PARITY_SCORE_THRESHOLD = 0.3
DEFAULT_PARITY_P_VALUE_THRESHOLD = 0.05


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


class Parity(Evaluator):
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

    Using configuration:

    >>> config = Parity.Config(score_threshold=0.4, p_value_threshold=0.01)
    >>> parity = Parity(config=config)

    output = parity(metadata.binned_data, metadata.class_labels.tolist())
    """

    class Config(EvaluatorConfig):
        """
        Configuration for Parity evaluator.

        Attributes
        ----------
        score_threshold : float, default 0.3
            Threshold for identifying highly correlated factors.
        p_value_threshold : float, default 0.05
            P-value threshold for statistical significance.
        """

        score_threshold: float = DEFAULT_PARITY_SCORE_THRESHOLD
        p_value_threshold: float = DEFAULT_PARITY_P_VALUE_THRESHOLD

    metadata: Metadata
    score_threshold: float
    p_value_threshold: float
    config: Config

    def __init__(
        self,
        score_threshold: float | None = None,
        p_value_threshold: float | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__(locals())

    @set_metadata(state=["score_threshold", "p_value_threshold"])
    def evaluate(self, data: AnnotatedDataset[Any] | Metadata) -> ParityOutput:
        """
        Calculate chi-square statistics for the dataset.

        Parameters
        ----------
        data : AnnotatedDataset[Any] or Metadata
            Either an annotated dataset (which will be converted to Metadata) or any object
            implementing the Metadata protocol.

        Returns
        -------
        ParityOutput
            DataFrame containing score, p_value, and correlation flags for each factor,
            along with insufficient data details.

        Examples
        --------
        Randomly creating some "continuous" and categorical variables using ``np.random.default_rng``

        >>> from dataeval import Metadata
        >>> metadata = Metadata(dataset)

        >>> parity = Parity()
        >>> result = parity.evaluate(metadata)
        >>> result.factors
        shape: (5, 5)
        ┌─────────────┬──────────┬──────────┬───────────────┬───────────────────────┐
        │ factor_name ┆ score    ┆ p_value  ┆ is_correlated ┆ has_insufficient_data │
        │ ---         ┆ ---      ┆ ---      ┆ ---           ┆ ---                   │
        │ cat         ┆ f64      ┆ f64      ┆ bool          ┆ bool                  │
        ╞═════════════╪══════════╪══════════╪═══════════════╪═══════════════════════╡
        │ angle       ┆ 0.0      ┆ 0.43066  ┆ false         ┆ true                  │
        │ id          ┆ 0.0      ┆ 0.466239 ┆ false         ┆ true                  │
        │ location    ┆ 0.0      ┆ 0.707677 ┆ false         ┆ true                  │
        │ time_of_day ┆ 0.157489 ┆ 0.07135  ┆ false         ┆ true                  │
        │ weather     ┆ 0.0      ┆ 0.567789 ┆ false         ┆ false                 │
        └─────────────┴──────────┴──────────┴───────────────┴───────────────────────┘
        """
        # Convert AnnotatedDataset to Metadata if needed
        if isinstance(data, Metadata):
            self.metadata = data
        else:
            self.metadata = _Metadata(data)

        factor_names = self.metadata.factor_names
        class_labels = self.metadata.class_labels
        index2label = _get_index2label(self.metadata)

        if not factor_names:
            raise ValueError("No factors found in provided metadata.")

        output = parity(self.metadata.factor_data, class_labels)

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
            },
            schema={
                "factor_name": pl.Categorical("lexical"),
                "score": pl.Float64,
                "p_value": pl.Float64,
                "is_correlated": pl.Boolean,
                "has_insufficient_data": pl.Boolean,
            },
        )

        return ParityOutput(factors=factors_df, insufficient_data=insufficient_data)
