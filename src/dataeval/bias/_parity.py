__all__ = []

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl

from dataeval import Metadata
from dataeval._experimental import experimental
from dataeval._helpers import factor_code_names, factors_excluding, is_metadata_like, resolve_label_axis
from dataeval._log import get_logger
from dataeval.core._parity import parity
from dataeval.protocols import AnnotatedDataset, MetadataLike
from dataeval.types import DictOutput, Evaluator, EvaluatorConfig, set_metadata

_logger = get_logger(__name__)

DEFAULT_PARITY_SCORE_THRESHOLD = 0.3
DEFAULT_PARITY_P_VALUE_THRESHOLD = 0.05


@experimental
@dataclass(frozen=True, repr=False)
class ParityOutput(DictOutput):
    """
    Output class for the :class:`.Parity` :term:`bias<Bias>` evaluator.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    Contains a polars DataFrame with Cramér's V scores and threshold flags.

    Attributes
    ----------
    factors : pl.DataFrame
        DataFrame with columns:
        - factor_name: str - Name of the metadata factor
        - score: float - Bias-Corrected Cramér's V statistic
        - p_value: float - P-value from G-test (Log-Likelihood Ratio)
        - is_significant: bool - True if score >= score_threshold AND p_value <= p_value_threshold
        - has_insufficient_data: bool - True if any cells have < 5 samples
    insufficient_data : dict[str, dict[str, dict[str, int]]]
        Dictionary flagging specific data subsets with low sample counts (< 5).
        Structure: {factor_name: {factor_level_name: {class_label: count}}}.

        The factor's level is named from its recorded encoding — ``"[0, 12.4)"`` for a
        binned factor, ``"rain"`` for a categorical one — so the entry says which subset
        to collect more of. A container carrying no encoding record falls back to the
        code as a string.
    """

    factors: pl.DataFrame
    insufficient_data: dict[str, dict[str, dict[str, int]]]


@experimental
class Parity(Evaluator):
    """
    Compute statistical parity using Bias-Corrected Cramér's V.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

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
    label : str or Sequence[str] or None, default None
        Factor(s) to condition on instead of the class labels. None reads
        :attr:`~dataeval.Metadata.class_labels`, which requires the metadata be viewed
        at its label level; naming a factor is how the same question is asked at a
        coarser view, where there is no single class label per row. Several names are
        combined into one composite axis.

    Attributes
    ----------
    metadata : MetadataLike
        Preprocessed metadata
    score_threshold : float
        Threshold for identifying highly correlated factors
    p_value_threshold : float
        P-value threshold for statistical significance
    label : str or Sequence[str] or None
        Factor(s) conditioned on instead of the class labels

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
        label : str or Sequence[str] or None, default None
            Factor(s) to condition on instead of the class labels. None reads
            :attr:`~dataeval.Metadata.class_labels`, which requires the metadata be
            viewed at its label level; naming a factor is how the same question is
            asked at a coarser view, where there is no single class label per row.
        """

        score_threshold: float = DEFAULT_PARITY_SCORE_THRESHOLD
        p_value_threshold: float = DEFAULT_PARITY_P_VALUE_THRESHOLD
        label: str | Sequence[str] | None = None

    metadata: MetadataLike
    score_threshold: float
    p_value_threshold: float
    label: str | Sequence[str] | None
    config: Config

    def __init__(
        self,
        score_threshold: float | None = None,
        p_value_threshold: float | None = None,
        label: str | Sequence[str] | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__(locals())

    @set_metadata(state=["score_threshold", "p_value_threshold", "label", "encoding_digest"])
    def evaluate(self, data: AnnotatedDataset[Any] | MetadataLike) -> ParityOutput:
        """
        Compute chi-square statistics for the dataset.

        Parameters
        ----------
        data : AnnotatedDataset[Any] or MetadataLike
            Either an annotated dataset (which will be converted to Metadata) or any object
            implementing the MetadataLike protocol.

        Returns
        -------
        ParityOutput
            DataFrame containing score, p_value, and correlation flags for each factor,
            along with insufficient data details.

        Examples
        --------
        Randomly creating some "continuous" and categorical variables using :func:`np.random.default_rng <numpy.random.default_rng>`

        >>> from dataeval import Metadata
        >>> metadata = Metadata(dataset)

        >>> parity = Parity()
        >>> result = parity.evaluate(metadata)
        >>> result.factors
        shape: (5, 5)
        ┌─────────────┬──────────┬────────────┬────────────────┬───────────────────────┐
        │ factor_name ┆ score    ┆ p_value    ┆ is_significant ┆ has_insufficient_data │
        │ ---         ┆ ---      ┆ ---        ┆ ---            ┆ ---                   │
        │ cat         ┆ f64      ┆ f64        ┆ bool           ┆ bool                  │
        ╞═════════════╪══════════╪════════════╪════════════════╪═══════════════════════╡
        │ angle       ┆ 0.123336 ┆ 0.183186   ┆ false          ┆ true                  │
        │ id          ┆ 0.0      ┆ 0.912633   ┆ false          ┆ true                  │
        │ location    ┆ 0.475116 ┆ 1.5062e-11 ┆ true           ┆ true                  │
        │ time_of_day ┆ 0.275172 ┆ 0.000526   ┆ false          ┆ true                  │
        │ weather     ┆ 0.147734 ┆ 0.123125   ┆ false          ┆ true                  │
        └─────────────┴──────────┴────────────┴────────────────┴───────────────────────┘
        """  # noqa: E501
        # Convert AnnotatedDataset to Metadata if needed
        if is_metadata_like(data):
            self.metadata = data
        else:
            self.metadata = Metadata(data)

        axis = resolve_label_axis(self.metadata, self.label)
        factor_data, factor_names, _ = factors_excluding(self.metadata, axis.excluded)
        class_labels = axis.values
        index2label = axis.names

        if not factor_names:
            raise ValueError("No factors found in provided metadata.")

        output = parity(factor_data, class_labels)

        # The factor's level is resolved the way the class label already was. This is the
        # only output that hands a user a bare code, and a bare code cannot be acted on:
        # knowing that `illum_lux = 3` is under-sampled says nothing about which lighting
        # to go and collect.
        level_names = factor_code_names(self.metadata, factor_data, factor_names)
        insufficient_data = {
            factor_names[k]: {
                level_names[k].get(int(vk), str(vk)): {index2label.get(vvk, str(vvk)): vvv for vvk, vvv in vv.items()}
                for vk, vv in v.items()
            }
            for k, v in output["insufficient_data"].items()
        }

        if insufficient_data:
            _logger.warning(
                f"Factors {list(insufficient_data)} did not meet the recommended "
                "5 occurrences for each value-label combination.",
            )

        # Create factors DataFrame - build as columnar data
        factor_name_col: list[str] = []
        score_col: list[float] = []
        p_value_col: list[float] = []
        is_significant_col: list[bool] = []
        has_insufficient_data_col: list[bool] = []

        for i, factor_name in enumerate(factor_names):
            score = float(output["scores"][i])
            p_value = float(output["p_values"][i])
            is_significant = bool(score >= self.score_threshold and p_value <= self.p_value_threshold)
            has_insufficient_data_flag = bool(factor_name in insufficient_data)

            factor_name_col.append(factor_name)
            score_col.append(score)
            p_value_col.append(p_value)
            is_significant_col.append(is_significant)
            has_insufficient_data_col.append(has_insufficient_data_flag)

        factors_df = pl.DataFrame(
            {
                "factor_name": factor_name_col,
                "score": score_col,
                "p_value": p_value_col,
                "is_significant": is_significant_col,
                "has_insufficient_data": has_insufficient_data_col,
            },
            schema={
                "factor_name": pl.Categorical("lexical"),
                "score": pl.Float64,
                "p_value": pl.Float64,
                "is_significant": pl.Boolean,
                "has_insufficient_data": pl.Boolean,
            },
        )

        return ParityOutput(factors=factors_df, insufficient_data=insufficient_data)
