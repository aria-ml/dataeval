__all__ = []

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

from dataeval import Metadata
from dataeval._helpers import factors_excluding, is_metadata_like, resolve_label_axis
from dataeval.core._bin import get_counts
from dataeval.core._diversity import diversity_shannon, diversity_simpson
from dataeval.protocols import AnnotatedDataset, MetadataLike
from dataeval.types import DictOutput, Evaluator, EvaluatorConfig, set_metadata

_DIVERSITY_FN_MAP = {"simpson": diversity_simpson, "shannon": diversity_shannon}

DEFAULT_DIVERSITY_THRESHOLD = 0.5
DEFAULT_DIVERSITY_METHOD = "simpson"


@dataclass(frozen=True, repr=False)
class DiversityOutput(DictOutput):
    """
    Output class for the :class:`.Diversity` :term:`bias<Bias>` evaluator.

    Contains two polars DataFrames with diversity scores and threshold flags.

    Attributes
    ----------
    factors : pl.DataFrame
        DataFrame with columns:
        - factor_name: str - Name of the metadata factor
        - diversity_value: float - Diversity score for this factor
        - is_low_diversity: bool - True if diversity_value <= threshold
    classwise : pl.DataFrame
        DataFrame with columns:
        - class_name: str - Name of the class
        - factor_name: str - Name of the metadata factor
        - diversity_value: float - Diversity score for this class-factor combination
        - is_low_diversity: bool - True if diversity_value <= threshold
    """

    factors: pl.DataFrame
    classwise: pl.DataFrame

    @property
    def plot_type(self) -> Literal["diversity"]:
        return "diversity"


class Diversity(Evaluator):
    """
    Computes diversity and classwise diversity for discrete/categorical variables.

    Uses user provided bins or the auto_bin_method in Metadata to discretize continuous variables.

    The method specified defines diversity as the inverse Simpson diversity index linearly rescaled to
    the unit interval [0, 1], or the normalized form of the Shannon entropy.

    diversity = 1 implies that samples are evenly distributed across a particular factor
    diversity = 0 implies that all samples belong to one category/bin

    Identifies factors with low diversity based on a threshold.

    Parameters
    ----------
    method : "simpson" or "shannon", default "simpson"
        The methodology used for defining diversity. When "simpson" is used,
        the index is linearly rescaled so that 1.0 represents maximum diversity
        (even distribution) and 0.0 represents minimum diversity (all samples in one bin).
    threshold : float, default 0.5
        Threshold for identifying low diversity. Factors with diversity values
        at or below this threshold are flagged as having low diversity.
    label : str or Sequence[str] or None, default None
        Factor(s) to condition on instead of the class labels. None reads
        :attr:`~dataeval.Metadata.class_labels`, which requires the metadata be viewed
        at its label level; naming a factor is how the same question is asked at a
        coarser view, where there is no single class label per row. Several names are
        combined into one composite axis.

    Attributes
    ----------
    metadata : MetadataLike
        Preprocessed metadata from the last evaluate() call.
    method : Literal["simpson", "shannon"]
        The methodology used for defining diversity
    threshold : float
        Threshold for identifying low diversity factors
    label : str or Sequence[str] or None
        Factor(s) conditioned on instead of the class labels

    See Also
    --------
    :func:`scipy.stats.entropy`

    Notes
    -----
    - The expression is undefined for q=1, but it approaches the Shannon entropy in the limit.
    - If there is only one category, the diversity index takes a value of 0.
    - Factors with diversity values <= threshold represent low diversity and are flagged.

    References
    ----------
    [1] Diversity and evenness: A unifying notation and its consequences.
        Hill, M. O. (1973). Ecology, 54(2), 427-432.
    [2] Indices of diversity and evenness.
        Heip, C. H. R., Herman, P. M. J., & Soetaert, K. (1998). Oceanis, 24(4), 61-87.

    Examples
    --------
    Initialize the Diversity class:

    >>> diversity = Diversity()

    Specifying custom method and threshold:

    >>> diversity = Diversity(method="shannon", threshold=0.6)

    Using configuration:

    >>> config = Diversity.Config(method="shannon", threshold=0.6)
    >>> diversity = Diversity(config=config)
    """

    class Config(EvaluatorConfig):
        """
        Configuration for Diversity evaluator.

        Attributes
        ----------
        method : {"simpson", "shannon"}, default "simpson"
            The methodology used for defining diversity.
        threshold : float, default 0.5
            Threshold for identifying low diversity.
        label : str or Sequence[str] or None, default None
            Factor(s) to condition on instead of the class labels. None reads
            :attr:`~dataeval.Metadata.class_labels`, which requires the metadata be
            viewed at its label level; naming a factor is how the same question is
            asked at a coarser view, where there is no single class label per row.
        """

        method: Literal["simpson", "shannon"] = DEFAULT_DIVERSITY_METHOD
        threshold: float = DEFAULT_DIVERSITY_THRESHOLD
        label: str | Sequence[str] | None = None

    metadata: MetadataLike
    method: Literal["simpson", "shannon"]
    threshold: float
    label: str | Sequence[str] | None
    config: Config

    def __init__(
        self,
        method: Literal["simpson", "shannon"] | None = None,
        threshold: float | None = None,
        label: str | Sequence[str] | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__(locals())

    @set_metadata(state=["method", "threshold", "label", "encoding_digest"])
    def evaluate(self, data: AnnotatedDataset[Any] | MetadataLike) -> DiversityOutput:  # noqa: C901
        """
        Compute diversity and classwise diversity for the dataset.

        Parameters
        ----------
        data : AnnotatedDataset[Any] or MetadataLike
            Either an annotated dataset (which will be converted to Metadata)
            or any object implementing the MetadataLike protocol.

        Returns
        -------
        DiversityOutput
            Two DataFrames containing diversity scores and low diversity flags:
            - factors: Factor-level diversity scores
            - classwise: Class-factor-level diversity scores

        Example
        -------
        Compute the diversity index of metadata and class labels

        >>> from dataeval import Metadata
        >>> metadata = Metadata(dataset)

        >>> diversity = Diversity(method="simpson", threshold=0.5)
        >>> result = diversity.evaluate(metadata)
        >>> result.factors
        shape: (6, 3)
        ┌─────────────┬─────────────────┬──────────────────┐
        │ factor_name ┆ diversity_value ┆ is_low_diversity │
        │ ---         ┆ ---             ┆ ---              │
        │ cat         ┆ f64             ┆ bool             │
        ╞═════════════╪═════════════════╪══════════════════╡
        │ class_label ┆ 0.983706        ┆ false            │
        │ angle       ┆ 0.89455         ┆ false            │
        │ id          ┆ 0.984866        ┆ false            │
        │ location    ┆ 0.824033        ┆ false            │
        │ time_of_day ┆ 0.903475        ┆ false            │
        │ weather     ┆ 0.950688        ┆ false            │
        └─────────────┴─────────────────┴──────────────────┘

        >>> result.classwise
        shape: (20, 4)
        ┌────────────┬─────────────┬─────────────────┬──────────────────┐
        │ class_name ┆ factor_name ┆ diversity_value ┆ is_low_diversity │
        │ ---        ┆ ---         ┆ ---             ┆ ---              │
        │ cat        ┆ cat         ┆ f64             ┆ bool             │
        ╞════════════╪═════════════╪═════════════════╪══════════════════╡
        │ person     ┆ angle       ┆ 0.575269        ┆ false            │
        │ person     ┆ id          ┆ 0.912791        ┆ false            │
        │ person     ┆ location    ┆ 0.034991        ┆ true             │
        │ person     ┆ time_of_day ┆ 0.532468        ┆ false            │
        │ person     ┆ weather     ┆ 0.833333        ┆ false            │
        │ …          ┆ …           ┆ …               ┆ …                │
        │ plane      ┆ angle       ┆ 0.797153        ┆ false            │
        │ plane      ┆ id          ┆ 0.973154        ┆ false            │
        │ plane      ┆ location    ┆ 0.683403        ┆ false            │
        │ plane      ┆ time_of_day ┆ 0.875622        ┆ false            │
        │ plane      ┆ weather     ┆ 0.918288        ┆ false            │
        └────────────┴─────────────┴─────────────────┴──────────────────┘
        """
        # Convert AnnotatedDataset to Metadata if needed
        if is_metadata_like(data):
            self.metadata = data
        else:
            self.metadata = Metadata(data)

        if not self.metadata.factor_names:
            raise ValueError("No factors found in provided metadata.")

        if self.method not in _DIVERSITY_FN_MAP:
            raise ValueError(f"Invalid method '{self.method}'. Supported methods are '{list(_DIVERSITY_FN_MAP)}'.")

        diversity_fn = _DIVERSITY_FN_MAP[self.method]
        # The axis is whatever is being conditioned on: the class labels by default, or
        # the named factor(s), which is the only way to ask this at a view above the
        # label level. A factor serving as the axis is dropped from the factors measured
        # against it.
        axis = resolve_label_axis(self.metadata, self.label)
        factor_data, factor_names, _ = factors_excluding(self.metadata, axis.excluded)
        class_lbl = axis.values
        index2label = axis.names

        class_labels_with_binned_data = np.hstack((class_lbl[:, np.newaxis], factor_data))
        cnts = get_counts(class_labels_with_binned_data)
        num_bins = np.bincount(np.nonzero(cnts)[1])
        diversity_index = diversity_fn(cnts, num_bins)

        u_classes = np.unique(class_lbl)
        num_factors = len(factor_names)
        classwise_div = np.full((len(u_classes), num_factors), np.nan)
        for idx, cls in enumerate(u_classes):
            subset_mask = class_lbl == cls
            cls_cnts = get_counts(factor_data[subset_mask], min_num_bins=cnts.shape[0])
            classwise_div[idx, :] = diversity_fn(cls_cnts, num_bins[1:])

        # Create factors DataFrame
        # diversity_index[0] is class_labels, [1:] are the metadata factors
        # Include class_label as the first factor (index 0), then all metadata factors
        all_factor_names = [axis.label] + list(factor_names)
        factors_df = pl.DataFrame(
            {
                "factor_name": all_factor_names,
                "diversity_value": diversity_index,
                "is_low_diversity": (diversity_index <= self.threshold).astype(bool),
            },
            schema={
                "factor_name": pl.Categorical("lexical"),
                "diversity_value": pl.Float64,
                "is_low_diversity": pl.Boolean,
            },
        )

        # Create classwise DataFrame - build as columnar data
        class_name_col: list[str] = []
        factor_name_col: list[str] = []
        diversity_value_col: list[float] = []
        is_low_diversity_col: list[bool] = []

        for class_idx in range(classwise_div.shape[0]):
            class_name = index2label.get(int(u_classes[class_idx]), str(u_classes[class_idx]))
            for factor_idx in range(num_factors):
                div_value = classwise_div[class_idx, factor_idx]
                if not np.isnan(div_value):
                    class_name_col.append(class_name)
                    factor_name_col.append(factor_names[factor_idx])
                    diversity_value_col.append(float(div_value))
                    is_low_diversity_col.append(bool(div_value <= self.threshold))

        classwise_df = pl.DataFrame(
            {
                "class_name": class_name_col,
                "factor_name": factor_name_col,
                "diversity_value": diversity_value_col,
                "is_low_diversity": is_low_diversity_col,
            },
            schema={
                "class_name": pl.Categorical("lexical"),
                "factor_name": pl.Categorical("lexical"),
                "diversity_value": pl.Float64,
                "is_low_diversity": pl.Boolean,
            },
        )

        return DiversityOutput(factors=factors_df, classwise=classwise_df)
