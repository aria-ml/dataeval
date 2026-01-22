__all__ = []

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

from dataeval import Metadata as _Metadata
from dataeval.core._bin import get_counts
from dataeval.core._diversity import diversity_shannon, diversity_simpson
from dataeval.protocols import AnnotatedDataset, Metadata
from dataeval.types import DictOutput, set_metadata
from dataeval.utils.data import _get_index2label

_DIVERSITY_FN_MAP = {"simpson": diversity_simpson, "shannon": diversity_shannon}


@dataclass(frozen=True)
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


class Diversity:
    """
    Computes diversity and classwise diversity for discrete/categorical variables through
    standard histogram binning, for continuous variables.

    The method specified defines diversity as the inverse Simpson diversity index linearly rescaled to
    the unit interval, or the normalized form of the Shannon entropy.

    diversity = 1 implies that samples are evenly distributed across a particular factor
    diversity = 0 implies that all samples belong to one category/bin

    Identifies factors with low diversity based on a threshold.

    Parameters
    ----------
    method : "simpson" or "shannon", default "simpson"
        The methodology used for defining diversity
    threshold : float, default 0.5
        Threshold for identifying low diversity. Factors with diversity values
        at or below this threshold are flagged as having low diversity.

    Attributes
    ----------
    metadata : Metadata
        Preprocessed metadata from the last evaluate() call.
    method : Literal["simpson", "shannon"]
        The methodology used for defining diversity
    threshold : float
        Threshold for identifying low diversity factors

    Notes
    -----
    - The expression is undefined for q=1, but it approaches the Shannon entropy in the limit.
    - If there is only one category, the diversity index takes a value of 0.
    - Factors with diversity values <= threshold represent low diversity and are flagged.

    Examples
    --------
    Initialize the Diversity class:

    >>> diversity = Diversity()

    Specifying custom method and threshold:

    >>> diversity = Diversity(method="shannon", threshold=0.6)

    See Also
    --------
    scipy.stats.entropy
    """

    def __init__(
        self,
        method: Literal["simpson", "shannon"] = "simpson",
        threshold: float = 0.5,
    ) -> None:
        self.metadata: Metadata
        self.method: Literal["simpson", "shannon"] = method
        self.threshold = threshold

    @set_metadata(state=["method", "threshold"])
    def evaluate(self, data: AnnotatedDataset[Any] | Metadata) -> DiversityOutput:
        """
        Compute diversity and classwise diversity for the dataset.

        Parameters
        ----------
        data : AnnotatedDataset[Any] or Metadata
            Either an annotated dataset (which will be converted to Metadata)
            or any object implementing the Metadata protocol.

        Returns
        -------
        DiversityOutput
            Two DataFrames containing diversity scores and low diversity flags:
            - factors: Factor-level diversity scores
            - classwise: Class-factor-level diversity scores

        Example
        -------
        Compute the diversity index of metadata and class labels

        >>> metadata = generate_random_metadata(
        ...     labels=["doctor", "artist", "teacher"],
        ...     factors={"age": [25, 30, 35, 45], "income": [50000, 65000, 80000], "gender": ["M", "F"]},
        ...     length=100,
        ...     random_seed=175,
        ... )

        >>> diversity = Diversity(method="simpson", threshold=0.5)
        >>> result = diversity.evaluate(metadata)
        >>> result.factors
        shape: (4, 3)
        ┌─────────────┬─────────────────┬──────────────────┐
        │ factor_name ┆ diversity_value ┆ is_low_diversity │
        │ ---         ┆ ---             ┆ ---              │
        │ cat         ┆ f64             ┆ bool             │
        ╞═════════════╪═════════════════╪══════════════════╡
        │ class_label ┆ 0.937608        ┆ false            │
        │ age         ┆ 0.907669        ┆ false            │
        │ gender      ┆ 0.992826        ┆ false            │
        │ income      ┆ 0.954334        ┆ false            │
        └─────────────┴─────────────────┴──────────────────┘

        >>> result.classwise
        shape: (9, 4)
        ┌────────────┬─────────────┬─────────────────┬──────────────────┐
        │ class_name ┆ factor_name ┆ diversity_value ┆ is_low_diversity │
        │ ---        ┆ ---         ┆ ---             ┆ ---              │
        │ cat        ┆ cat         ┆ f64             ┆ bool             │
        ╞════════════╪═════════════╪═════════════════╪══════════════════╡
        │ doctor     ┆ age         ┆ 0.619268        ┆ false            │
        │ doctor     ┆ gender      ┆ 0.832507        ┆ false            │
        │ doctor     ┆ income      ┆ 0.269775        ┆ true             │
        │ artist     ┆ age         ┆ 0.556777        ┆ false            │
        │ artist     ┆ gender      ┆ 0.715294        ┆ false            │
        │ artist     ┆ income      ┆ 0.334096        ┆ true             │
        │ teacher    ┆ age         ┆ 0.477477        ┆ true             │
        │ teacher    ┆ gender      ┆ 0.86722         ┆ false            │
        │ teacher    ┆ income      ┆ 0.703209        ┆ false            │
        └────────────┴─────────────┴─────────────────┴──────────────────┘
        """
        # Convert AnnotatedDataset to Metadata if needed
        if isinstance(data, Metadata):
            self.metadata = data
        else:
            self.metadata = _Metadata(data)

        if not self.metadata.factor_names:
            raise ValueError("No factors found in provided metadata.")

        if self.method not in _DIVERSITY_FN_MAP:
            raise ValueError(f"Invalid method '{self.method}'. Supported methods are '{list(_DIVERSITY_FN_MAP)}'.")

        diversity_fn = _DIVERSITY_FN_MAP[self.method]
        factor_data = self.metadata.factor_data
        factor_names = self.metadata.factor_names
        class_lbl = self.metadata.class_labels
        index2label = _get_index2label(self.metadata)

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
        all_factor_names = ["class_label"] + list(factor_names)
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
            class_name = index2label.get(class_idx, str(class_idx))
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
