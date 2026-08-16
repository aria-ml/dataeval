__all__ = []

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

from dataeval import Metadata
from dataeval._helpers import factors_excluding, is_metadata_like, resolve_label_axis
from dataeval.core._mutual_info import mutual_info, mutual_info_classwise
from dataeval.exceptions import DeprecatedWarning
from dataeval.protocols import AnnotatedDataset, MetadataLike
from dataeval.types import DictOutput, Evaluator, EvaluatorConfig, set_metadata

DEFAULT_BALANCE_NUM_NEIGHBORS = 5
DEFAULT_BALANCE_CLASS_IMBALANCE_THRESHOLD = 0.3
DEFAULT_BALANCE_FACTOR_CORRELATION_THRESHOLD = 0.5


@dataclass(frozen=True, repr=False)
class BalanceOutput(DictOutput):
    """
    Output class for the :class:`.Balance` :term:`bias<Bias>` evaluator.

    Contains three polars DataFrames with normalized mutual information scores and threshold flags.

    Attributes
    ----------
    balance : pl.DataFrame
        DataFrame with global class-to-factor normalized mutual information:

        - factor_name: str - Name of the metadata factor. Includes "class_label"
          which represents the self-information (always 1.0).
        - mi_value: float - Share of the class label's entropy this factor accounts
          for, corrected for chance: 1.0 for a factor that determines the class
          outright, 0.0 for one that explains no more than its own cardinality would
          by chance. Ranks factors of comparable cardinality directly; a factor taking
          several hundred values is scored conservatively by comparison.
    factors : pl.DataFrame
        DataFrame with inter-factor normalized mutual information correlations:

        - factor1: str - Name of the first factor
        - factor2: str - Name of the second factor
        - mi_value: float - Normalized mutual information value
        - is_correlated: bool - True if mi_value > factor_correlation_threshold
    classwise : pl.DataFrame
        DataFrame with per-class-to-factor normalized mutual information. Unlike
        ``balance``, this excludes "class_label", whose mutual information with a
        single class is 1.0 by construction:

        - class_name: str - Name of the class
        - factor_name: str - Name of the metadata factor
        - mi_value: float - Normalized mutual information value
        - is_imbalanced: bool - True if mi_value > class_imbalance_threshold
    """

    balance: pl.DataFrame
    factors: pl.DataFrame
    classwise: pl.DataFrame

    @property
    def plot_type(self) -> Literal["balance"]:
        return "balance"


class Balance(Evaluator):
    """
    Computes normalized mutual information (NMI) between factors (class label, metadata, label/image properties).

    Identifies imbalanced classes and highly correlated metadata factors based on
    NMI thresholds.

    Parameters
    ----------
    num_neighbors : int, default 5
        Deprecated, removed in 1.2. Every factor reaching this evaluator holds bin
        indices, which takes the discrete estimator path where a neighborhood size is
        never consulted, so this value has no effect on the result. Setting it warns.
    class_imbalance_threshold : float, default 0.3
        Threshold for identifying imbalanced classes. Classes with NMI above this
        threshold with any metadata factor are considered imbalanced.
    factor_correlation_threshold : float, default 0.5
        Threshold for identifying highly correlated metadata factors. Factor pairs
        with NMI above this threshold are considered highly correlated.
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
    num_neighbors : int
        Deprecated, removed in 1.2. Has no effect; see Parameters.
    class_imbalance_threshold : float
        Threshold for identifying imbalanced classes
    factor_correlation_threshold : float
        Threshold for identifying highly correlated metadata factors
    label : str or Sequence[str] or None
        Factor(s) conditioned on instead of the class labels

    See Also
    --------
    :func:`sklearn.feature_selection.mutual_info_classif`
    :func:`sklearn.feature_selection.mutual_info_regression`
    :func:`sklearn.metrics.mutual_info_score`

    Notes
    -----
    We use `mutual_info_classif` from sklearn since class label is categorical.
    `mutual_info_classif` outputs are consistent up to O(1e-4) and depend on a random
    seed. MI is computed differently for categorical and continuous variables, and
    in all cases normalized or transformed to [0, 1] prior to being returned.

    All three DataFrames are corrected for the mutual information a factor's cardinality
    alone would produce, and they differ in what that corrected value is divided by.
    ``balance`` and ``classwise`` divide by the class entropy, so their factors can be
    ranked against one another. ``factors`` compares two metadata factors, where neither
    side is privileged, and divides by the smaller of the two entropies. Values are
    therefore comparable within each DataFrame but not across them; see
    :func:`~dataeval.core.mutual_info`.

    References
    ----------
    [1] Information theoretic measures for clusterings comparison: Variants, properties, normalization and correction
        for chance.
        Vinh, N. X., Epps, J., & Bailey, J. (2010). Journal of Machine Learning Research, 11, 2837-2854.
        https://jmlr.org/papers/v11/vinh10a.html
    [2] Estimating mutual information.
        Kraskov, A., Stogbauer, H., & Grassberger, P. (2004). Physical Review E, 69(6), 066138.
        https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138
    [3] Mutual information between discrete and continuous data sets.
        Ross, B. C. (2014). PLOS ONE, 9(2), e87357.
        https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357

    Examples
    --------
    Initialize the Balance class:

    >>> balance = Balance()

    Specifying custom thresholds:

    >>> balance = Balance(class_imbalance_threshold=0.2, factor_correlation_threshold=0.6)

    Using configuration:

    >>> config = Balance.Config(class_imbalance_threshold=0.2, factor_correlation_threshold=0.6)
    >>> balance = Balance(config=config)
    """

    class Config(EvaluatorConfig):
        """
        Configuration for Balance evaluator.

        Attributes
        ----------
        num_neighbors : int, default 5
            Deprecated, removed in 1.2. Has no effect; see :class:`.Balance`.
        class_imbalance_threshold : float, default 0.3
            Threshold for identifying imbalanced classes.
        factor_correlation_threshold : float, default 0.5
            Threshold for identifying highly correlated metadata factors.
        label : str or Sequence[str] or None, default None
            Factor(s) to condition on instead of the class labels. None reads
            :attr:`~dataeval.Metadata.class_labels`, which requires the metadata be
            viewed at its label level; naming a factor is how the same question is
            asked at a coarser view, where there is no single class label per row.
        """

        num_neighbors: int = DEFAULT_BALANCE_NUM_NEIGHBORS
        class_imbalance_threshold: float = DEFAULT_BALANCE_CLASS_IMBALANCE_THRESHOLD
        factor_correlation_threshold: float = DEFAULT_BALANCE_FACTOR_CORRELATION_THRESHOLD
        label: str | Sequence[str] | None = None

    metadata: MetadataLike
    num_neighbors: int
    class_imbalance_threshold: float
    factor_correlation_threshold: float
    label: str | Sequence[str] | None
    config: Config

    def __init__(
        self,
        num_neighbors: int | None = None,
        class_imbalance_threshold: float | None = None,
        factor_correlation_threshold: float | None = None,
        label: str | Sequence[str] | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__(locals())
        if self.num_neighbors != DEFAULT_BALANCE_NUM_NEIGHBORS:
            warnings.warn(
                "Balance.num_neighbors is deprecated and will be removed in 1.2. It has no "
                "effect: every factor reaching the estimator arrives already binned, which "
                "takes the discrete path where a neighborhood size is never consulted.",
                DeprecatedWarning,
                stacklevel=2,
            )

    @set_metadata(state=["num_neighbors", "class_imbalance_threshold", "factor_correlation_threshold", "label"])
    def evaluate(self, data: AnnotatedDataset[Any] | MetadataLike) -> BalanceOutput:  # noqa: C901
        """
        Compute normalized mutual information between factors and identify imbalanced classes.

        Parameters
        ----------
        data : AnnotatedDataset[Any] or MetadataLike
            Either an annotated dataset (which will be converted to Metadata)
            or any object implementing the MetadataLike protocol.

        Returns
        -------
        BalanceOutput
            Three DataFrames containing NMI scores and threshold flags:

            - balance: Global class-to-factor mutual information
            - factors: Inter-factor mutual information
            - classwise: Per-class-to-factor mutual information

        Example
        -------
        Return balance (NMI) of factors with class_labels

        >>> from dataeval import Metadata
        >>> metadata = Metadata(dataset)

        >>> balance = Balance()
        >>> result = balance.evaluate(metadata)

        Reading the column as a ranking: where an image was taken accounts for a quarter of
        what the class label tells you and when it was taken for a twelfth, while the
        weather, the camera angle and the image's own id account for none of it. A model
        trained here could reach for the location instead of the object.

        >>> result.balance
        shape: (6, 2)
        ┌─────────────┬──────────┐
        │ factor_name ┆ mi_value │
        │ ---         ┆ ---      │
        │ cat         ┆ f64      │
        ╞═════════════╪══════════╡
        │ class_label ┆ 1.0      │
        │ angle       ┆ 0.010253 │
        │ id          ┆ 0.0      │
        │ location    ┆ 0.244383 │
        │ time_of_day ┆ 0.080863 │
        │ weather     ┆ 0.015113 │
        └─────────────┴──────────┘

        >>> result.factors
        shape: (20, 4)
        ┌─────────────┬─────────────┬──────────┬───────────────┐
        │ factor1     ┆ factor2     ┆ mi_value ┆ is_correlated │
        │ ---         ┆ ---         ┆ ---      ┆ ---           │
        │ cat         ┆ cat         ┆ f64      ┆ bool          │
        ╞═════════════╪═════════════╪══════════╪═══════════════╡
        │ angle       ┆ id          ┆ 0.017837 ┆ false         │
        │ angle       ┆ location    ┆ 0.071866 ┆ false         │
        │ angle       ┆ time_of_day ┆ 0.014648 ┆ false         │
        │ angle       ┆ weather     ┆ 0.001868 ┆ false         │
        │ id          ┆ angle       ┆ 0.017837 ┆ false         │
        │ …           ┆ …           ┆ …        ┆ …             │
        │ time_of_day ┆ weather     ┆ 0.007897 ┆ false         │
        │ weather     ┆ angle       ┆ 0.001868 ┆ false         │
        │ weather     ┆ id          ┆ 0.0      ┆ false         │
        │ weather     ┆ location    ┆ 0.084927 ┆ false         │
        │ weather     ┆ time_of_day ┆ 0.007897 ┆ false         │
        └─────────────┴─────────────┴──────────┴───────────────┘

        >>> result.classwise
        shape: (20, 4)
        ┌────────────┬─────────────┬──────────┬───────────────┐
        │ class_name ┆ factor_name ┆ mi_value ┆ is_imbalanced │
        │ ---        ┆ ---         ┆ ---      ┆ ---           │
        │ cat        ┆ cat         ┆ f64      ┆ bool          │
        ╞════════════╪═════════════╪══════════╪═══════════════╡
        │ boat       ┆ angle       ┆ 0.0      ┆ false         │
        │ boat       ┆ id          ┆ 0.0      ┆ false         │
        │ boat       ┆ location    ┆ 0.123615 ┆ false         │
        │ boat       ┆ time_of_day ┆ 0.064551 ┆ false         │
        │ boat       ┆ weather     ┆ 0.003375 ┆ false         │
        │ …          ┆ …           ┆ …        ┆ …             │
        │ plane      ┆ angle       ┆ 0.0      ┆ false         │
        │ plane      ┆ id          ┆ 0.0      ┆ false         │
        │ plane      ┆ location    ┆ 0.194114 ┆ false         │
        │ plane      ┆ time_of_day ┆ 0.033996 ┆ false         │
        │ plane      ┆ weather     ┆ 0.0      ┆ false         │
        └────────────┴─────────────┴──────────┴───────────────┘
        """
        # Convert AnnotatedDataset to Metadata if needed
        if is_metadata_like(data):
            self.metadata = data
        else:
            self.metadata = Metadata(data)

        if not self.metadata.factor_names:
            raise ValueError("No factors found in provided metadata.")

        # The axis is whatever is being conditioned on: the class labels by default, or
        # the named factor(s). A factor serving as the axis is dropped from the factors
        # analysed against it, since it would otherwise report perfect correlation with
        # itself.
        axis = resolve_label_axis(self.metadata, self.label)
        factor_data, factor_names, _ = factors_excluding(self.metadata, axis.excluded)

        # Every column of factor_data holds discrete integer bins: MetadataLike requires a
        # continuous factor to be binned before it reaches here, so what arrives is bin
        # indices either way. The factor's *original* continuous/discrete nature is
        # provenance, not a description of these values, and passing it as though it were
        # would give a binned factor an infinite entropy. Its class-to-factor score would
        # then be normalized by the class entropy rather than its own, holding it below
        # H(factor)/H(class) no matter how strongly the class determines it.
        binned = [True] * len(factor_names)

        mi = mutual_info(axis.values, factor_data, binned, self.num_neighbors)

        # Calculate classwise balance
        classwise = mutual_info_classwise(axis.values, factor_data, binned, self.num_neighbors)

        index2label = axis.names

        # Create classwise DataFrame - build as columnar data
        # classwise is (num_classes, num_factors+1) where column 0 is the label axis
        # measured against itself. That self-information is 1.0 for every class by
        # construction, so it is dropped: it says nothing about the class and would
        # trip the imbalance threshold for every row.
        class_name_col: list[str] = []
        factor_name_col: list[str] = []
        mi_value_col: list[float] = []
        is_imbalanced_col: list[bool] = []

        u_classes = np.unique(axis.values)
        for class_idx in range(classwise.shape[0]):
            class_name = index2label.get(int(u_classes[class_idx]), str(u_classes[class_idx]))
            for factor_idx, factor_name in enumerate(factor_names, start=1):
                mi_value = classwise[class_idx, factor_idx]
                class_name_col.append(class_name)
                factor_name_col.append(factor_name)
                mi_value_col.append(float(mi_value))
                is_imbalanced_col.append(bool(mi_value > self.class_imbalance_threshold))

        classwise_df = pl.DataFrame(
            {
                "class_name": class_name_col,
                "factor_name": factor_name_col,
                "mi_value": mi_value_col,
                "is_imbalanced": is_imbalanced_col,
            },
            schema={
                "class_name": pl.Categorical("lexical"),
                "factor_name": pl.Categorical("lexical"),
                "mi_value": pl.Float64,
                "is_imbalanced": pl.Boolean,
            },
        ).sort(["class_name", "factor_name"], descending=[False, False])

        # Create factors DataFrame for inter-factor correlations - build as columnar data
        # mi["interfactor"] is symmetric matrix of metadata factors (excluding class_label)
        interfactor_matrix = mi["interfactor"]
        num_metadata_factors = interfactor_matrix.shape[0]

        factor1_col: list[str] = []
        factor2_col: list[str] = []
        mi_value_col_factors: list[float] = []
        is_correlated_col: list[bool] = []

        for i in range(num_metadata_factors):
            for j in range(num_metadata_factors):
                # skip diagonal
                if i == j:
                    continue
                mi_value = interfactor_matrix[i, j]
                factor1_col.append(factor_names[i])
                factor2_col.append(factor_names[j])
                mi_value_col_factors.append(float(mi_value))
                is_correlated_col.append(bool(mi_value > self.factor_correlation_threshold))

        factors_df = pl.DataFrame(
            {
                "factor1": factor1_col,
                "factor2": factor2_col,
                "mi_value": mi_value_col_factors,
                "is_correlated": is_correlated_col,
            },
            schema={
                "factor1": pl.Categorical("lexical"),
                "factor2": pl.Categorical("lexical"),
                "mi_value": pl.Float64,
                "is_correlated": pl.Boolean,
            },
        ).sort(["factor1", "factor2"])

        # Create balance DataFrame for global class-to-factor MI
        # mi["class_to_factor"] has shape (num_factors+1,) where index 0 is class_label's self-MI
        # Include all values: class_label + metadata factors
        class_to_factor = mi["class_to_factor"]
        sorted_factor_names = sorted(factor_names)
        all_factor_names = [axis.label] + sorted_factor_names
        # Map sorted factor names to their original indices in class_to_factor
        mi_values = [float(class_to_factor[0])] + [
            float(class_to_factor[factor_names.index(fn) + 1]) for fn in sorted_factor_names
        ]
        balance_df = pl.DataFrame(
            {
                "factor_name": all_factor_names,
                "mi_value": mi_values,
            },
            schema={
                "factor_name": pl.Categorical("lexical"),
                "mi_value": pl.Float64,
            },
        )

        return BalanceOutput(balance=balance_df, factors=factors_df, classwise=classwise_df)
