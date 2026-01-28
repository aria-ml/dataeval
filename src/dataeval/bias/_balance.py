__all__ = []

from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

from dataeval import Metadata as _Metadata
from dataeval._helpers import _get_index2label
from dataeval.core._mutual_info import mutual_info, mutual_info_classwise
from dataeval.protocols import AnnotatedDataset, Metadata
from dataeval.types import DictOutput, Evaluator, EvaluatorConfig, set_metadata

DEFAULT_BALANCE_NUM_NEIGHBORS = 5
DEFAULT_BALANCE_CLASS_IMBALANCE_THRESHOLD = 0.3
DEFAULT_BALANCE_FACTOR_CORRELATION_THRESHOLD = 0.5


@dataclass(frozen=True)
class BalanceOutput(DictOutput):
    """
    Output class for the :class:`.Balance` :term:`bias<Bias>` evaluator.

    Contains three polars DataFrames with mutual information scores and threshold flags.

    Attributes
    ----------
    balance : pl.DataFrame
        DataFrame with global class-to-factor mutual information:

        - factor_name: str - Name of the metadata factor
        - mi_value: float - Mutual information value between this factor and class labels
    factors : pl.DataFrame
        DataFrame with inter-factor mutual information correlations:

        - factor1: str - Name of the first factor
        - factor2: str - Name of the second factor
        - mi_value: float - Mutual information value
        - is_correlated: bool - True if mi_value > factor_correlation_threshold
    classwise : pl.DataFrame
        DataFrame with per-class-to-factor mutual information:

        - class_name: str - Name of the class
        - factor_name: str - Name of the metadata factor
        - mi_value: float - Mutual information value
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
    Calculates mutual information (MI) between factors (class label, metadata, label/image properties).

    Identifies imbalanced classes and highly correlated metadata factors based on
    mutual information thresholds.

    Parameters
    ----------
    num_neighbors : int, default 5
        Number of points to consider as neighbors
    class_imbalance_threshold : float, default 0.3
        Threshold for identifying imbalanced classes. Classes with MI above this
        threshold with any metadata factor are considered imbalanced.
    factor_correlation_threshold : float, default 0.5
        Threshold for identifying highly correlated metadata factors. Factor pairs
        with MI above this threshold are considered highly correlated.

    Attributes
    ----------
    metadata : Metadata
        Preprocessed metadata from the last evaluate() call.
    num_neighbors : int
        Number of points to consider as neighbors
    class_imbalance_threshold : float
        Threshold for identifying imbalanced classes
    factor_correlation_threshold : float
        Threshold for identifying highly correlated metadata factors

    Notes
    -----
    We use `mutual_info_classif` from sklearn since class label is categorical.
    `mutual_info_classif` outputs are consistent up to O(1e-4) and depend on a random
    seed. MI is computed differently for categorical and continuous variables.

    Examples
    --------
    Initialize the Balance class:

    >>> balance = Balance()

    Specifying custom thresholds:

    >>> balance = Balance(class_imbalance_threshold=0.2, factor_correlation_threshold=0.6)

    Using configuration:

    >>> config = Balance.Config(num_neighbors=10, class_imbalance_threshold=0.2)
    >>> balance = Balance(config=config)

    See Also
    --------
    sklearn.feature_selection.mutual_info_classif
    sklearn.feature_selection.mutual_info_regression
    sklearn.metrics.mutual_info_score
    """

    class Config(EvaluatorConfig):
        """
        Configuration for Balance evaluator.

        Attributes
        ----------
        num_neighbors : int, default 5
            Number of points to consider as neighbors.
        class_imbalance_threshold : float, default 0.3
            Threshold for identifying imbalanced classes.
        factor_correlation_threshold : float, default 0.5
            Threshold for identifying highly correlated metadata factors.
        """

        num_neighbors: int = DEFAULT_BALANCE_NUM_NEIGHBORS
        class_imbalance_threshold: float = DEFAULT_BALANCE_CLASS_IMBALANCE_THRESHOLD
        factor_correlation_threshold: float = DEFAULT_BALANCE_FACTOR_CORRELATION_THRESHOLD

    metadata: Metadata
    num_neighbors: int
    class_imbalance_threshold: float
    factor_correlation_threshold: float
    config: Config

    def __init__(
        self,
        num_neighbors: int | None = None,
        class_imbalance_threshold: float | None = None,
        factor_correlation_threshold: float | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__(locals())

    @set_metadata(state=["num_neighbors", "class_imbalance_threshold", "factor_correlation_threshold"])
    def evaluate(self, data: AnnotatedDataset[Any] | Metadata) -> BalanceOutput:
        """
        Compute mutual information between factors and identify imbalanced classes.

        Parameters
        ----------
        data : AnnotatedDataset[Any] or Metadata
            Either an annotated dataset (which will be converted to Metadata)
            or any object implementing the Metadata protocol.

        Returns
        -------
        BalanceOutput
            Three DataFrames containing MI scores and threshold flags:

            - balance: Global class-to-factor mutual information
            - factors: Inter-factor mutual information
            - classwise: Per-class-to-factor mutual information

        Example
        -------
        Return balance (mutual information) of factors with class_labels

        >>> from dataeval import Metadata
        >>> metadata = Metadata(dataset)

        >>> balance = Balance()
        >>> result = balance.evaluate(metadata)
        >>> result.balance
        shape: (6, 2)
        ┌─────────────┬──────────┐
        │ factor_name ┆ mi_value │
        │ ---         ┆ ---      │
        │ cat         ┆ f64      │
        ╞═════════════╪══════════╡
        │ class_label ┆ 1.0      │
        │ angle       ┆ 0.029047 │
        │ id          ┆ 0.575706 │
        │ location    ┆ 0.024849 │
        │ time_of_day ┆ 0.06278  │
        │ weather     ┆ 0.023614 │
        └─────────────┴──────────┘

        >>> result.factors
        shape: (20, 4)
        ┌─────────────┬─────────────┬──────────┬───────────────┐
        │ factor1     ┆ factor2     ┆ mi_value ┆ is_correlated │
        │ ---         ┆ ---         ┆ ---      ┆ ---           │
        │ cat         ┆ cat         ┆ f64      ┆ bool          │
        ╞═════════════╪═════════════╪══════════╪═══════════════╡
        │ angle       ┆ id          ┆ 1.0      ┆ true          │
        │ angle       ┆ location    ┆ 0.12422  ┆ false         │
        │ angle       ┆ time_of_day ┆ 0.072422 ┆ false         │
        │ angle       ┆ weather     ┆ 0.037279 ┆ false         │
        │ id          ┆ angle       ┆ 1.0      ┆ true          │
        │ …           ┆ …           ┆ …        ┆ …             │
        │ time_of_day ┆ weather     ┆ 0.023866 ┆ false         │
        │ weather     ┆ angle       ┆ 0.037279 ┆ false         │
        │ weather     ┆ id          ┆ 1.0      ┆ true          │
        │ weather     ┆ location    ┆ 0.047246 ┆ false         │
        │ weather     ┆ time_of_day ┆ 0.023866 ┆ false         │
        └─────────────┴─────────────┴──────────┴───────────────┘

        >>> result.classwise
        shape: (24, 4)
        ┌────────────┬─────────────┬──────────┬───────────────┐
        │ class_name ┆ factor_name ┆ mi_value ┆ is_imbalanced │
        │ ---        ┆ ---         ┆ ---      ┆ ---           │
        │ cat        ┆ cat         ┆ f64      ┆ bool          │
        ╞════════════╪═════════════╪══════════╪═══════════════╡
        │ boat       ┆ angle       ┆ 0.020807 ┆ false         │
        │ boat       ┆ class_label ┆ 1.0      ┆ true          │
        │ boat       ┆ id          ┆ 0.471488 ┆ true          │
        │ boat       ┆ location    ┆ 0.009547 ┆ false         │
        │ boat       ┆ time_of_day ┆ 0.04239  ┆ false         │
        │ …          ┆ …           ┆ …        ┆ …             │
        │ plane      ┆ class_label ┆ 1.0      ┆ true          │
        │ plane      ┆ id          ┆ 0.49531  ┆ true          │
        │ plane      ┆ location    ┆ 0.033162 ┆ false         │
        │ plane      ┆ time_of_day ┆ 0.040861 ┆ false         │
        │ plane      ┆ weather     ┆ 0.000407 ┆ false         │
        └────────────┴─────────────┴──────────┴───────────────┘
        """
        # Convert AnnotatedDataset to Metadata if needed
        if isinstance(data, Metadata):
            self.metadata = data
        else:
            self.metadata = _Metadata(data)

        if not self.metadata.factor_names:
            raise ValueError("No factors found in provided metadata.")

        is_discrete = list(self.metadata.is_discrete)

        mi = mutual_info(
            self.metadata.class_labels,
            self.metadata.factor_data,
            is_discrete,
            self.num_neighbors,
        )

        # Calculate classwise balance
        classwise = mutual_info_classwise(
            self.metadata.class_labels,
            self.metadata.factor_data,
            is_discrete,
            self.num_neighbors,
        )

        # Grabbing factor names for plotting function
        factor_names = list(self.metadata.factor_names)
        index2label = _get_index2label(self.metadata)

        # Create classwise DataFrame - build as columnar data
        # classwise is (num_classes, num_factors+1) where column 0 is class_label itself
        class_name_col: list[str] = []
        factor_name_col: list[str] = []
        mi_value_col: list[float] = []
        is_imbalanced_col: list[bool] = []

        # Include class_label as the first factor (index 0), then all metadata factors
        all_factor_names = ["class_label"] + factor_names

        for class_idx in range(classwise.shape[0]):
            class_name = index2label.get(class_idx, str(class_idx))
            for factor_idx in range(classwise.shape[1]):
                mi_value = classwise[class_idx, factor_idx]
                class_name_col.append(class_name)
                factor_name_col.append(all_factor_names[factor_idx])
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
        all_factor_names = ["class_label"] + sorted_factor_names
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
