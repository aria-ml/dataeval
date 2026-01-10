__all__ = []

from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

from dataeval import Metadata
from dataeval.core._mutual_info import mutual_info, mutual_info_classwise
from dataeval.protocols import AnnotatedDataset
from dataeval.types import DictOutput, set_metadata


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


class Balance:
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

    See Also
    --------
    sklearn.feature_selection.mutual_info_classif
    sklearn.feature_selection.mutual_info_regression
    sklearn.metrics.mutual_info_score
    """

    def __init__(
        self,
        num_neighbors: int = 5,
        class_imbalance_threshold: float = 0.3,
        factor_correlation_threshold: float = 0.5,
    ) -> None:
        self.metadata: Metadata
        self.num_neighbors = num_neighbors
        self.class_imbalance_threshold = class_imbalance_threshold
        self.factor_correlation_threshold = factor_correlation_threshold

    @set_metadata(state=["num_neighbors", "class_imbalance_threshold", "factor_correlation_threshold"])
    def evaluate(self, data: AnnotatedDataset[Any] | Metadata) -> BalanceOutput:
        """
        Compute mutual information between factors and identify imbalanced classes.

        Parameters
        ----------
        data : AnnotatedDataset[Any] or Metadata
            Either an annotated dataset (which will be converted to Metadata)
            or preprocessed Metadata directly.

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

        >>> metadata = generate_random_metadata(
        ...     labels=["doctor", "artist", "teacher"],
        ...     factors={"age": [25, 30, 35, 45], "income": [50000, 65000, 80000], "gender": ["M", "F"]},
        ...     length=100,
        ...     random_seed=175,
        ... )

        >>> balance = Balance()
        >>> result = balance.evaluate(metadata)
        >>> result.balance
        shape: (4, 2)
        ┌─────────────┬──────────┐
        │ factor_name ┆ mi_value │
        │ ---         ┆ ---      │
        │ cat         ┆ f64      │
        ╞═════════════╪══════════╡
        │ class_label ┆ 0.888187 │
        │ age         ┆ 0.251485 │
        │ gender      ┆ 0.00399  │
        │ income      ┆ 0.362771 │
        └─────────────┴──────────┘

        >>> result.factors
        shape: (6, 4)
        ┌─────────┬─────────┬──────────┬───────────────┐
        │ factor1 ┆ factor2 ┆ mi_value ┆ is_correlated │
        │ ---     ┆ ---     ┆ ---      ┆ ---           │
        │ cat     ┆ cat     ┆ f64      ┆ bool          │
        ╞═════════╪═════════╪══════════╪═══════════════╡
        │ age     ┆ gender  ┆ 0.046483 ┆ false         │
        │ age     ┆ income  ┆ 0.078066 ┆ false         │
        │ gender  ┆ age     ┆ 0.046483 ┆ false         │
        │ gender  ┆ income  ┆ 0.047947 ┆ false         │
        │ income  ┆ age     ┆ 0.078066 ┆ false         │
        │ income  ┆ gender  ┆ 0.047947 ┆ false         │
        └─────────┴─────────┴──────────┴───────────────┘

        >>> result.classwise
        shape: (9, 4)
        ┌────────────┬─────────────┬──────────┬───────────────┐
        │ class_name ┆ factor_name ┆ mi_value ┆ is_imbalanced │
        │ ---        ┆ ---         ┆ ---      ┆ ---           │
        │ cat        ┆ cat         ┆ f64      ┆ bool          │
        ╞════════════╪═════════════╪══════════╪═══════════════╡
        │ artist     ┆ age         ┆ 0.301469 ┆ true          │
        │ artist     ┆ gender      ┆ 0.04493  ┆ false         │
        │ artist     ┆ income      ┆ 0.250237 ┆ false         │
        │ doctor     ┆ age         ┆ 0.164287 ┆ false         │
        │ doctor     ┆ gender      ┆ 0.095962 ┆ false         │
        │ doctor     ┆ income      ┆ 0.46587  ┆ true          │
        │ teacher    ┆ age         ┆ 0.137221 ┆ false         │
        │ teacher    ┆ gender      ┆ 0.018392 ┆ false         │
        │ teacher    ┆ income      ┆ 0.160404 ┆ false         │
        └────────────┴─────────────┴──────────┴───────────────┘
        """
        # Convert AnnotatedDataset to Metadata if needed
        if isinstance(data, Metadata):
            self.metadata = data
        else:
            self.metadata = Metadata(data)

        if not self.metadata.factor_names:
            raise ValueError("No factors found in provided metadata.")

        factor_types = {k: v.factor_type for k, v in self.metadata.factor_info.items()}
        is_discrete = [factor_type != "continuous" for factor_type in factor_types.values()]

        mi = mutual_info(
            self.metadata.class_labels,
            self.metadata.binned_data,
            is_discrete,
            self.num_neighbors,
        )

        # Calculate classwise balance
        classwise = mutual_info_classwise(
            self.metadata.class_labels,
            self.metadata.binned_data,
            is_discrete,
            self.num_neighbors,
        )

        # Grabbing factor names for plotting function
        factor_names = list(self.metadata.factor_names)

        # Create classwise DataFrame - build as columnar data
        # classwise is (num_classes, num_factors+1) where column 0 is class_label itself
        class_name_col: list[str] = []
        factor_name_col: list[str] = []
        mi_value_col: list[float] = []
        is_imbalanced_col: list[bool] = []

        for class_idx in range(classwise.shape[0]):
            class_name = (
                self.metadata.index2label[class_idx] if class_idx in self.metadata.index2label else str(class_idx)
            )
            # Skip the first column (class_label's own MI with the binary class indicator)
            for factor_idx in range(1, classwise.shape[1]):
                mi_value = classwise[class_idx, factor_idx]
                class_name_col.append(class_name)
                factor_name_col.append(factor_names[factor_idx - 1])
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
