from __future__ import annotations

__all__ = []

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from dataeval.core._mutual_info import mutual_info, mutual_info_classwise
from dataeval.data import Metadata
from dataeval.protocols import AnnotatedDataset, ArrayLike
from dataeval.types import DictOutput, set_metadata
from dataeval.utils._plot import heatmap

if TYPE_CHECKING:
    from matplotlib.figure import Figure


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

    def plot(
        self,
        row_labels: ArrayLike | None = None,
        col_labels: ArrayLike | None = None,
        plot_classwise: bool = False,
    ) -> Figure:
        """
        Plot a heatmap of balance information.

        Parameters
        ----------
        row_labels : ArrayLike or None, default None
            List/Array containing the labels for rows in the histogram
        col_labels : ArrayLike or None, default None
            List/Array containing the labels for columns in the histogram
        plot_classwise : bool, default False
            Whether to plot per-class balance instead of global balance

        Returns
        -------
        matplotlib.figure.Figure

        Notes
        -----
        This method requires `matplotlib <https://matplotlib.org/>`_ to be installed.
        """
        if plot_classwise:
            # Convert classwise DataFrame to numpy array for heatmap
            class_names = self.classwise["class_name"].unique(maintain_order=True).to_list()
            factor_names = self.classwise["factor_name"].unique(maintain_order=True).to_list()

            # Reshape to matrix
            classwise_pivoted = self.classwise.pivot(on="factor_name", index="class_name", values="mi_value")
            # Drop the index column and get values only
            classwise_matrix = classwise_pivoted.select(pl.all().exclude("class_name")).to_numpy()

            if row_labels is None:
                row_labels = class_names
            if col_labels is None:
                col_labels = factor_names

            fig = heatmap(
                classwise_matrix,
                row_labels,
                col_labels,
                xlabel="Factors",
                ylabel="Class",
                cbarlabel="Normalized Mutual Information",
            )
        else:
            # Combine balance (class_to_factor) and factors (interfactor) results
            # This recreates the original visualization from the old implementation

            # Get all factor names from balance DataFrame (includes "class_label" + metadata factors)
            all_factor_names = self.balance["factor_name"].to_list()
            # Metadata factor names only (exclude class_label)
            factor_names = sorted(all_factor_names[1:])

            # Create matrix: first row is balance (class-to-factor MI for metadata factors only),
            # rest is interfactor MI
            balance_row = self.balance["mi_value"].to_numpy()[1:]  # Skip class_label self-MI
            interfactor_matrix = (
                self.factors.pivot(
                    on="factor2",
                    index="factor1",
                    values="mi_value",
                    aggregate_function=None,
                )
                .sort("factor1")  # Sort the rows alphabetically
                .select(factor_names)  # Select columns in the exact same order
                .to_numpy()  # Export to pure NumPy
            )

            # Combine: balance row + interfactor matrix
            data = np.concatenate([balance_row[np.newaxis, :], interfactor_matrix], axis=0)

            # Create mask for lower triangle (excluding diagonal)
            # This creates an upper triangular matrix for visualization
            # Shift diagonal down by 1 to account for the class_label row at the top
            mask = np.tril(np.ones_like(data, dtype=bool), k=-1)
            heat_data = np.where(mask, np.nan, data)[:-1, :]

            if row_labels is None:
                row_labels = ["class_label"] + factor_names[:-1]
            if col_labels is None:
                col_labels = factor_names

            fig = heatmap(heat_data, row_labels, col_labels, cbarlabel="Normalized Mutual Information")

        return fig


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
        │ class_label ┆ 1.01656  │
        │ age         ┆ 0.218666 │
        │ gender      ┆ 0.003119 │
        │ income      ┆ 0.292495 │
        └─────────────┴──────────┘

        >>> result.factors
        shape: (6, 4)
        ┌─────────┬─────────┬──────────┬───────────────┐
        │ factor1 ┆ factor2 ┆ mi_value ┆ is_correlated │
        │ ---     ┆ ---     ┆ ---      ┆ ---           │
        │ cat     ┆ cat     ┆ f64      ┆ bool          │
        ╞═════════╪═════════╪══════════╪═══════════════╡
        │ age     ┆ gender  ┆ 0.031473 ┆ false         │
        │ age     ┆ income  ┆ 0.069446 ┆ false         │
        │ gender  ┆ age     ┆ 0.031473 ┆ false         │
        │ gender  ┆ income  ┆ 0.037382 ┆ false         │
        │ income  ┆ age     ┆ 0.069446 ┆ false         │
        │ income  ┆ gender  ┆ 0.037382 ┆ false         │
        └─────────┴─────────┴──────────┴───────────────┘

        >>> result.classwise
        shape: (9, 4)
        ┌────────────┬─────────────┬──────────┬───────────────┐
        │ class_name ┆ factor_name ┆ mi_value ┆ is_imbalanced │
        │ ---        ┆ ---         ┆ ---      ┆ ---           │
        │ cat        ┆ cat         ┆ f64      ┆ bool          │
        ╞════════════╪═════════════╪══════════╪═══════════════╡
        │ artist     ┆ age         ┆ 0.185507 ┆ false         │
        │ artist     ┆ gender      ┆ 0.036066 ┆ false         │
        │ artist     ┆ income      ┆ 0.172931 ┆ false         │
        │ doctor     ┆ age         ┆ 0.088231 ┆ false         │
        │ doctor     ┆ gender      ┆ 0.073388 ┆ false         │
        │ doctor     ┆ income      ┆ 0.355217 ┆ true          │
        │ teacher    ┆ age         ┆ 0.075241 ┆ false         │
        │ teacher    ┆ gender      ┆ 0.014255 ┆ false         │
        │ teacher    ┆ income      ┆ 0.103269 ┆ false         │
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
