from __future__ import annotations

__all__ = []

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from dataeval.core._mutual_info import mutual_info, mutual_info_classwise
from dataeval.data import Metadata
from dataeval.types import DictOutput, set_metadata
from dataeval.utils._plot import heatmap

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class BalanceOutput(DictOutput):
    """
    Output class for :func:`.balance` :term:`bias<Bias>` metric.

    Attributes
    ----------
    balance : NDArray[np.float64]
        Estimate of mutual information between metadata factors and class label
    factors : NDArray[np.float64]
        Estimate of inter/intra-factor mutual information
    classwise : NDArray[np.float64]
        Estimate of mutual information between metadata factors and individual class labels
    factor_names : Sequence[str]
        Names of each metadata factor
    class_names : Sequence[str]
        List of the class labels present in the dataset
    """

    balance: NDArray[np.float64]
    factors: NDArray[np.float64]
    classwise: NDArray[np.float64]
    factor_names: Sequence[str]
    class_names: Sequence[str]

    def plot(
        self,
        row_labels: Sequence[Any] | NDArray[Any] | None = None,
        col_labels: Sequence[Any] | NDArray[Any] | None = None,
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
            if row_labels is None:
                row_labels = self.class_names
            if col_labels is None:
                col_labels = self.factor_names

            fig = heatmap(
                self.classwise,
                row_labels,
                col_labels,
                xlabel="Factors",
                ylabel="Class",
                cbarlabel="Normalized Mutual Information",
            )
        else:
            # Combine balance and factors results
            data = np.concatenate(
                [
                    self.balance[np.newaxis, 1:],
                    self.factors,
                ],
                axis=0,
            )
            # Create a mask for the upper triangle of the symmetrical array, ignoring the diagonal
            mask = np.triu(data + 1, k=0) < 1
            # Finalize the data for the plot, last row is last factor x last factor so it gets dropped
            heat_data = np.where(mask, np.nan, data)[:-1]
            # Creating label array for heat map axes
            heat_labels = self.factor_names

            if row_labels is None:
                row_labels = heat_labels[:-1]
            if col_labels is None:
                col_labels = heat_labels[1:]

            fig = heatmap(heat_data, row_labels, col_labels, cbarlabel="Normalized Mutual Information")

        return fig


@set_metadata
def balance(
    metadata: Metadata,
    num_neighbors: int = 5,
) -> BalanceOutput:
    """
    Mutual information (MI) between factors (class label, metadata, label/image properties).

    Parameters
    ----------
    metadata : Metadata
        Preprocessed metadata
    num_neighbors : int, default 5
        Number of points to consider as neighbors

    Returns
    -------
    BalanceOutput
        Contains separate arrays for class-to-factor MI and factor-to-factor MI.

    Notes
    -----
    We use `mutual_info_classif` from sklearn since class label is categorical.
    `mutual_info_classif` outputs are consistent up to O(1e-4) and depend on a random
    seed. MI is computed differently for categorical and continuous variables.

    Example
    -------
    Return balance (mutual information) of factors with class_labels

    >>> metadata = generate_random_metadata(
    ...     labels=["doctor", "artist", "teacher"],
    ...     factors={"age": [25, 30, 35, 45], "income": [50000, 65000, 80000], "gender": ["M", "F"]},
    ...     length=100,
    ...     random_seed=175,
    ... )

    >>> bal = balance(metadata)
    >>> bal.balance
    array([1.017, 0.034, 0.   , 0.028])

    Return intra/interfactor balance (mutual information)

    >>> bal.factors
    array([[1.   , 0.015, 0.038],
           [0.015, 1.   , 0.008],
           [0.038, 0.008, 1.   ]])

    Return classwise balance (mutual information) of factors with individual class_labels

    >>> bal.classwise
    array([[7.818e-01, 1.388e-02, 1.803e-03, 7.282e-04],
           [7.084e-01, 2.934e-02, 1.744e-02, 3.996e-03],
           [7.295e-01, 1.157e-02, 2.799e-02, 9.451e-04]])


    See Also
    --------
    sklearn.feature_selection.mutual_info_classif
    sklearn.feature_selection.mutual_info_regression
    sklearn.metrics.mutual_info_score
    """
    if not metadata.factor_names:
        raise ValueError("No factors found in provided metadata.")

    factor_types = {k: v.factor_type for k, v in metadata.factor_info.items()}
    is_discrete = [factor_type != "continuous" for factor_type in factor_types.values()]

    mi = mutual_info(
        metadata.class_labels,
        metadata.binned_data,
        is_discrete,
        num_neighbors,
    )

    # Calculate classwise balance
    classwise = mutual_info_classwise(
        metadata.class_labels,
        metadata.binned_data,
        is_discrete,
        num_neighbors,
    )

    # Grabbing factor names for plotting function
    factor_names = ["class_label"] + list(metadata.factor_names)

    return BalanceOutput(mi["class_to_factor"], mi["interfactor"], classwise, factor_names, metadata.class_names)
