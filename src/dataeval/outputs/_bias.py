from __future__ import annotations

__all__ = []

import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

with contextlib.suppress(ImportError):
    from matplotlib.figure import Figure

from dataeval.data._images import Images
from dataeval.outputs._base import Output
from dataeval.typing import ArrayLike, Dataset
from dataeval.utils._array import as_numpy, channels_first_to_last
from dataeval.utils._plot import heatmap

TData = TypeVar("TData", np.float64, NDArray[np.float64])


class ToDataFrameMixin:
    score: Any
    p_value: Any

    def to_dataframe(self) -> pd.DataFrame:
        """
        Exports the parity output results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame

        Notes
        -----
        This method requires `pandas <https://pandas.pydata.org/>`_ to be installed.
        """
        return pd.DataFrame(
            index=self.factor_names,  # type: ignore - Sequence[str] is documented as acceptable index type
            data={
                "score": self.score.round(2),
                "p-value": self.p_value.round(2),
            },
        )


@dataclass(frozen=True)
class ParityOutput(ToDataFrameMixin, Output):
    """
    Output class for :func:`.parity` :term:`bias<Bias>` metrics.

    Attributes
    ----------
    score : NDArray[np.float64]
        chi-squared score(s) of the test
    p_value : NDArray[np.float64]
        p-value(s) of the test
    factor_names : Sequence[str]
        Names of each metadata factor
    insufficient_data: dict
        Dictionary of metadata factors with less than 5 class occurrences per value
    """

    score: NDArray[np.float64]
    p_value: NDArray[np.float64]
    factor_names: Sequence[str]
    insufficient_data: Mapping[str, Mapping[int, Mapping[str, int]]]


@dataclass(frozen=True)
class LabelParityOutput(ToDataFrameMixin, Output):
    """
    Output class for :func:`.label_parity` :term:`bias<Bias>` metrics.

    Attributes
    ----------
    score : np.float64
        chi-squared score(s) of the test
    p_value : np.float64
        p-value(s) of the test
    """

    score: np.float64
    p_value: np.float64


@dataclass(frozen=True)
class CoverageOutput(Output):
    """
    Output class for :func:`.coverage` :term:`bias<Bias>` metric.

    Attributes
    ----------
    uncovered_indices : NDArray[np.intp]
        Array of uncovered indices
    critical_value_radii : NDArray[np.float64]
        Array of critical value radii
    coverage_radius : float
        Radius for :term:`coverage<Coverage>`
    """

    uncovered_indices: NDArray[np.intp]
    critical_value_radii: NDArray[np.float64]
    coverage_radius: float

    def plot(self, images: Images[Any] | Dataset[Any], top_k: int = 6) -> Figure:
        """
        Plot the top k images together for visualization.

        Parameters
        ----------
        images : Images or Dataset
            Original images (not embeddings) in (N, C, H, W) or (N, H, W) format
        top_k : int, default 6
            Number of images to plot (plotting assumes groups of 3)

        Returns
        -------
        matplotlib.figure.Figure

        Notes
        -----
        This method requires `matplotlib <https://matplotlib.org/>`_ to be installed.
        """

        import matplotlib.pyplot as plt

        images = Images(images) if isinstance(images, Dataset) else images
        if np.max(self.uncovered_indices) > len(images):
            raise ValueError(
                f"Uncovered indices {self.uncovered_indices} specify images "
                f"unavailable in the provided number of images {len(images)}."
            )

        # Determine which images to plot
        selected_indices = self.uncovered_indices[:top_k]

        # Plot the images
        num_images = min(top_k, len(selected_indices))

        rows = int(np.ceil(num_images / 3))
        cols = min(3, num_images)
        fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

        # Flatten axes using numpy array explicitly for compatibility
        axs_flat = np.asarray(axs).flatten()

        for image, ax in zip(images[:num_images], axs_flat):
            image = channels_first_to_last(as_numpy(image))
            ax.imshow(image)
            ax.axis("off")

        for ax in axs_flat[num_images:]:
            ax.axis("off")

        fig.tight_layout()
        return fig


@dataclass(frozen=True)
class CompletenessOutput(Output):
    """
    Output from the completeness function.

    Attributes
    ----------
    fraction_filled : float
        Fraction of boxes that contain at least one data point
    empty_box_centers : List[np.ndarray]
        List of coordinates for centers of empty boxes
    """

    fraction_filled: float
    empty_box_centers: NDArray[np.float64]


@dataclass(frozen=True)
class BalanceOutput(Output):
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


@dataclass(frozen=True)
class DiversityOutput(Output):
    """
    Output class for :func:`.diversity` :term:`bias<Bias>` metric.

    Attributes
    ----------
    diversity_index : NDArray[np.double]
        :term:`Diversity` index for classes and factors
    classwise : NDArray[np.double]
        Classwise diversity index [n_class x n_factor]
    factor_names : Sequence[str]
        Names of each metadata factor
    class_names : Sequence[str]
        Class labels for each value in the dataset
    """

    diversity_index: NDArray[np.double]
    classwise: NDArray[np.double]
    factor_names: Sequence[str]
    class_names: Sequence[str]

    def plot(
        self,
        row_labels: ArrayLike | None = None,
        col_labels: ArrayLike | None = None,
        plot_classwise: bool = False,
    ) -> Figure:
        """
        Plot a heatmap of diversity information.

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
                cbarlabel=f"Normalized {asdict(self.meta())['arguments']['method'].title()} Index",
            )

        else:
            # Creating label array for heat map axes
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 8))
            heat_labels = ["class_labels"] + list(self.factor_names)
            ax.bar(heat_labels, self.diversity_index)
            ax.set_xlabel("Factors")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            fig.tight_layout()

        return fig
