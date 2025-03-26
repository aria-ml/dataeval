from __future__ import annotations

__all__ = []

import contextlib
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeVar, overload

import numpy as np
from numpy.typing import NDArray

with contextlib.suppress(ImportError):
    import pandas as pd
    from matplotlib.figure import Figure

from dataeval.outputs._base import Output
from dataeval.typing import ArrayLike
from dataeval.utils._array import to_numpy
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
        import pandas as pd

        return pd.DataFrame(
            index=self.factor_names,  # type: ignore - list[str] is documented as acceptable index type
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
    factor_names : list[str]
        Names of each metadata factor
    insufficient_data: dict
        Dictionary of metadata factors with less than 5 class occurrences per value
    """

    score: NDArray[np.float64]
    p_value: NDArray[np.float64]
    factor_names: list[str]
    insufficient_data: dict[str, dict[int, dict[str, int]]]


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

    def plot(self, images: ArrayLike, top_k: int = 6) -> Figure:
        """
        Plot the top k images together for visualization.

        Parameters
        ----------
        images : ArrayLike
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

        # Determine which images to plot
        highest_uncovered_indices = self.uncovered_indices[:top_k]

        # Grab the images
        selected_images = to_numpy(images)[highest_uncovered_indices]

        # Plot the images
        num_images = min(top_k, len(images))

        ndim = selected_images.ndim
        if ndim == 4:
            selected_images = np.moveaxis(selected_images, 1, -1)
        elif ndim == 3:
            selected_images = np.repeat(selected_images[:, :, :, np.newaxis], 3, axis=-1)
        else:
            raise ValueError(
                f"Expected a (N,C,H,W) or a (N, H, W) set of images, but got a {ndim}-dimensional set of images."
            )

        rows = int(np.ceil(num_images / 3))
        fig, axs = plt.subplots(rows, 3, figsize=(9, 3 * rows))

        if rows == 1:
            for j in range(3):
                if j >= len(selected_images):
                    continue
                axs[j].imshow(selected_images[j])
                axs[j].axis("off")
        else:
            for i in range(rows):
                for j in range(3):
                    i_j = i * 3 + j
                    if i_j >= len(selected_images):
                        continue
                    axs[i, j].imshow(selected_images[i_j])
                    axs[i, j].axis("off")

        fig.tight_layout()
        return fig


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
    factor_names : list[str]
        Names of each metadata factor
    class_names : list[str]
        List of the class labels present in the dataset
    """

    balance: NDArray[np.float64]
    factors: NDArray[np.float64]
    classwise: NDArray[np.float64]
    factor_names: list[str]
    class_names: list[str]

    @overload
    def _by_factor_type(
        self,
        attr: Literal["factor_names"],
        factor_type: Literal["discrete", "continuous", "both"],
    ) -> list[str]: ...

    @overload
    def _by_factor_type(
        self,
        attr: Literal["balance", "factors", "classwise"],
        factor_type: Literal["discrete", "continuous", "both"],
    ) -> NDArray[np.float64]: ...

    def _by_factor_type(
        self,
        attr: Literal["balance", "factors", "classwise", "factor_names"],
        factor_type: Literal["discrete", "continuous", "both"],
    ) -> NDArray[np.float64] | list[str]:
        # if not filtering by factor_type then just return the requested attribute without mask
        if factor_type == "both":
            return getattr(self, attr)

        # create the mask for the selected factor_type
        mask_lambda = (
            (lambda x: "-continuous" not in x) if factor_type == "discrete" else (lambda x: "-discrete" not in x)
        )

        # return the masked attribute
        if attr == "factor_names":
            return [x.replace(f"-{factor_type}", "") for x in self.factor_names if mask_lambda(x)]
        else:
            factor_type_mask = np.asarray([mask_lambda(x) for x in self.factor_names])
            if attr == "factors":
                return self.factors[factor_type_mask[1:]][:, factor_type_mask[1:]]
            elif attr == "balance":
                return self.balance[factor_type_mask]
            elif attr == "classwise":
                return self.classwise[:, factor_type_mask]

    def plot(
        self,
        row_labels: list[Any] | NDArray[Any] | None = None,
        col_labels: list[Any] | NDArray[Any] | None = None,
        plot_classwise: bool = False,
        factor_type: Literal["discrete", "continuous", "both"] = "discrete",
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
        factor_type : "discrete", "continuous", or "both", default "discrete"
            Whether to plot discretized values, continuous values, or to include both

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
                col_labels = self._by_factor_type("factor_names", factor_type)

            fig = heatmap(
                self._by_factor_type("classwise", factor_type),
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
                    self._by_factor_type("balance", factor_type)[np.newaxis, 1:],
                    self._by_factor_type("factors", factor_type),
                ],
                axis=0,
            )
            # Create a mask for the upper triangle of the symmetrical array, ignoring the diagonal
            mask = np.triu(data + 1, k=0) < 1
            # Finalize the data for the plot, last row is last factor x last factor so it gets dropped
            heat_data = np.where(mask, np.nan, data)[:-1]
            # Creating label array for heat map axes
            heat_labels = self._by_factor_type("factor_names", factor_type)

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
    factor_names : list[str]
        Names of each metadata factor
    class_names : list[str]
        Class labels for each value in the dataset
    """

    diversity_index: NDArray[np.double]
    classwise: NDArray[np.double]
    factor_names: list[str]
    class_names: list[str]

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
            heat_labels = np.concatenate((["class"], self.factor_names))
            ax.bar(heat_labels, self.diversity_index)
            ax.set_xlabel("Factors")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            fig.tight_layout()

        return fig
