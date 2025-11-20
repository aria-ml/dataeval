from __future__ import annotations

__all__ = []

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from dataeval.core._bin import get_counts
from dataeval.core._diversity import diversity_shannon, diversity_simpson
from dataeval.data import Metadata
from dataeval.protocols import ArrayLike
from dataeval.types import DictOutput, set_metadata
from dataeval.utils._method import get_method
from dataeval.utils._plot import heatmap

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_DIVERSITY_FN_MAP = {"simpson": diversity_simpson, "shannon": diversity_shannon}


@dataclass(frozen=True)
class DiversityOutput(DictOutput):
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


@set_metadata
def diversity(
    metadata: Metadata,
    method: Literal["simpson", "shannon"] = "simpson",
) -> DiversityOutput:
    """
    Compute :term:`diversity<Diversity>` and classwise diversity for \
        discrete/categorical variables through standard histogram binning, \
        for continuous variables.

    The method specified defines diversity as the inverse Simpson diversity index linearly rescaled to
    the unit interval, or the normalized form of the Shannon entropy.

    diversity = 1 implies that samples are evenly distributed across a particular factor
    diversity = 0 implies that all samples belong to one category/bin

    Parameters
    ----------
    metadata : Metadata
        Preprocessed metadata
    method : "simpson" or "shannon", default "simpson"
        The methodology used for defining diversity

    Returns
    -------
    DiversityOutput
        Diversity index per column of self.data or each factor in self.names and
        classwise diversity [n_class x n_factor]

    Notes
    -----
    - The expression is undefined for q=1, but it approaches the Shannon entropy in the limit.
    - If there is only one category, the diversity index takes a value of 0.

    Example
    -------
    Compute the diversity index of metadata and class labels

    >>> metadata = generate_random_metadata(
    ...     labels=["doctor", "artist", "teacher"],
    ...     factors={
    ...         "age": [25, 30, 35, 45],
    ...         "income": [50000, 65000, 80000],
    ...         "gender": ["M", "F"]},
    ...     length=100,
    ...     random_seed=175)

    >>> div_simp = diversity(metadata, method="simpson")
    >>> div_simp.diversity_index
    array([0.938, 0.944, 0.888, 0.987])

    >>> div_simp.classwise
    array([[0.964, 0.858, 0.973],
           [0.747, 0.727, 0.997],
           [0.829, 0.915, 0.965]])

    Compute Shannon diversity index of metadata and class labels

    >>> div_shan = diversity(metadata, method="shannon")
    >>> div_shan.diversity_index
    array([0.981, 0.983, 0.962, 0.995])

    >>> div_shan.classwise
    array([[0.99 , 0.948, 0.99 ],
           [0.921, 0.878, 0.999],
           [0.939, 0.972, 0.987]])

    See Also
    --------
    scipy.stats.entropy
    """
    if not metadata.factor_names:
        raise ValueError("No factors found in provided metadata.")

    diversity_fn = get_method(_DIVERSITY_FN_MAP, method)
    binned_data = metadata.binned_data
    factor_names = metadata.factor_names
    class_lbl = metadata.class_labels

    class_labels_with_binned_data = np.hstack((class_lbl[:, np.newaxis], binned_data))
    cnts = get_counts(class_labels_with_binned_data)
    num_bins = np.bincount(np.nonzero(cnts)[1])
    diversity_index = diversity_fn(cnts, num_bins)

    u_classes = np.unique(class_lbl)
    num_factors = len(factor_names)
    classwise_div = np.full((len(u_classes), num_factors), np.nan)
    for idx, cls in enumerate(u_classes):
        subset_mask = class_lbl == cls
        cls_cnts = get_counts(binned_data[subset_mask], min_num_bins=cnts.shape[0])
        classwise_div[idx, :] = diversity_fn(cls_cnts, num_bins[1:])

    return DiversityOutput(diversity_index, classwise_div, factor_names, metadata.class_names)
