from __future__ import annotations

__all__ = ["DiversityOutput", "diversity"]

import contextlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import scipy as sp
from numpy.typing import ArrayLike, NDArray

from dataeval.metrics.bias.metadata_preprocessing import MetadataOutput
from dataeval.metrics.bias.metadata_utils import diversity_bar_plot, get_counts, heatmap
from dataeval.output import Output, set_metadata
from dataeval.utils.shared import get_method

with contextlib.suppress(ImportError):
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class DiversityOutput(Output):
    """
    Output class for :func:`diversity` :term:`bias<Bias>` metric

    Attributes
    ----------
    diversity_index : NDArray[np.double]
        :term:`Diversity` index for classes and factors
    classwise : NDArray[np.double]
        Classwise diversity index [n_class x n_factor]
    factor_names : list[str]
        Names of each metadata factor
    class_list : NDArray[Any]
        Class labels for each value in the dataset
    """

    diversity_index: NDArray[np.double]
    classwise: NDArray[np.double]
    factor_names: list[str]
    class_list: NDArray[Any]

    def plot(
        self,
        row_labels: ArrayLike | None = None,
        col_labels: ArrayLike | None = None,
        plot_classwise: bool = False,
    ) -> Figure:
        """
        Plot a heatmap of diversity information

        Parameters
        ----------
        row_labels : ArrayLike or None, default None
            List/Array containing the labels for rows in the histogram
        col_labels : ArrayLike or None, default None
            List/Array containing the labels for columns in the histogram
        plot_classwise : bool, default False
            Whether to plot per-class balance instead of global balance
        """
        if plot_classwise:
            if row_labels is None:
                row_labels = self.class_list
            if col_labels is None:
                col_labels = self.factor_names

            fig = heatmap(
                self.classwise,
                row_labels,
                col_labels,
                xlabel="Factors",
                ylabel="Class",
                cbarlabel=f"Normalized {self.meta()['arguments']['method'].title()} Index",
            )

        else:
            # Creating label array for heat map axes
            heat_labels = np.concatenate((["class"], self.factor_names))

            fig = diversity_bar_plot(heat_labels, self.diversity_index)

        return fig


def diversity_shannon(
    counts: NDArray[np.int_],
    num_bins: NDArray[np.int_],
) -> NDArray[np.double]:
    """
    Compute :term:`diversity<Diversity>` for discrete/categorical variables and, through standard
    histogram binning, for continuous variables.

    We define diversity as a normalized form of the Shannon entropy.

    diversity = 1 implies that samples are evenly distributed across a particular factor
    diversity = 0 implies that all samples belong to one category/bin

    Parameters
    ----------
    counts : NDArray[np.int_]
        Array containing bin counts for each factor
    num_bins : NDArray[np.int_]
        Number of bins with values for each factor

    Returns
    -------
    diversity_index : NDArray[np.double]
        Diversity index per column of X

    See Also
    --------
    scipy.stats.entropy
    """
    raw_entropy = sp.stats.entropy(counts, axis=0)
    ent_norm = np.empty(raw_entropy.shape)
    ent_norm[num_bins != 1] = raw_entropy[num_bins != 1] / np.log(num_bins[num_bins != 1])
    ent_norm[num_bins == 1] = 0
    return ent_norm


def diversity_simpson(
    counts: NDArray[np.int_],
    num_bins: NDArray[np.int_],
) -> NDArray[np.double]:
    """
    Compute :term:`diversity<Diversity>` for discrete/categorical variables and, through standard
    histogram binning, for continuous variables.

    We define diversity as the inverse Simpson diversity index linearly rescaled to the unit interval.

    diversity = 1 implies that samples are evenly distributed across a particular factor
    diversity = 0 implies that all samples belong to one category/bin

    Parameters
    ----------
    counts : NDArray[np.int_]
        Array containing bin counts for each factor
    num_bins : NDArray[np.int_]
        Number of bins with values for each factor

    Note
    ----
    If there is only one category, the diversity index takes a value of 0.

    Returns
    -------
    diversity_index : NDArray[np.double]
        Diversity index per column of X
    """
    ev_index = np.empty(counts.shape[1])
    # loop over columns for convenience
    for col, cnts in enumerate(counts.T):
        # relative frequencies
        p_i = cnts / np.sum(cnts)
        # inverse Simpson index
        s_0 = 1 / np.sum(p_i**2)
        if num_bins[col] == 1:
            ev_index[col] = 0
        else:
            # normalized by number of bins
            ev_index[col] = (s_0 - 1) / (num_bins[col] - 1)
    return ev_index


@set_metadata
def diversity(
    metadata: MetadataOutput,
    method: Literal["simpson", "shannon"] = "simpson",
) -> DiversityOutput:
    """
    Compute :term:`diversity<Diversity>` and classwise diversity for discrete/categorical variables and,
    through standard histogram binning, for continuous variables.

    We define diversity as a normalized form of the inverse Simpson diversity index.

    diversity = 1 implies that samples are evenly distributed across a particular factor
    diversity = 0 implies that all samples belong to one category/bin

    Parameters
    ----------
    metadata : MetadataOutput
        Output after running `metadata_preprocessing`

    Note
    ----
    - The expression is undefined for q=1, but it approaches the Shannon entropy in the limit.
    - If there is only one category, the diversity index takes a value of 0.

    Returns
    -------
    DiversityOutput
        Diversity index per column of self.data or each factor in self.names and
        classwise diversity [n_class x n_factor]

    Example
    -------
    Compute Simpson diversity index of metadata and class labels

    >>> div_simp = diversity(metadata, method="simpson")
    >>> div_simp.diversity_index
    array([0.72413793, 0.88636364, 0.72413793])

    >>> div_simp.classwise
    array([[0.69230769, 0.68965517],
           [0.5       , 0.8       ]])

    Compute Shannon diversity index of metadata and class labels

    >>> div_shan = diversity(metadata, method="shannon")
    >>> div_shan.diversity_index
    array([0.8812909 , 0.96748876, 0.8812909 ])

    >>> div_shan.classwise
    array([[0.91651644, 0.86312057],
           [0.68260619, 0.91829583]])

    See Also
    --------
    scipy.stats.entropy
    """
    diversity_fn = get_method({"simpson": diversity_simpson, "shannon": diversity_shannon}, method)
    discretized_data = np.hstack((metadata.class_labels[:, np.newaxis], metadata.discrete_data))
    cnts = get_counts(discretized_data)
    num_bins = np.bincount(np.nonzero(cnts)[1])
    diversity_index = diversity_fn(cnts, num_bins)

    class_lbl = metadata.class_labels

    u_classes = np.unique(class_lbl)
    num_factors = len(metadata.discrete_factor_names)
    classwise_div = np.full((len(u_classes), num_factors), np.nan)
    for idx, cls in enumerate(u_classes):
        subset_mask = class_lbl == cls
        cls_cnts = get_counts(metadata.discrete_data[subset_mask], min_num_bins=cnts.shape[0])
        classwise_div[idx, :] = diversity_fn(cls_cnts, num_bins[1:])

    return DiversityOutput(diversity_index, classwise_div, metadata.discrete_factor_names, metadata.class_names)
