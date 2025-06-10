from __future__ import annotations

__all__ = []

from typing import Literal

import numpy as np
import scipy as sp
from numpy.typing import NDArray

from dataeval.data import Metadata
from dataeval.outputs import DiversityOutput
from dataeval.outputs._base import set_metadata
from dataeval.utils._bin import get_counts
from dataeval.utils._method import get_method


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


_DIVERSITY_FN_MAP = {"simpson": diversity_simpson, "shannon": diversity_shannon}


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

    Note
    ----
    - The expression is undefined for q=1, but it approaches the Shannon entropy in the limit.
    - If there is only one category, the diversity index takes a value of 0.

    Example
    -------
    Compute the diversity index of metadata and class labels

    >>> div_simp = diversity(metadata, method="simpson")
    >>> div_simp.diversity_index
    array([0.6  , 0.8  , 0.809, 1.   ])

    >>> div_simp.classwise
    array([[0.8  , 0.5  , 0.8  ],
           [0.528, 0.63 , 0.976]])

    Compute Shannon diversity index of metadata and class labels

    >>> div_shan = diversity(metadata, method="shannon")
    >>> div_shan.diversity_index
    array([0.811, 0.918, 0.943, 1.   ])

    >>> div_shan.classwise
    array([[0.918, 0.683, 0.918],
           [0.764, 0.814, 0.991]])

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
