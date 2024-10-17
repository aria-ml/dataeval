from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval._internal.metrics.utils import entropy, get_counts, get_method, get_num_bins, preprocess_metadata
from dataeval._internal.output import OutputMetadata, set_metadata


@dataclass(frozen=True)
class DiversityOutput(OutputMetadata):
    """
    Output class for :func:`diversity` bias metric

    Attributes
    ----------
    diversity_index : NDArray[np.float64]
        Diversity index for classes and factors
    classwise : NDArray[np.float64]
        Classwise diversity index [n_class x n_factor]
    """

    diversity_index: NDArray[np.float64]
    classwise: NDArray[np.float64]


def diversity_shannon(
    data: NDArray,
    names: list[str],
    is_categorical: list[bool],
    subset_mask: NDArray[np.bool_] | None = None,
) -> NDArray:
    """
    Compute diversity for discrete/categorical variables and, through standard
    histogram binning, for continuous variables.

    We define diversity as a normalized form of the Shannon entropy.

    diversity = 1 implies that samples are evenly distributed across a particular factor
    diversity = 0 implies that all samples belong to one category/bin

    Parameters
    ----------
    data: NDArray
        Array containing numerical values for metadata factors
    names: list[str]
        Names of metadata factors -- keys of the metadata dictionary
    is_categorical: list[bool]
        List of flags to identify whether variables are categorical (True) or
        continuous (False)
    subset_mask: NDArray[np.bool_] | None
        Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts

    Note
    ----
    For continuous variables, histogram bins are chosen automatically.  See `numpy.histogram` for details.

    Returns
    -------
    diversity_index: NDArray
        Diversity index per column of X

    See Also
    --------
    numpy.histogram
    """

    # entropy computed using global auto bins so that we can properly normalize
    ent_unnormalized = entropy(data, names, is_categorical, normalized=False, subset_mask=subset_mask)
    # normalize by global counts rather than classwise counts
    num_bins = get_num_bins(data, names, is_categorical=is_categorical, subset_mask=subset_mask)
    ent_norm = np.empty(ent_unnormalized.shape)
    ent_norm[num_bins != 1] = ent_unnormalized[num_bins != 1] / np.log(num_bins[num_bins != 1])
    ent_norm[num_bins == 1] = 0
    return ent_norm


def diversity_simpson(
    data: NDArray,
    names: list[str],
    is_categorical: list[bool],
    subset_mask: NDArray[np.bool_] | None = None,
) -> NDArray:
    """
    Compute diversity for discrete/categorical variables and, through standard
    histogram binning, for continuous variables.

    We define diversity as the inverse Simpson diversity index linearly rescaled to the unit interval.

    diversity = 1 implies that samples are evenly distributed across a particular factor
    diversity = 0 implies that all samples belong to one category/bin

    Parameters
    ----------
    data: NDArray
        Array containing numerical values for metadata factors
    names: list[str]
        Names of metadata factors -- keys of the metadata dictionary
    is_categorical: list[bool]
        List of flags to identify whether variables are categorical (True) or
        continuous (False)
    subset_mask: NDArray[np.bool_] | None
        Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts

    Note
    ----
    For continuous variables, histogram bins are chosen automatically.  See
        numpy.histogram for details.
    If there is only one category, the diversity index takes a value of 0.

    Returns
    -------
    NDArray
        Diversity index per column of X

    See Also
    --------
    numpy.histogram
    """

    hist_counts, _ = get_counts(data, names, is_categorical, subset_mask)
    # normalize by global counts, not classwise counts
    num_bins = get_num_bins(data, names, is_categorical)

    ev_index = np.empty(len(names))
    # loop over columns for convenience
    for col, cnts in enumerate(hist_counts.values()):
        # relative frequencies
        p_i = cnts / cnts.sum()
        # inverse Simpson index normalized by (number of bins)
        s_0 = 1 / np.sum(p_i**2) / num_bins[col]
        if num_bins[col] == 1:
            ev_index[col] = 0
        else:
            ev_index[col] = (s_0 * num_bins[col] - 1) / (num_bins[col] - 1)
    return ev_index


DIVERSITY_FN_MAP = {"simpson": diversity_simpson, "shannon": diversity_shannon}


@set_metadata("dataeval.metrics")
def diversity(
    class_labels: ArrayLike, metadata: Mapping[str, ArrayLike], method: Literal["shannon", "simpson"] = "simpson"
) -> DiversityOutput:
    """
    Compute diversity and classwise diversity for discrete/categorical variables and, through standard
    histogram binning, for continuous variables.

    We define diversity as a normalized form of the inverse Simpson diversity index.

    diversity = 1 implies that samples are evenly distributed across a particular factor
    diversity = 0 implies that all samples belong to one category/bin

    Parameters
    ----------
    class_labels: ArrayLike
        List of class labels for each image
    metadata: Mapping[str, ArrayLike]
        Dict of list of metadata factors for each image
    method: Literal["shannon", "simpson"], default "simpson"
        Indicates which diversity index should be computed

    Note
    ----
    - For continuous variables, histogram bins are chosen automatically. See numpy.histogram for details.
    - The expression is undefined for q=1, but it approaches the Shannon entropy in the limit.
    - If there is only one category, the diversity index takes a value of 1 = 1/N = 1/1. Entropy will take a value of 0.

    Returns
    -------
    DiversityOutput
        Diversity index per column of self.data or each factor in self.names and
        classwise diversity [n_class x n_factor]

    Example
    -------
    Compute Simpson diversity index of metadata and class labels

    >>> div_simp = diversity(class_labels, metadata, method="simpson")
    >>> div_simp.diversity_index
    array([0.18103448, 0.18103448, 0.88636364])

    >>> div_simp.classwise
    array([[0.17241379, 0.39473684],
           [0.2       , 0.2       ]])

    Compute Shannon diversity index of metadata and class labels

    >>> div_shan = diversity(class_labels, metadata, method="shannon")
    >>> div_shan.diversity_index
    array([0.37955133, 0.37955133, 0.96748876])

    >>> div_shan.classwise
    array([[0.43156028, 0.83224889],
           [0.57938016, 0.57938016]])

    See Also
    --------
    numpy.histogram
    """
    diversity_fn = get_method(DIVERSITY_FN_MAP, method)
    data, names, is_categorical = preprocess_metadata(class_labels, metadata)
    diversity_index = diversity_fn(data, names, is_categorical, None).astype(np.float64)

    class_idx = names.index("class_label")
    class_lbl = data[:, class_idx]

    u_classes = np.unique(class_lbl)
    num_factors = len(names)
    diversity = np.empty((len(u_classes), num_factors))
    diversity[:] = np.nan
    for idx, cls in enumerate(u_classes):
        subset_mask = class_lbl == cls
        diversity[idx, :] = diversity_fn(data, names, is_categorical, subset_mask)
    div_no_class = np.concatenate((diversity[:, :class_idx], diversity[:, (class_idx + 1) :]), axis=1)

    return DiversityOutput(diversity_index, div_no_class)
