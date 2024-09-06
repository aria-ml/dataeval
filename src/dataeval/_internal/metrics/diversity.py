from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from dataeval._internal.metrics.utils import entropy, get_counts, get_method, get_num_bins, preprocess_metadata
from dataeval._internal.output import OutputMetadata, set_metadata


@dataclass(frozen=True)
class DiversityOutput(OutputMetadata):
    """
    Attributes
    ----------
    diversity_index : NDArray[np.float64]
        Diversity index for classes and factors
    """

    diversity_index: NDArray[np.float64]


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
    subset_mask: NDArray[np.bool_] | None
        Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts

    Notes
    -----
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

    We define diversity as a normalized form of the inverse Simpson diversity
    index.

    diversity = 1 implies that samples are evenly distributed across a particular factor
    diversity = 1/num_categories implies that all samples belong to one category/bin

    Parameters
    ----------
    subset_mask: NDArray[np.bool_] | None
        Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts

    Notes
    -----
    For continuous variables, histogram bins are chosen automatically.  See
        numpy.histogram for details.
    The expression is undefined for q=1, but it approaches the Shannon entropy
        in the limit.
    If there is only one category, the diversity index takes a value of 1 =
        1/N = 1/1.  Entropy will take a value of 0.

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
        ev_index[col] = 1 / np.sum(p_i**2) / num_bins[col]

    return ev_index


DIVERSITY_FN_MAP = {"simpson": diversity_simpson, "shannon": diversity_shannon}


@set_metadata("dataeval.metrics")
def diversity(
    class_labels: Sequence[int], metadata: list[dict], method: Literal["shannon", "simpson"] = "simpson"
) -> DiversityOutput:
    """
    Compute diversity for discrete/categorical variables and, through standard
    histogram binning, for continuous variables.

    diversity = 1 implies that samples are evenly distributed across a particular factor
    diversity = 0 implies that all samples belong to one category/bin

    Parameters
    ----------
    class_labels: Sequence[int]
        List of class labels for each image
    metadata: List[Dict]
        List of metadata factors for each image
    metric: Literal["shannon", "simpson"], default "simpson"
        string variable indicating which diversity index should be used.
        Permissible values include "simpson" and "shannon"

    Notes
    -----
    - For continuous variables, histogram bins are chosen automatically. See numpy.histogram for details.

    Returns
    -------
    DiversityOutput
        Diversity index per column of self.data or each factor in self.names

    Example
    -------
    Compute Simpson diversity index of metadata and class labels

    >>> diversity(class_labels, metadata, method="simpson").diversity_index
    array([0.34482759, 0.34482759, 0.90909091])

    Compute Shannon diversity index of metadata and class labels

    >>> diversity(class_labels, metadata, method="shannon").diversity_index
    array([0.37955133, 0.37955133, 0.96748876])


    See Also
    --------
    numpy.histogram
    """
    diversity_fn = get_method(DIVERSITY_FN_MAP, method)
    data, names, is_categorical = preprocess_metadata(class_labels, metadata)
    diversity_index = diversity_fn(data, names, is_categorical, None).astype(np.float64)
    return DiversityOutput(diversity_index)


@set_metadata("dataeval.metrics")
def diversity_classwise(
    class_labels: Sequence[int], metadata: list[dict], method: Literal["shannon", "simpson"] = "simpson"
) -> DiversityOutput:
    """
    Compute diversity for discrete/categorical variables and, through standard
    histogram binning, for continuous variables.

    We define diversity as a normalized form of the inverse Simpson diversity
    index.

    diversity = 1 implies that samples are evenly distributed across a particular factor
    diversity = 1/num_categories implies that all samples belong to one category/bin

    Parameters
    ----------
    class_labels: Sequence[int]
        List of class labels for each image
    metadata: List[Dict]
        List of metadata factors for each image

    Notes
    -----
    - For continuous variables, histogram bins are chosen automatically. See numpy.histogram for details.
    - The expression is undefined for q=1, but it approaches the Shannon entropy in the limit.
    - If there is only one category, the diversity index takes a value of 1 = 1/N = 1/1. Entropy will take a value of 0.

    Returns
    -------
    DiversityOutput
        Diversity index [n_class x n_factor]

    Example
    -------
    Compute classwise Simpson diversity index of metadata and class labels

    >>> diversity_classwise(class_labels, metadata, method="simpson").diversity_index
    array([[0.33793103, 0.51578947],
           [0.36      , 0.36      ]])

    Compute classwise Shannon diversity index of metadata and class labels

    >>> diversity_classwise(class_labels, metadata, method="shannon").diversity_index
    array([[0.43156028, 0.83224889],
           [0.57938016, 0.57938016]])


    See Also
    --------
    numpy.histogram
    """
    diversity_fn = get_method(DIVERSITY_FN_MAP, method)
    data, names, is_categorical = preprocess_metadata(class_labels, metadata)
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
    return DiversityOutput(div_no_class)
