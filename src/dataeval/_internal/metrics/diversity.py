from typing import Dict, List, Literal, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from dataeval._internal.metrics.utils import entropy, get_counts, get_num_bins, preprocess_metadata
from dataeval._internal.utils import get_method


def diversity_shannon(
    data: np.ndarray,
    names: List[str],
    is_categorical: List[bool],
    subset_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute diversity for discrete/categorical variables and, through standard
    histogram binning, for continuous variables.

    We define diversity as a normalized form of the Shannon entropy.

    diversity = 1 implies that samples are evenly distributed across a particular factor
    diversity = 0 implies that all samples belong to one category/bin

    Parameters
    ----------
    subset_mask: Optional[np.ndarray[bool]]
        Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts

    Notes
    -----
    - For continuous variables, histogram bins are chosen automatically.  See
    numpy.histogram for details.

    Returns
    -------
    diversity_index: np.ndarray
        Diversity index per column of X

    See Also
    --------
    numpy.histogram
    """

    # entropy computed using global auto bins so that we can properly normalize
    ent_unnormalized = entropy(data, names, is_categorical, normalized=False, subset_mask=subset_mask)
    # normalize by global counts rather than classwise counts
    num_bins = get_num_bins(data, names, is_categorical=is_categorical, subset_mask=subset_mask)
    return ent_unnormalized / np.log(num_bins)


def diversity_simpson(
    data: np.ndarray,
    names: List[str],
    is_categorical: List[bool],
    subset_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute diversity for discrete/categorical variables and, through standard
    histogram binning, for continuous variables.

    We define diversity as a normalized form of the inverse Simpson diversity
    index.

    diversity = 1 implies that samples are evenly distributed across a particular factor
    diversity = 1/num_categories implies that all samples belong to one category/bin

    Parameters
    ----------
    subset_mask: Optional[np.ndarray[bool]]
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
    np.ndarray
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


def diversity(
    class_labels: Sequence[int], metadata: List[Dict], method: Literal["shannon", "simpson"] = "simpson"
) -> NDArray[np.float64]:
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
    - For continuous variables, histogram bins are chosen automatically.  See
    numpy.histogram for details.

    Returns
    -------
    NDArray[np.float64]
        Diversity index per column of self.data or each factor in self.names

    See Also
    --------
    numpy.histogram

    """
    diversity_fn = get_method(DIVERSITY_FN_MAP, method)
    data, names, is_categorical = preprocess_metadata(class_labels, metadata)
    return diversity_fn(data, names, is_categorical, None).astype(np.float64)


def diversity_classwise(
    class_labels: Sequence[int], metadata: List[Dict], method: Literal["shannon", "simpson"] = "simpson"
) -> NDArray[np.float64]:
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
    For continuous variables, histogram bins are chosen automatically.  See
        numpy.histogram for details.
    The expression is undefined for q=1, but it approaches the Shannon entropy
        in the limit.
    If there is only one category, the diversity index takes a value of 1 =
        1/N = 1/1.  Entropy will take a value of 0.

    Returns
    -------
    NDArray[np.float64]
        Diversity index [n_class x n_factor]

    See Also
    --------
    diversity_simpson
    diversity_shannon
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
    return div_no_class
