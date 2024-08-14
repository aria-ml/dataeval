from typing import Dict, List

import numpy as np
from scipy.stats import entropy


def _get_counts(
    data: np.ndarray, names: list[str], is_categorical: List, subset_mask: np.ndarray = np.empty(shape=0)
) -> tuple[Dict, Dict]:
    """
    Initialize dictionary of histogram counts --- treat categorical values
    as histogram bins.

    Parameters
    ----------
    subset_mask: Optional[np.ndarray[bool]]
        Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts

    Returns
    -------
    counts: Dict
        histogram counts per metadata factor in `factors`.  Each
        factor will have a different number of bins.  Counts get reused
        across metrics, so hist_counts are cached but only if computed
        globally, i.e. without masked samples.
    """

    hist_counts, hist_bins = {}, {}
    # np.where needed to satisfy linter
    mask = np.where(subset_mask if len(subset_mask) > 0 else np.ones(data.shape[0], dtype=bool))

    for cdx, fn in enumerate(names):
        # linter doesn't like double indexing
        col_data = data[mask, cdx].squeeze()
        if is_categorical[cdx]:
            # if discrete, use unique values as bins
            bins, cnts = np.unique(col_data, return_counts=True)
        else:
            bins = hist_bins.get(fn, "auto")
            cnts, bins = np.histogram(col_data, bins=bins, density=True)

        hist_counts[fn] = cnts
        hist_bins[fn] = bins

    return hist_counts, hist_bins


def _entropy(
    data: np.ndarray,
    names: list,
    is_categorical: List,
    normalized: bool = False,
    subset_mask: np.ndarray = np.empty(shape=0),
) -> np.ndarray:
    """
    Meant for use with Bias metrics, Balance, Diversity, ClasswiseBalance,
    and Classwise Diversity.

    Compute entropy for discrete/categorical variables and, through standard
    histogram binning, for continuous variables.


    Parameters
    ----------
    normalized: bool
        Flag that determines whether or not to normalize entropy by log(num_bins)
    subset_mask: Optional[np.ndarray[bool]]
        Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts


    Notes
    -----
    For continuous variables, histogram bins are chosen automatically.  See
    numpy.histogram for details.

    Returns
    -------
    ent: np.ndarray[float]
        Entropy estimate per column of X

    See Also
    --------
    numpy.histogram
    scipy.stats.entropy
    """

    num_factors = len(names)
    hist_counts, _ = _get_counts(data, names, is_categorical, subset_mask=subset_mask)

    ev_index = np.empty(num_factors)
    for col, cnts in enumerate(hist_counts.values()):
        # entropy in nats, normalizes counts
        ev_index[col] = entropy(cnts)
        if normalized:
            if len(cnts) == 1:
                # log(0)
                ev_index[col] = 0
            else:
                ev_index[col] /= np.log(len(cnts))
    return ev_index


def _get_num_bins(
    data: np.ndarray, names: list, is_categorical: List, subset_mask: np.ndarray = np.empty(shape=0)
) -> np.ndarray:
    """
    Number of bins or unique values for each metadata factor, used to
    normalize entropy/diversity.

    Parameters
    ----------
    subset_mask: Optional[np.ndarray[bool]]
        Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts
    """
    # likely cached
    hist_counts, _ = _get_counts(data, names, is_categorical, subset_mask)
    num_bins = np.empty(len(hist_counts))
    for idx, cnts in enumerate(hist_counts.values()):
        num_bins[idx] = len(cnts)

    return num_bins


def _infer_categorical(X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Compute fraction of feature values that are unique --- intended to be used
    for inferring whether variables are categorical.
    """
    if X.ndim == 1:
        X = np.expand_dims(X, axis=1)
    num_samples = X.shape[0]
    pct_unique = np.empty(X.shape[1])
    for col in range(X.shape[1]):  # type: ignore
        uvals = np.unique(X[:, col], axis=0)
        pct_unique[col] = len(uvals) / num_samples
    return pct_unique < threshold
