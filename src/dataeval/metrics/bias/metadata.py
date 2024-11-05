from __future__ import annotations

__all__ = []

from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import entropy as sp_entropy

from dataeval.interop import to_numpy


def get_counts(
    data: NDArray[np.int_], names: list[str], is_categorical: list[bool], subset_mask: NDArray[np.bool_] | None = None
) -> tuple[dict[str, NDArray[np.int_]], dict[str, NDArray[np.int_]]]:
    """
    Initialize dictionary of histogram counts --- treat categorical values
    as histogram bins.

    Parameters
    ----------
    subset_mask: NDArray[np.bool_] | None
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
    mask = np.where(subset_mask if subset_mask is not None else np.ones(data.shape[0], dtype=bool))

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


def entropy(
    data: NDArray[Any],
    names: list[str],
    is_categorical: list[bool],
    normalized: bool = False,
    subset_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """
    Meant for use with :term:`bias<Bias>` metrics, :term:`balance<Balance>`, :term:`diversity<Diversity>`,
    ClasswiseBalance, and Classwise Diversity.

    Compute entropy for discrete/categorical variables and for continuous variables through standard
    histogram binning.

    Parameters
    ----------
    normalized: bool
        Flag that determines whether or not to normalize entropy by log(num_bins)
    subset_mask: NDArray[np.bool_] | None
        Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts

    Note
    ----
    For continuous variables, histogram bins are chosen automatically.  See
    numpy.histogram for details.

    Returns
    -------
    ent: NDArray[np.float64]
        Entropy estimate per column of X

    See Also
    --------
    numpy.histogram
    scipy.stats.entropy
    """

    num_factors = len(names)
    hist_counts, _ = get_counts(data, names, is_categorical, subset_mask)

    ev_index = np.empty(num_factors)
    for col, cnts in enumerate(hist_counts.values()):
        # entropy in nats, normalizes counts
        ev_index[col] = sp_entropy(cnts)
        if normalized:
            if len(cnts) == 1:
                # log(0)
                ev_index[col] = 0
            else:
                ev_index[col] /= np.log(len(cnts))
    return ev_index


def get_num_bins(
    data: NDArray[Any], names: list[str], is_categorical: list[bool], subset_mask: NDArray[np.bool_] | None = None
) -> NDArray[np.float64]:
    """
    Number of bins or unique values for each metadata factor, used to
    normalize entropy/:term:`diversity<Diversity>`.

    Parameters
    ----------
    subset_mask: NDArray[np.bool_] | None
        Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts

    Returns
    -------
    NDArray[np.float64]
    """
    # likely cached
    hist_counts, _ = get_counts(data, names, is_categorical, subset_mask)
    num_bins = np.empty(len(hist_counts))
    for idx, cnts in enumerate(hist_counts.values()):
        num_bins[idx] = len(cnts)

    return num_bins


def infer_categorical(arr: NDArray[Any], threshold: float = 0.2) -> NDArray[Any]:
    """
    Compute fraction of feature values that are unique --- intended to be used
    for inferring whether variables are categorical.
    """
    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=1)
    num_samples = arr.shape[0]
    pct_unique = np.empty(arr.shape[1])
    for col in range(arr.shape[1]):  # type: ignore
        uvals = np.unique(arr[:, col], axis=0)
        pct_unique[col] = len(uvals) / num_samples
    return pct_unique < threshold


def preprocess_metadata(
    class_labels: ArrayLike, metadata: Mapping[str, ArrayLike], cat_thresh: float = 0.2
) -> tuple[NDArray[Any], list[str], list[bool]]:
    # convert class_labels and dict of lists to matrix of metadata values
    preprocessed_metadata = {"class_label": np.asarray(class_labels, dtype=int)}

    # map columns of dict that are not numeric (e.g. string) to numeric values
    # that mutual information and diversity functions can accommodate.  Each
    # unique string receives a unique integer value.
    for k, v in metadata.items():
        # if not numeric
        v = to_numpy(v)
        if not np.issubdtype(v.dtype, np.number):
            _, mapped_vals = np.unique(v, return_inverse=True)
            preprocessed_metadata[k] = mapped_vals
        else:
            preprocessed_metadata[k] = v

    data = np.stack(list(preprocessed_metadata.values()), axis=-1)
    names = list(preprocessed_metadata.keys())
    is_categorical = [infer_categorical(preprocessed_metadata[var], cat_thresh)[0] for var in names]

    return data, names, is_categorical
