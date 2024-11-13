from __future__ import annotations

__all__ = []

from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import entropy as sp_entropy

from dataeval.interop import to_numpy


def get_counts(
    data: NDArray[Any],
    names: list[str],
    continuous_factor_bincounts: Mapping[str, int],
    subset_mask: NDArray[np.bool_] | None = None,
    cached_hist: Mapping[str, Mapping[str, ArrayLike]] | None = None,
) -> tuple[Mapping[str, ArrayLike], Mapping[str, ArrayLike], Mapping[str, Mapping[str, ArrayLike]]]:
    """
    Initialize dictionary of histogram counts --- treat categorical values
    as histogram bins.

    Parameters
    ----------
    data: NDArray
        Array containing numerical values for metadata factors
    names: list[str]
        Names of metadata factors -- keys of the metadata dictionary
    continuous_factor_bincounts: Dict[str, int], default None
        The factors in names that have continuous values and the array of bin counts to
        discretize values into. All factors are treated as having discrete values unless they
        are specified as keys in this dictionary. Each element of this array must occur as a key
        in names.
        Names of metadata factors -- keys of the metadata dictionary
    subset_mask: NDArray[np.bool_] | None
        Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts
    cached_hist: Dict[str, Dict[str, ArrayLike]] | None, default None
        If specified, provides the bin information that will be used to discretize continuous factors.
        Keys are metadata factors, values are {"cnts": counts, "bins": bins}


    Returns
    -------
    hist_counts: Dict
        histogram counts per metadata factor in `factors`.  Each
        factor will have a different number of bins.  Counts get reused
        across metrics, so hist_counts are cached but only if computed
        globally, i.e. without masked samples.
    hist_bins: NDArray
        Bin boundaries used for discretization
    cached_hist: Dict[str, Dict[str, ArrayLike]]
        Cached bin information for discretizing continuous factors.
        Keys are metadata factors, values are {"cnts": counts, "bins": bins}
    """

    hist_counts, hist_bins = {}, {}
    # np.where needed to satisfy linter
    # TODO: The commented line results in discretization according to nonglobal standards,
    # need to figure out an elegant solution to this problem
    mask = np.where(subset_mask if subset_mask is not None else np.ones(data.shape[0], dtype=bool))
    # mask = np.where(np.ones(data.shape[0], dtype=bool))
    if not cached_hist:
        cached_hist = {}

    for cdx, fn in enumerate(names):
        # linter doesn't like double indexing
        col_data = data[mask, cdx].squeeze()
        if continuous_factor_bincounts and fn in continuous_factor_bincounts:
            # bins = hist_bins.get(fn, "auto")
            num_bins = continuous_factor_bincounts[fn]
            if fn in cached_hist:
                cnts, bins = cached_hist[fn]["cnts"], cached_hist[fn]["bins"]
            else:
                cnts, bins = np.histogram(data[:, cdx].squeeze(), bins=num_bins, density=True)
                cached_hist[fn] = {"cnts": cnts, "bins": bins}
                bins[-1] = np.inf
                bins[0] = -np.inf
            disc_col_data = np.digitize(col_data, bins)
            bins, cnts = np.unique(disc_col_data, return_counts=True)
        else:
            # if discrete, use unique values as bins
            if fn in cached_hist:
                cnts, bins = cached_hist[fn]["cnts"], cached_hist[fn]["bins"]
            else:
                bins, cnts = np.unique(col_data, return_counts=True)
                cached_hist[fn] = {"cnts": cnts, "bins": bins}

        hist_counts[fn] = cnts
        hist_bins[fn] = bins

    return hist_counts, hist_bins, cached_hist


def entropy(
    data: NDArray[Any],
    names: list[str],
    continuous_factor_bincounts: Mapping[str, int],
    normalized: bool = False,
    subset_mask: NDArray[np.bool_] | None = None,
    cached_hist: Mapping[str, Mapping[str, ArrayLike]] | None = None,
) -> Mapping[str, Mapping[str, ArrayLike]]:
    """
    Meant for use with Bias metrics, Balance, Diversity, ClasswiseBalance,
    and Classwise Diversity.

    Compute entropy for discrete/categorical variables and for continuous variables through standard
    histogram binning.

    Parameters
    ----------
    data: NDArray
        Array containing numerical values for metadata factors
    names: list[str]
        Names of metadata factors -- keys of the metadata dictionary
    continuous_factor_bincounts: Dict[str, int], default None
        The factors in names that have continuous values and the array of bin counts to
        discretize values into. All factors are treated as having discrete values unless they
        are specified as keys in this dictionary. Each element of this array must occur as a key
        in names.
    normalized: bool
        Flag that determines whether or not to normalize entropy by log(num_bins)
    subset_mask: NDArray[np.bool_] | None
        Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts
    cached_hist: Dict[str, Dict[str, ArrayLike]] | None, default None
        If specified, provides the bin information that will be used to discretize continuous factors.
        Keys are metadata factors, values are {"cnts": counts, "bins": bins}

    Notes
    -----
    For continuous variables, histogram bins are chosen automatically.  See
    numpy.histogram for details.

    Returns
    -------
    ent: NDArray[np.float64]
        Entropy estimate per column of X
    cached_hist: Dict[str, Dict[str, ArrayLike]]
        Cached bin information for discretizing continuous factors.
        Keys are metadata factors, values are {"cnts": counts, "bins": bins}

    See Also
    --------
    numpy.histogram
    scipy.stats.entropy
    """

    num_factors = len(names)
    hist_counts, _, cached_hist = get_counts(data, names, continuous_factor_bincounts, subset_mask, cached_hist)

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
    return ev_index, cached_hist


def get_num_bins(
    data: NDArray[Any],
    names: list[str],
    continuous_factor_bincounts: Mapping[str, int],
    subset_mask: NDArray[np.bool_] | None = None,
    cached_hist: Mapping[str, Mapping[str, ArrayLike]] | None = None,
) -> tuple[NDArray[np.float64], Mapping[str, Mapping[str, ArrayLike]]]:
    """
    Number of bins or unique values for each metadata factor, used to
    normalize entropy/diversity.

    Parameters
    ----------
    data: NDArray
        Array containing numerical values for metadata factors
    names: list[str]
        Names of metadata factors -- keys of the metadata dictionary
    continuous_factor_bincounts: Dict[str, int], default None
        The factors in names that have continuous values and the array of bin counts to
        discretize values into. All factors are treated as having discrete values unless they
        are specified as keys in this dictionary. Each element of this array must occur as a key
        in names.
    subset_mask: NDArray[np.bool_] | None
        Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts
    cached_hist: Dict[str, Dict[str, ArrayLike]] | None, default None
        If specified, provides the bin information that will be used to discretize continuous factors.
        Keys are metadata factors, values are {"cnts": counts, "bins": bins}

    Returns
    -------
    num_bins: NDArray[np.float64]
        Number of bins used in the discretization for each value in names.
    cached_hist: Dict[str, Dict[str, ArrayLike]]
        Cached bin information for discretizing continuous factors.
        Keys are metadata factors, values are {"cnts": counts, "bins": bins}
    """
    # likely cached
    hist_counts, _, cached_hist = get_counts(data, names, continuous_factor_bincounts, subset_mask, cached_hist)
    num_bins = np.empty(len(hist_counts))
    for idx, cnts in enumerate(hist_counts.values()):
        num_bins[idx] = len(cnts)

    return num_bins, cached_hist


def infer_categorical(X: NDArray[np.float64], threshold: float = 0.2) -> NDArray[np.bool_]:
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


def heatmap(
    data: NDArray[Any],
    row_labels: NDArray[Any],
    col_labels: NDArray[Any],
    xlabel: str = "",
    ylabel: str = "",
    cbarlabel: str = "",
) -> None:
    """
    Plots a formatted heatmap

    Parameters
    ----------
    data: NDArray
        Array containing numerical values for factors to plot
    row_labels: NDArray
        Array containing the labels for rows in the histogram
    col_labels: NDArray
        Array containing the labels for columns in the histogram
    xlabel: str, default ""
        X-axis label
    ylabel: str, default ""
        Y-axis label
    cbarlabel: str, default ""
        Label for the colorbar

    """
    import matplotlib
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the heatmap
    im = ax.imshow(data, vmin=0, vmax=1.0)

    # Create colorbar
    cbar = fig.colorbar(im, shrink=0.5)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"])
    cbar.set_label(cbarlabel, loc="center")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    valfmt = matplotlib.ticker.FuncFormatter(format_text)  # type: ignore

    # Normalize the threshold to the images color range.
    threshold = im.norm(1.0) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = {"horizontalalignment": "center", "verticalalignment": "center"}

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    textcolors = ("white", "black")
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)  # type: ignore
            texts.append(text)

    fig.tight_layout()
    plt.show()


# Function to define how the text is displayed in the heatmap
def format_text(*args: str) -> str:
    """
    Helper function to format text for heatmap()

    Parameters
    ----------
    *args: Tuple (str, str)
        Text to be formatted. Second element is ignored, but is a
        mandatory pass-through argument as per matplotlib.ticket.FuncFormatter

    Returns
    -------
    str
        Formatted text
    """
    x = args[0]
    return f"{x:.2f}".replace("0.00", "0").replace("0.", ".").replace("nan", "")
