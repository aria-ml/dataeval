from __future__ import annotations

__all__ = []

import contextlib
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import entropy as sp_entropy

from dataeval.interop import to_numpy

with contextlib.suppress(ImportError):
    from matplotlib.figure import Figure

CLASS_LABEL = "class_label"


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
) -> tuple[NDArray[Any], list[str], list[bool], NDArray[np.str_]]:
    # if class_labels is not numeric
    class_array = to_numpy(class_labels)
    if not np.issubdtype(class_array.dtype, np.number):
        unique_classes, numerical_labels = np.unique(class_array, return_inverse=True)
    else:
        numerical_labels = np.asarray(class_array, dtype=int)
        unique_classes = np.unique(class_array)

    # convert class_labels and dict of lists to matrix of metadata values
    preprocessed_metadata = {CLASS_LABEL: numerical_labels}

    # map columns of dict that are not numeric (e.g. string) to numeric values
    # that mutual information and diversity functions can accommodate.  Each
    # unique string receives a unique integer value.
    for k, v in metadata.items():
        if k == CLASS_LABEL:
            k = "label_class"
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

    return data, names, is_categorical, unique_classes


def heatmap(
    data: NDArray[Any],
    row_labels: list[str] | NDArray[Any],
    col_labels: list[str] | NDArray[Any],
    xlabel: str = "",
    ylabel: str = "",
    cbarlabel: str = "",
) -> Figure:
    """
    Plots a formatted heatmap

    Parameters
    ----------
    data : NDArray
        Array containing numerical values for factors to plot
    row_labels : ArrayLike
        List/Array containing the labels for rows in the histogram
    col_labels : ArrayLike
        List/Array containing the labels for columns in the histogram
    xlabel : str, default ""
        X-axis label
    ylabel : str, default ""
        Y-axis label
    cbarlabel : str, default ""
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
    return fig


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


def diversity_bar_plot(labels: NDArray[Any], bar_heights: NDArray[Any]) -> Figure:
    """
    Plots a formatted bar plot

    Parameters
    ----------
    labels : NDArray
        Array containing the labels for each bar
    bar_heights : NDArray
        Array containing the values for each bar
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.bar(labels, bar_heights)
    ax.set_xlabel("Factors")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fig.tight_layout()
    return fig


def coverage_plot(images: NDArray[Any], num_images: int) -> Figure:
    """
    Creates a single plot of all of the provided images

    Parameters
    ----------
    images : NDArray
        Array containing only the desired images to plot
    """
    import matplotlib.pyplot as plt

    num_images = min(num_images, len(images))

    if images.ndim == 4:
        images = np.moveaxis(images, 1, -1)
    elif images.ndim == 3:
        images = np.repeat(images[:, :, :, np.newaxis], 3, axis=-1)
    else:
        raise ValueError(
            f"Expected a (N,C,H,W) or a (N, H, W) set of images, but got a {images.ndim}-dimensional set of images."
        )

    rows = np.ceil(num_images / 3).astype(int)
    fig, axs = plt.subplots(rows, 3, figsize=(9, 3 * rows))

    if rows == 1:
        for j in range(3):
            if j >= len(images):
                continue
            axs[j].imshow(images[j])
            axs[j].axis("off")
    else:
        for i in range(rows):
            for j in range(3):
                i_j = i * 3 + j
                if i_j >= len(images):
                    continue
                axs[i, j].imshow(images[i_j])
                axs[i, j].axis("off")

    fig.tight_layout()
    return fig
