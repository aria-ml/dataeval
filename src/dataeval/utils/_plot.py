from __future__ import annotations

__all__ = []

import contextlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dataeval.typing import ArrayLike
from dataeval.utils._array import to_numpy

with contextlib.suppress(ImportError):
    from matplotlib.figure import Figure


def heatmap(
    data: ArrayLike,
    row_labels: list[str] | ArrayLike,
    col_labels: list[str] | ArrayLike,
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

    Returns
    -------
    matplotlib.figure.Figure
        Formatted heatmap
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    np_data = to_numpy(data)
    rows: list[str] = [str(n) for n in to_numpy(row_labels)]
    cols: list[str] = [str(n) for n in to_numpy(col_labels)]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the heatmap
    im = ax.imshow(np_data, vmin=0, vmax=1.0)

    # Create colorbar
    cbar = fig.colorbar(im, shrink=0.5)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"])
    cbar.set_label(cbarlabel, loc="center")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(np_data.shape[1]), labels=cols)
    ax.set_yticks(np.arange(np_data.shape[0]), labels=rows)

    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    light_gray = "0.9"
    # Turn spines on and create light gray easily visible grid.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(light_gray)

    xticks = np.arange(np_data.shape[1] + 1) - 0.5
    yticks = np.arange(np_data.shape[0] + 1) - 0.5
    ax.set_xticks(xticks, minor=True)
    ax.set_yticks(yticks, minor=True)
    ax.grid(which="minor", color=light_gray, linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    valfmt = FuncFormatter(format_text)

    # Normalize the threshold to the images color range.
    threshold = im.norm(1.0) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = {"horizontalalignment": "center", "verticalalignment": "center"}

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    textcolors = ("white", "black")
    texts = []
    for i in range(np_data.shape[0]):
        for j in range(np_data.shape[1]):
            kw.update(color=textcolors[int(im.norm(np_data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(np_data[i, j], None), **kw)  # type: ignore
            texts.append(text)

    fig.tight_layout()
    return fig


# Function to define how the text is displayed in the heatmap
def format_text(*args: str) -> str:
    """
    Helper function to format text for heatmap()

    Parameters
    ----------
    *args : tuple[str, str]
        Text to be formatted. Second element is ignored, but is a
        mandatory pass-through argument as per matplotlib.ticket.FuncFormatter

    Returns
    -------
    str
        Formatted text
    """
    x = args[0]
    return f"{x:.2f}".replace("0.00", "0").replace("0.", ".").replace("nan", "")


def histogram_plot(
    data_dict: Mapping[str, Any],
    log: bool = True,
    xlabel: str = "values",
    ylabel: str = "counts",
) -> Figure:
    """
    Plots a formatted histogram

    Parameters
    ----------
    data_dict : dict
        Dictionary containing the metrics and their value arrays
    log : bool, default True
        If True, plots the histogram on a semi-log scale (y axis)
    xlabel : str, default "values"
        X-axis label
    ylabel : str, default "counts"
        Y-axis label

    Returns
    -------
    matplotlib.figure.Figure
        Formatted plot of histograms
    """
    import matplotlib.pyplot as plt

    num_metrics = len(data_dict)
    rows = math.ceil(num_metrics / 3)
    cols = min(num_metrics, 3)
    fig, axs = plt.subplots(rows, 3, figsize=(cols * 3 + 1, rows * 3))
    axs_flat = np.asarray(axs).flatten()
    for ax, metric in zip(
        axs_flat,
        data_dict,
    ):
        # Plot the histogram for the chosen metric
        ax.hist(data_dict[metric].astype(np.float64), bins=20, log=log)

        # Add labels to the histogram
        ax.set_title(metric)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

    for ax in axs_flat[num_metrics:]:
        ax.axis("off")
        ax.set_visible(False)

    fig.tight_layout()
    return fig


def channel_histogram_plot(
    data_dict: Mapping[str, Any],
    log: bool = True,
    max_channels: int = 3,
    ch_mask: Sequence[bool] | None = None,
    xlabel: str = "values",
    ylabel: str = "counts",
) -> Figure:
    """
    Plots a formatted heatmap

    Parameters
    ----------
    data_dict : dict
        Dictionary containing the metrics and their value arrays
    log : bool, default True
        If True, plots the histogram on a semi-log scale (y axis)
    xlabel : str, default "values"
        X-axis label
    ylabel : str, default "counts"
        Y-axis label

    Returns
    -------
    matplotlib.figure.Figure
        Formatted plot of histograms
    """
    import matplotlib.pyplot as plt

    channelwise_metrics = ["mean", "std", "var", "skew", "zeros", "brightness", "contrast", "darkness", "entropy"]
    data_keys = [key for key in data_dict if key in channelwise_metrics]
    label_kwargs = {"label": [f"Channel {i}" for i in range(max_channels)]}

    num_metrics = len(data_keys)
    rows = math.ceil(num_metrics / 3)
    cols = min(num_metrics, 3)
    fig, axs = plt.subplots(rows, 3, figsize=(cols * 3 + 1, rows * 3))
    axs_flat = np.asarray(axs).flatten()
    for ax, metric in zip(
        axs_flat,
        data_keys,
    ):
        # Plot the histogram for the chosen metric
        data = data_dict[metric][ch_mask].reshape(-1, max_channels)
        ax.hist(
            data.astype(np.float64),
            bins=20,
            density=True,
            log=log,
            **label_kwargs,
        )
        # Only plot the labels once for channels
        if label_kwargs:
            ax.legend()
            label_kwargs = {}

        # Add labels to the histogram
        ax.set_title(metric)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

    for ax in axs_flat[num_metrics:]:
        ax.axis("off")
        ax.set_visible(False)

    fig.tight_layout()
    return fig
