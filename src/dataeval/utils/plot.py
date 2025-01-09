from __future__ import annotations

__all__ = []

import contextlib

import numpy as np
from numpy.typing import ArrayLike

from dataeval.interop import to_numpy

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
    rows = row_labels if isinstance(row_labels, list) else to_numpy(row_labels)
    cols = col_labels if isinstance(col_labels, list) else to_numpy(col_labels)

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
