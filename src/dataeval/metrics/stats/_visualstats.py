from __future__ import annotations

__all__ = []

from typing import Any

from dataeval.core._processor import process
from dataeval.core.processors._visualstats import VisualStatsPerChannelProcessor, VisualStatsProcessor
from dataeval.metrics.stats._base import convert_output, unzip_dataset
from dataeval.outputs import VisualStatsOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import ArrayLike, Dataset


@set_metadata
def visualstats(
    dataset: Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]],
    *,
    per_box: bool = False,
    per_channel: bool = False,
) -> VisualStatsOutput:
    """
    Calculates visual :term:`statistics` for each image.

    This function computes various visual metrics (e.g., :term:`brightness<Brightness>`, darkness, contrast, blurriness)
    on the images as a whole.

    Parameters
    ----------
    dataset : Dataset
        Dataset to perform calculations on.
    per_box : bool, default False
        If True, perform calculations on each bounding box.
    per_channel : bool, default False
        If True, perform calculations on each channel.

    Returns
    -------
    VisualStatsOutput
        A dictionary-like object containing the computed visual statistics for each image. The keys correspond
        to the names of the statistics (e.g., 'brightness', 'blurriness'), and the values are lists of results for
        each image or :term:`NumPy` arrays when the results are multi-dimensional.

    See Also
    --------
    dimensionstats, pixelstats, Outliers

    Examples
    --------
    Calculate the visual statistics of a dataset of 8 images, whose shape is (C, H, W).

    >>> results = visualstats(dataset)
    >>> print(results.brightness)
    [0.084 0.13  0.259 0.38  0.508 0.63  0.755 0.88 ]
    >>> print(results.contrast)
    [2.04  1.331 1.261 1.279 1.253 1.268 1.265 1.263]
    """
    processor = VisualStatsPerChannelProcessor if per_channel else VisualStatsProcessor
    stats = process(*unzip_dataset(dataset, per_box), processor)
    return convert_output(VisualStatsOutput, stats)
