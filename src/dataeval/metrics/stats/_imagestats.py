from __future__ import annotations

__all__ = []

from typing import Any

from dataeval.core._processor import process
from dataeval.core.processors._pixelstats import PixelStatsPerChannelProcessor, PixelStatsProcessor
from dataeval.core.processors._visualstats import VisualStatsPerChannelProcessor, VisualStatsProcessor
from dataeval.metrics.stats._base import convert_output, unzip_dataset
from dataeval.outputs import ImageStatsOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import ArrayLike, Dataset


@set_metadata
def imagestats(
    dataset: Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]],
    *,
    per_box: bool = False,
    per_channel: bool = False,
) -> ImageStatsOutput:
    """
    Calculates various :term:`statistics<Statistics>` for each image.

    This function computes pixel and visual metrics on the images or
    individual bounding boxes for each image.

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
    ImageStatsOutput
        Output class containing the outputs of various stats functions

    Examples
    --------
    Calculate dimension, pixel and visual statistics for a dataset containing 8
    images.

    >>> stats = imagestats(dataset)
    >>> print(stats.mean)
    [0.181 0.132 0.248 0.373 0.464 0.613 0.734 0.854]

    >>> print(stats.sharpness)
    [20.23 20.23 23.33 20.23 77.06 20.23 20.23 20.23]

    Calculate the pixel and visual stats for a dataset containing 6 3-channel
    images and 2 1-channel images for a total of 20 channels.

    >>> ch_stats = imagestats(dataset, per_channel=True)
    >>> print(ch_stats.brightness)
    [0.027 0.152 0.277 0.127 0.135 0.142 0.259 0.377 0.385 0.392 0.508 0.626
     0.634 0.642 0.751 0.759 0.767 0.876 0.884 0.892]
    """
    processors = (
        [PixelStatsPerChannelProcessor, VisualStatsPerChannelProcessor]
        if per_channel
        else [PixelStatsProcessor, VisualStatsProcessor]
    )
    stats = process(*unzip_dataset(dataset, per_box), processors=processors)
    return convert_output(ImageStatsOutput, stats)
