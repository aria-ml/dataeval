from __future__ import annotations

__all__ = []

from typing import Any

from dataeval.core._processor import process
from dataeval.core.processors._pixelstats import PixelStatsPerChannelProcessor, PixelStatsProcessor
from dataeval.metrics.stats._base import convert_output, unzip_dataset
from dataeval.outputs import PixelStatsOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import ArrayLike, Dataset


@set_metadata
def pixelstats(
    dataset: Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]],
    *,
    per_box: bool = False,
    per_channel: bool = False,
) -> PixelStatsOutput:
    """
    Calculates pixel :term:`statistics<Statistics>` for each image.

    This function computes various statistical metrics (e.g., mean, standard deviation, entropy)
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
    PixelStatsOutput
        A dictionary-like object containing the computed statistics for each image. The keys correspond
        to the names of the statistics (e.g., 'mean', 'std'), and the values are lists of results for
        each image or :term:`NumPy` arrays when the results are multi-dimensional.

    See Also
    --------
    dimensionstats, visualstats, Outliers

    Note
    ----
    - All metrics are scaled based on the perceived bit depth (which is derived from the largest pixel value)
      to allow for better comparison between images stored in different formats and different resolutions.
    - `zeros` and `missing` are presented as a percentage of total pixel counts

    Examples
    --------
    Calculate the pixel statistics of a dataset of 8 images, whose shape is (C, H, W).

    >>> results = pixelstats(dataset)
    >>> print(results.mean)
    [0.181 0.132 0.248 0.373 0.464 0.613 0.734 0.854]
    >>> print(results.entropy)
    [4.527 1.883 0.811 1.883 0.298 1.883 1.883 1.883]
    """
    processor = PixelStatsPerChannelProcessor if per_channel else PixelStatsProcessor
    stats = process(*unzip_dataset(dataset, per_box), processor)
    return convert_output(PixelStatsOutput, stats)
