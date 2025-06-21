from __future__ import annotations

__all__ = []

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.stats import entropy, kurtosis, skew

from dataeval.metrics.stats._base import StatsProcessor, run_stats
from dataeval.outputs import PixelStatsOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import ArrayLike, Dataset


class PixelStatsProcessor(StatsProcessor[PixelStatsOutput]):
    output_class: type = PixelStatsOutput
    cache_keys = {"histogram"}
    image_function_map: dict[str, Callable[[StatsProcessor[PixelStatsOutput]], Any]] = {
        "mean": lambda x: np.nanmean(x.scaled),
        "std": lambda x: np.nanstd(x.scaled),
        "var": lambda x: np.nanvar(x.scaled),
        "skew": lambda x: skew(x.scaled.ravel(), nan_policy="omit"),
        "kurtosis": lambda x: kurtosis(x.scaled.ravel(), nan_policy="omit"),
        "histogram": lambda x: np.histogram(x.scaled, 256, (0, 1))[0],
        "entropy": lambda x: entropy(x.get("histogram")),
    }
    channel_function_map: dict[str, Callable[[StatsProcessor[PixelStatsOutput]], Any]] = {
        "mean": lambda x: np.nanmean(x.scaled, axis=1),
        "std": lambda x: np.nanstd(x.scaled, axis=1),
        "var": lambda x: np.nanvar(x.scaled, axis=1),
        "skew": lambda x: skew(x.scaled, axis=1, nan_policy="omit"),
        "kurtosis": lambda x: kurtosis(x.scaled, axis=1, nan_policy="omit"),
        "histogram": lambda x: np.apply_along_axis(lambda y: np.histogram(y, 256, (0, 1))[0], 1, x.scaled),
        "entropy": lambda x: entropy(x.get("histogram"), axis=1),
    }


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

    Examples
    --------
    Calculate the pixel statistics of a dataset of 8 images, whose shape is (C, H, W).

    >>> results = pixelstats(dataset)
    >>> print(results.mean)
    [0.181 0.132 0.248 0.373 0.464 0.613 0.734 0.854]
    >>> print(results.entropy)
    [4.527 1.883 0.811 1.883 0.298 1.883 1.883 1.883]
    """
    return run_stats(dataset, per_box, per_channel, [PixelStatsProcessor])[0]
