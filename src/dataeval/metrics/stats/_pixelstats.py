from __future__ import annotations

__all__ = []

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy, kurtosis, skew

from dataeval._output import set_metadata
from dataeval.metrics.stats._base import BaseStatsOutput, HistogramPlotMixin, StatsProcessor, run_stats
from dataeval.typing import ArrayLike


@dataclass(frozen=True)
class PixelStatsOutput(BaseStatsOutput, HistogramPlotMixin):
    """
    Output class for :func:`.pixelstats` stats metric.

    Attributes
    ----------
    mean : NDArray[np.float16]
        Mean of the pixel values of the images
    std : NDArray[np.float16]
        Standard deviation of the pixel values of the images
    var : NDArray[np.float16]
        :term:`Variance` of the pixel values of the images
    skew : NDArray[np.float16]
        Skew of the pixel values of the images
    kurtosis : NDArray[np.float16]
        Kurtosis of the pixel values of the images
    histogram : NDArray[np.uint32]
        Histogram of the pixel values of the images across 256 bins scaled between 0 and 1
    entropy : NDArray[np.float16]
        Entropy of the pixel values of the images
    """

    mean: NDArray[np.float16]
    std: NDArray[np.float16]
    var: NDArray[np.float16]
    skew: NDArray[np.float16]
    kurtosis: NDArray[np.float16]
    histogram: NDArray[np.uint32]
    entropy: NDArray[np.float16]

    _excluded_keys = ["histogram"]


class PixelStatsProcessor(StatsProcessor[PixelStatsOutput]):
    output_class: type = PixelStatsOutput
    image_function_map: dict[str, Callable[[StatsProcessor[PixelStatsOutput]], Any]] = {
        "mean": lambda x: np.mean(x.scaled),
        "std": lambda x: np.std(x.scaled),
        "var": lambda x: np.var(x.scaled),
        "skew": lambda x: np.nan_to_num(skew(x.scaled.ravel())),
        "kurtosis": lambda x: np.nan_to_num(kurtosis(x.scaled.ravel())),
        "histogram": lambda x: np.histogram(x.scaled, 256, (0, 1))[0],
        "entropy": lambda x: entropy(x.get("histogram")),
    }
    channel_function_map: dict[str, Callable[[StatsProcessor[PixelStatsOutput]], Any]] = {
        "mean": lambda x: np.mean(x.scaled, axis=1),
        "std": lambda x: np.std(x.scaled, axis=1),
        "var": lambda x: np.var(x.scaled, axis=1),
        "skew": lambda x: np.nan_to_num(skew(x.scaled, axis=1)),
        "kurtosis": lambda x: np.nan_to_num(kurtosis(x.scaled, axis=1)),
        "histogram": lambda x: np.apply_along_axis(lambda y: np.histogram(y, 256, (0, 1))[0], 1, x.scaled),
        "entropy": lambda x: entropy(x.get("histogram"), axis=1),
    }


@set_metadata
def pixelstats(
    images: Iterable[ArrayLike],
    bboxes: Iterable[ArrayLike] | None = None,
    per_channel: bool = False,
) -> PixelStatsOutput:
    """
    Calculates pixel :term:`statistics<Statistics>` for each image.

    This function computes various statistical metrics (e.g., mean, standard deviation, entropy)
    on the images as a whole.

    Parameters
    ----------
    images : Iterable[ArrayLike]
        Images to perform calculations on
    bboxes : Iterable[ArrayLike] or None
        Bounding boxes in `xyxy` format for each image to perform calculations

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
    Calculating the statistics on the images, whose shape is (C, H, W)

    >>> results = pixelstats(stats_images)
    >>> print(results.mean)
    [0.2903 0.2108 0.397  0.596  0.743 ]
    >>> print(results.entropy)
    [4.99  2.371 1.179 2.406 0.668]
    """
    return run_stats(images, bboxes, per_channel, [PixelStatsProcessor])[0]
