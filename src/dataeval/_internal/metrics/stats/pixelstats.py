from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import entropy, kurtosis, skew

from dataeval._internal.metrics.stats.base import BaseStatsOutput, StatsProcessor, run_stats
from dataeval._internal.output import set_metadata


class PixelStatsProcessor(StatsProcessor):
    cache_keys = ["histogram"]
    image_function_map = {
        "mean": lambda self: np.mean(self.scaled),
        "std": lambda x: np.std(x.scaled),
        "var": lambda x: np.var(x.scaled),
        "skew": lambda x: np.nan_to_num(skew(x.scaled.ravel())),
        "kurtosis": lambda x: np.nan_to_num(kurtosis(x.scaled.ravel())),
        "histogram": lambda x: np.histogram(x.scaled, 256, (0, 1))[0],
        "entropy": lambda x: entropy(x.get("histogram")),
    }
    channel_function_map = {
        "mean": lambda x: np.mean(x.scaled, axis=1),
        "std": lambda x: np.std(x.scaled, axis=1),
        "var": lambda x: np.var(x.scaled, axis=1),
        "skew": lambda x: np.nan_to_num(skew(x.scaled, axis=1)),
        "kurtosis": lambda x: np.nan_to_num(kurtosis(x.scaled, axis=1)),
        "histogram": lambda x: np.apply_along_axis(lambda y: np.histogram(y, 256, (0, 1))[0], 1, x.scaled),
        "entropy": lambda x: entropy(x.get("histogram"), axis=1),
    }


@dataclass(frozen=True)
class PixelStatsOutput(BaseStatsOutput):
    """
    Output class for :func:`pixelstats` stats metric

    Attributes
    ----------
    mean : NDArray[np.float16]
        Mean of the pixel values of the images
    std : NDArray[np.float16]
        Standard deviation of the pixel values of the images
    var : NDArray[np.float16]
        Variance of the pixel values of the images
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


@set_metadata("dataeval.metrics")
def pixelstats(
    images: Iterable[ArrayLike],
    bboxes: Iterable[ArrayLike] | None = None,
    per_channel: bool = False,
) -> PixelStatsOutput:
    """
    Calculates pixel statistics for each image

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
        each image or numpy arrays when the results are multi-dimensional.

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

    >>> results = pixelstats(images)
    >>> print(results.mean)
    [0.04828 0.562   0.06726 0.09937 0.1315  0.1636  0.1957  0.2278  0.26
     0.292   0.3242  0.3562  0.3884  0.4204  0.4526  0.4846  0.5166  0.549
     0.581   0.6133  0.6455  0.6772  0.7095  0.7417  0.774   0.8057  0.838
     0.87    0.9023  0.934  ]
    >>> print(results.entropy)
    [3.238  3.303  0.8125 1.028  0.8223 1.046  0.8247 1.041  0.8203 1.012
     0.812  0.9883 0.795  0.9243 0.9243 0.795  0.9907 0.8125 1.028  0.8223
     1.046  0.8247 1.041  0.8203 1.012  0.812  0.9883 0.795  0.9243 0.9243]
    """
    output = run_stats(images, bboxes, per_channel, PixelStatsProcessor, PixelStatsOutput)
    return PixelStatsOutput(**output)
