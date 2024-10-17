from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval._internal.metrics.stats.base import BaseStatsOutput, StatsProcessor, run_stats
from dataeval._internal.metrics.utils import edge_filter
from dataeval._internal.output import set_metadata

QUARTILES = (0, 25, 50, 75, 100)


class VisualStatsProcessor(StatsProcessor):
    cache_keys = ["percentiles"]
    image_function_map = {
        "brightness": lambda x: x.get("percentiles")[-2],
        "blurriness": lambda x: np.std(edge_filter(np.mean(x.image, axis=0))),
        "contrast": lambda x: np.nan_to_num(
            (np.max(x.get("percentiles")) - np.min(x.get("percentiles"))) / np.mean(x.get("percentiles"))
        ),
        "darkness": lambda x: x.get("percentiles")[1],
        "missing": lambda x: np.sum(np.isnan(x.image)) / np.prod(x.shape[-2:]),
        "zeros": lambda x: np.count_nonzero(x.image == 0) / np.prod(x.shape[-2:]),
        "percentiles": lambda x: np.nanpercentile(x.scaled, q=QUARTILES),
    }
    channel_function_map = {
        "brightness": lambda x: x.get("percentiles")[:, -2],
        "blurriness": lambda x: np.std(np.vectorize(edge_filter, signature="(m,n)->(m,n)")(x.image), axis=(1, 2)),
        "contrast": lambda x: np.nan_to_num(
            (np.max(x.get("percentiles"), axis=1) - np.min(x.get("percentiles"), axis=1))
            / np.mean(x.get("percentiles"), axis=1)
        ),
        "darkness": lambda x: x.get("percentiles")[:, 1],
        "missing": lambda x: np.sum(np.isnan(x.image), axis=(1, 2)) / np.prod(x.shape[-2:]),
        "zeros": lambda x: np.count_nonzero(x.image == 0, axis=(1, 2)) / np.prod(x.shape[-2:]),
        "percentiles": lambda x: np.nanpercentile(x.scaled, q=QUARTILES, axis=1).T,
    }


@dataclass(frozen=True)
class VisualStatsOutput(BaseStatsOutput):
    """
    Output class for :func:`visualstats` stats metric

    Attributes
    ----------
    brightness : NDArray[np.float16]
        Brightness of the images
    blurriness : NDArray[np.float16]
        Blurriness of the images
    contrast : NDArray[np.float16]
        Image contrast ratio
    darkness : NDArray[np.float16]
        Darkness of the images
    missing : NDArray[np.float16]
        Percentage of the images with missing pixels
    zeros : NDArray[np.float16]
        Percentage of the images with zero value pixels
    percentiles : NDArray[np.float16]
        Percentiles of the pixel values of the images with quartiles of (0, 25, 50, 75, 100)
    """

    brightness: NDArray[np.float16]
    blurriness: NDArray[np.float16]
    contrast: NDArray[np.float16]
    darkness: NDArray[np.float16]
    missing: NDArray[np.float16]
    zeros: NDArray[np.float16]
    percentiles: NDArray[np.float16]


@set_metadata("dataeval.metrics")
def visualstats(
    images: Iterable[ArrayLike],
    bboxes: Iterable[ArrayLike] | None = None,
    per_channel: bool = False,
) -> VisualStatsOutput:
    """
    Calculates visual statistics for each image

    This function computes various visual metrics (e.g., brightness, darkness, contrast, blurriness)
    on the images as a whole.

    Parameters
    ----------
    images : Iterable[ArrayLike]
        Images to perform calculations on
    bboxes : Iterable[ArrayLike] or None
        Bounding boxes in `xyxy` format for each image to perform calculations on

    Returns
    -------
    VisualStatsOutput
        A dictionary-like object containing the computed visual statistics for each image. The keys correspond
        to the names of the statistics (e.g., 'brightness', 'blurriness'), and the values are lists of results for
        each image or numpy arrays when the results are multi-dimensional.

    See Also
    --------
    dimensionstats, pixelstats, Outliers

    Note
    ----
    - `zeros` and `missing` are presented as a percentage of total pixel counts

    Examples
    --------
    Calculating the statistics on the images, whose shape is (C, H, W)

    >>> results = visualstats(images)
    >>> print(results.brightness)
    [0.0737 0.607  0.0713 0.1046 0.138  0.1713 0.2046 0.2379 0.2712 0.3047
     0.338  0.3713 0.4045 0.438  0.4712 0.5044 0.538  0.5713 0.6045 0.638
     0.6714 0.7046 0.738  0.7715 0.8047 0.838  0.871  0.905  0.938  0.971 ]
    >>> print(results.contrast)
    [2.041 1.332 1.293 1.279 1.272 1.268 1.265 1.263 1.261 1.26  1.259 1.258
     1.258 1.257 1.257 1.256 1.256 1.255 1.255 1.255 1.255 1.254 1.254 1.254
     1.254 1.254 1.254 1.253 1.253 1.253]
    """
    output = run_stats(images, bboxes, per_channel, VisualStatsProcessor, VisualStatsOutput)
    return VisualStatsOutput(**output)
