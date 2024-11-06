from __future__ import annotations

__all__ = ["VisualStatsOutput", "visualstats"]

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval.metrics.stats.base import BaseStatsOutput, StatsProcessor, run_stats
from dataeval.output import set_metadata
from dataeval.utils.image import edge_filter

QUARTILES = (0, 25, 50, 75, 100)


@dataclass(frozen=True)
class VisualStatsOutput(BaseStatsOutput):
    """
    Output class for :func:`visualstats` stats metric

    Attributes
    ----------
    brightness : NDArray[np.float16]
        Brightness of the images
    contrast : NDArray[np.float16]
        Image contrast ratio
    darkness : NDArray[np.float16]
        Darkness of the images
    missing : NDArray[np.float16]
        Percentage of the images with missing pixels
    sharpness : NDArray[np.float16]
        Sharpness of the images
    zeros : NDArray[np.float16]
        Percentage of the images with zero value pixels
    percentiles : NDArray[np.float16]
        Percentiles of the pixel values of the images with quartiles of (0, 25, 50, 75, 100)
    """

    brightness: NDArray[np.float16]
    contrast: NDArray[np.float16]
    darkness: NDArray[np.float16]
    missing: NDArray[np.float16]
    sharpness: NDArray[np.float16]
    zeros: NDArray[np.float16]
    percentiles: NDArray[np.float16]


class VisualStatsProcessor(StatsProcessor[VisualStatsOutput]):
    output_class: type = VisualStatsOutput
    image_function_map: dict[str, Callable[[StatsProcessor[VisualStatsOutput]], Any]] = {
        "brightness": lambda x: x.get("percentiles")[1],
        "contrast": lambda x: np.nan_to_num(
            (np.max(x.get("percentiles")) - np.min(x.get("percentiles"))) / np.mean(x.get("percentiles"))
        ),
        "darkness": lambda x: x.get("percentiles")[-2],
        "missing": lambda x: np.count_nonzero(np.isnan(np.sum(x.image, axis=0))) / np.prod(x.shape[-2:]),
        "sharpness": lambda x: np.std(edge_filter(np.mean(x.image, axis=0))),
        "zeros": lambda x: np.count_nonzero(np.sum(x.image, axis=0) == 0) / np.prod(x.shape[-2:]),
        "percentiles": lambda x: np.nanpercentile(x.scaled, q=QUARTILES),
    }
    channel_function_map: dict[str, Callable[[StatsProcessor[VisualStatsOutput]], Any]] = {
        "brightness": lambda x: x.get("percentiles")[:, 1],
        "contrast": lambda x: np.nan_to_num(
            (np.max(x.get("percentiles"), axis=1) - np.min(x.get("percentiles"), axis=1))
            / np.mean(x.get("percentiles"), axis=1)
        ),
        "darkness": lambda x: x.get("percentiles")[:, -2],
        "missing": lambda x: np.count_nonzero(np.isnan(x.image), axis=(1, 2)) / np.prod(x.shape[-2:]),
        "sharpness": lambda x: np.std(np.vectorize(edge_filter, signature="(m,n)->(m,n)")(x.image), axis=(1, 2)),
        "zeros": lambda x: np.count_nonzero(x.image == 0, axis=(1, 2)) / np.prod(x.shape[-2:]),
        "percentiles": lambda x: np.nanpercentile(x.scaled, q=QUARTILES, axis=1).T,
    }


@set_metadata()
def visualstats(
    images: Iterable[ArrayLike],
    bboxes: Iterable[ArrayLike] | None = None,
    per_channel: bool = False,
) -> VisualStatsOutput:
    """
    Calculates visual statistics for each image

    This function computes various visual metrics (e.g., :term:`brightness<Brightness>`, darkness, contrast, blurriness)
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
        each image or :term:`NumPy` arrays when the results are multi-dimensional.

    See Also
    --------
    dimensionstats, pixelstats, Outliers

    Note
    ----
    - `zeros` and `missing` are presented as a percentage of total pixel counts

    Examples
    --------
    Calculating the :term:`statistics<Statistics>` on the images, whose shape is (C, H, W)

    >>> results = visualstats(images)
    >>> print(results.brightness)
    [0.02246 0.5557  0.06805 0.1014  0.1348  0.1681  0.2014  0.2347  0.268
     0.3015  0.3347  0.3682  0.4014  0.4348  0.468   0.5015  0.5347  0.568
     0.6016  0.635   0.668   0.701   0.735   0.768   0.8013  0.835   0.868
     0.9014  0.9346  0.9683 ]
    >>> print(results.contrast)
    [2.041 1.332 1.293 1.279 1.272 1.268 1.265 1.263 1.261 1.26  1.259 1.258
     1.258 1.257 1.257 1.256 1.256 1.255 1.255 1.255 1.255 1.254 1.254 1.254
     1.254 1.254 1.254 1.253 1.253 1.253]
    """
    return run_stats(images, bboxes, per_channel, [VisualStatsProcessor])[0]
