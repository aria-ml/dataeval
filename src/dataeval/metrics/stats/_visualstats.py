from __future__ import annotations

__all__ = []

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from dataeval._output import set_metadata
from dataeval.metrics.stats._base import BaseStatsOutput, StatsProcessor, run_stats
from dataeval.typing import ArrayLike, Dataset
from dataeval.utils._image import edge_filter

QUARTILES = (0, 25, 50, 75, 100)


@dataclass(frozen=True)
class VisualStatsOutput(BaseStatsOutput):
    """
    Output class for :func:`.visualstats` stats metric.

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
        "contrast": lambda x: 0
        if np.mean(x.get("percentiles")) == 0
        else (np.max(x.get("percentiles")) - np.min(x.get("percentiles"))) / np.mean(x.get("percentiles")),
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

    Note
    ----
    - `zeros` and `missing` are presented as a percentage of total pixel counts

    Examples
    --------
    Calculate the visual statistics of a dataset of 8 images, whose shape is (C, H, W).

    >>> results = visualstats(dataset)
    >>> print(results.brightness)
    [0.084 0.13  0.259 0.38  0.508 0.63  0.755 0.88 ]
    >>> print(results.contrast)
    [2.04  1.331 1.261 1.279 1.253 1.268 1.265 1.263]
    """
    return run_stats(dataset, per_box, per_channel, [VisualStatsProcessor])[0]
