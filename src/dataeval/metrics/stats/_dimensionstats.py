from __future__ import annotations

__all__ = []

from collections.abc import Callable
from typing import Any

import numpy as np

from dataeval.metrics.stats._base import StatsProcessor, run_stats
from dataeval.outputs import DimensionStatsOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import ArrayLike, Dataset
from dataeval.utils._image import get_bitdepth


class DimensionStatsProcessor(StatsProcessor[DimensionStatsOutput]):
    output_class: type = DimensionStatsOutput
    image_function_map: dict[str, Callable[[StatsProcessor[DimensionStatsOutput]], Any]] = {
        "offset_x": lambda x: x.box.x0,
        "offset_y": lambda x: x.box.y0,
        "width": lambda x: x.box.width,
        "height": lambda x: x.box.height,
        "channels": lambda x: x.shape[-3],
        "size": lambda x: x.box.width * x.box.height,
        "aspect_ratio": lambda x: 0.0 if x.box.height == 0 else x.box.width / x.box.height,
        "depth": lambda x: get_bitdepth(x.raw).depth,
        "center": lambda x: np.asarray([(x.box.x0 + x.box.x1) / 2, (x.box.y0 + x.box.y1) / 2]),
        "distance_center": lambda x: np.sqrt(
            np.square(((x.box.x0 + x.box.x1) / 2) - (x.raw.shape[-1] / 2))
            + np.square(((x.box.y0 + x.box.y1) / 2) - (x.raw.shape[-2] / 2))
        ),
        "distance_edge": lambda x: np.min(
            [np.abs(x.box.x0), np.abs(x.box.y0), np.abs(x.box.x1 - x.raw.shape[-1]), np.abs(x.box.y1 - x.raw.shape[-2])]
        ),
    }


@set_metadata
def dimensionstats(
    dataset: Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]],
    *,
    per_box: bool = False,
) -> DimensionStatsOutput:
    """
    Calculates dimension :term:`statistics<Statistics>` for each image.

    This function computes various dimensional metrics (e.g., width, height, channels)
    on the images or individual bounding boxes for each image.

    Parameters
    ----------
    dataset : Dataset
        Dataset to perform calculations on.
    per_box : bool, default False
        If True, perform calculations on each bounding box.

    Returns
    -------
    DimensionStatsOutput
        A dictionary-like object containing the computed dimension statistics for each image or bounding
        box. The keys correspond to the names of the statistics (e.g., 'width', 'height'), and the values
        are lists of results for each image or :term:NumPy` arrays when the results are multi-dimensional.

    See Also
    --------
    pixelstats, visualstats, Outliers

    Examples
    --------
    Calculate the dimension statistics of a dataset of 8 images, whose shape is (C, H, W).

    >>> results = dimensionstats(dataset)
    >>> print(results.aspect_ratio)
    [1.    1.    1.333 1.    0.667 1.    1.    1.   ]
    >>> print(results.channels)
    [3 3 1 3 1 3 3 3]
    """
    return run_stats(dataset, per_box, False, [DimensionStatsProcessor])[0]
