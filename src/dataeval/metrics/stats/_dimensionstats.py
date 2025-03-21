from __future__ import annotations

__all__ = []

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
from numpy.typing import NDArray

from dataeval._output import set_metadata
from dataeval.metrics.stats._base import BaseStatsOutput, HistogramPlotMixin, StatsProcessor, run_stats
from dataeval.typing import ArrayLike
from dataeval.utils._image import get_bitdepth


@dataclass(frozen=True)
class DimensionStatsOutput(BaseStatsOutput, HistogramPlotMixin):
    """
    Output class for :func:`.dimensionstats` stats metric.

    Attributes
    ----------
    left : NDArray[np.int32]
        Offsets from the left edge of images in pixels
    top : NDArray[np.int32]
        Offsets from the top edge of images in pixels
    width : NDArray[np.uint32]
        Width of the images in pixels
    height : NDArray[np.uint32]
        Height of the images in pixels
    channels : NDArray[np.uint8]
        Channel count of the images in pixels
    size : NDArray[np.uint32]
        Size of the images in pixels
    aspect_ratio : NDArray[np.float16]
        :term:`ASspect Ratio<Aspect Ratio>` of the images (width/height)
    depth : NDArray[np.uint8]
        Color depth of the images in bits
    center : NDArray[np.uint16]
        Offset from center in [x,y] coordinates of the images in pixels
    distance : NDArray[np.float16]
        Distance in pixels from center
    """

    left: NDArray[np.int32]
    top: NDArray[np.int32]
    width: NDArray[np.uint32]
    height: NDArray[np.uint32]
    channels: NDArray[np.uint8]
    size: NDArray[np.uint32]
    aspect_ratio: NDArray[np.float16]
    depth: NDArray[np.uint8]
    center: NDArray[np.int16]
    distance: NDArray[np.float16]


class DimensionStatsProcessor(StatsProcessor[DimensionStatsOutput]):
    output_class: type = DimensionStatsOutput
    image_function_map: dict[str, Callable[[StatsProcessor[DimensionStatsOutput]], Any]] = {
        "left": lambda x: x.box[0],
        "top": lambda x: x.box[1],
        "width": lambda x: x.box[2] - x.box[0],
        "height": lambda x: x.box[3] - x.box[1],
        "channels": lambda x: x.shape[-3],
        "size": lambda x: (x.box[2] - x.box[0]) * (x.box[3] - x.box[1]),
        "aspect_ratio": lambda x: (x.box[2] - x.box[0]) / (x.box[3] - x.box[1]),
        "depth": lambda x: get_bitdepth(x.image).depth,
        "center": lambda x: np.asarray([(x.box[0] + x.box[2]) / 2, (x.box[1] + x.box[3]) / 2]),
        "distance": lambda x: np.sqrt(
            np.square(((x.box[0] + x.box[2]) / 2) - (x.shape[-1] / 2))
            + np.square(((x.box[1] + x.box[3]) / 2) - (x.shape[-2] / 2))
        ),
    }


@set_metadata
def dimensionstats(
    images: Iterable[ArrayLike],
    bboxes: Iterable[ArrayLike] | None = None,
) -> DimensionStatsOutput:
    """
    Calculates dimension :term:`statistics<Statistics>` for each image.

    This function computes various dimensional metrics (e.g., width, height, channels)
    on the images or individual bounding boxes for each image.

    Parameters
    ----------
    images : Iterable[ArrayLike]
        Images to perform calculations on
    bboxes : Iterable[ArrayLike] or None
        Bounding boxes in `xyxy` format for each image to perform calculations on

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
    Calculating the dimension statistics on the images, whose shape is (C, H, W)

    >>> results = dimensionstats(stats_images)
    >>> print(results.aspect_ratio)
    [1.    1.    1.333 1.    0.667]
    >>> print(results.channels)
    [3 3 1 3 1]
    """
    return run_stats(images, bboxes, False, [DimensionStatsProcessor])[0]
