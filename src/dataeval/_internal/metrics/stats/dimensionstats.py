from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval._internal.metrics.stats.base import BaseStatsOutput, StatsProcessor, run_stats
from dataeval._internal.metrics.utils import get_bitdepth
from dataeval._internal.output import set_metadata


@dataclass(frozen=True)
class DimensionStatsOutput(BaseStatsOutput):
    """
    Output class for :func:`dimensionstats` stats metric

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
        Aspect ratio of the images (width/height)
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
    output_class = DimensionStatsOutput
    image_function_map = {
        "left": lambda x: x.box[0],
        "top": lambda x: x.box[1],
        "width": lambda x: x.shape[-1],
        "height": lambda x: x.shape[-2],
        "channels": lambda x: x.shape[-3],
        "size": lambda x: np.prod(x.shape[-2:]),
        "aspect_ratio": lambda x: x.shape[-1] / x.shape[-2],
        "depth": lambda x: get_bitdepth(x.image).depth,
        "center": lambda x: np.asarray([(x.box[0] + x.box[2]) / 2, (x.box[1] + x.box[3]) / 2]),
        "distance": lambda x: np.sqrt(
            np.square(((x.box[0] + x.box[2]) / 2) - (x.width / 2))
            + np.square(((x.box[1] + x.box[3]) / 2) - (x.height / 2))
        ),
    }


@set_metadata("dataeval.metrics")
def dimensionstats(
    images: Iterable[ArrayLike],
    bboxes: Iterable[ArrayLike] | None = None,
) -> DimensionStatsOutput:
    """
    Calculates dimension statistics for each image

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
        are lists of results for each image or numpy arrays when the results are multi-dimensional.

    See Also
    --------
    pixelstats, visualstats, Outliers

    Examples
    --------
    Calculating the dimension statistics on the images, whose shape is (C, H, W)

    >>> results = dimensionstats(images)
    >>> print(results.aspect_ratio)
    [0.75  0.75  0.75  0.75  0.75  0.75  1.333 0.75  0.75  1.   ]
    >>> print(results.channels)
    [1 1 1 1 1 1 3 1 1 3]
    """
    return run_stats(images, bboxes, False, [DimensionStatsProcessor])[0]
