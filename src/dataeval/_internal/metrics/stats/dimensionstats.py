from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval._internal.metrics.stats.base import BaseStatsOutput, StatsFunctionMap, run_stats
from dataeval._internal.metrics.utils import get_bitdepth
from dataeval._internal.output import set_metadata


class DimensionStatsFunctionMap(StatsFunctionMap):
    image = {
        "left": lambda x: np.uint16(x.box[0]),
        "top": lambda x: np.uint16(x.box[1]),
        "width": lambda x: np.uint16(x.box[2]),
        "height": lambda x: np.uint16(x.box[3]),
        "channels": lambda x: np.uint8(x.shape[-3]),
        "size": lambda x: np.uint32(np.prod(x.shape[-2:])),
        "aspect_ratio": lambda x: np.float16(x.shape[-1] / x.shape[-2]),
        "depth": lambda x: np.uint8(get_bitdepth(x.image).depth),
        "center": lambda x: np.asarray([(x.box[0] + x.box[2]) / 2, (x.box[1] + x.box[3]) / 2], dtype=np.uint16),
    }


@dataclass(frozen=True)
class DimensionStatsOutput(BaseStatsOutput):
    """
    Attributes
    ----------
    left : NDArray[np.uint16]
        Offsets from the left edge of images in pixels
    top : NDArray[np.uint16]
        Offsets from the top edge of images in pixels
    width : NDArray[np.uint16]
        Width of the images in pixels
    height : NDArray[np.uint16]
        Height of the images in pixels
    channels : NDArray[np.uint8]
        Channel count of the images in pixels
    size : NDArray[np.uint32]
        Size of the images in pixels
    aspect_ratio : NDArray[np.float16]
        Aspect ratio of the images (width/height)
    depth : NDArray[np.uint8]
        Color depth of the images in bits
    """

    left: NDArray[np.uint16]
    top: NDArray[np.uint16]
    width: NDArray[np.uint16]
    height: NDArray[np.uint16]
    channels: NDArray[np.uint8]
    size: NDArray[np.uint32]
    aspect_ratio: NDArray[np.float16]
    depth: NDArray[np.uint8]


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
        Bounding boxes for each image to perform calculations on

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
    output = run_stats(images, bboxes, False, DimensionStatsFunctionMap(), DimensionStatsOutput)
    return DimensionStatsOutput(**output)
