from __future__ import annotations

__all__ = []

from dataclasses import dataclass
from typing import Any, Literal, overload

from dataeval._output import set_metadata
from dataeval.metrics.stats._base import run_stats
from dataeval.metrics.stats._dimensionstats import DimensionStatsOutput, DimensionStatsProcessor
from dataeval.metrics.stats._pixelstats import PixelStatsOutput, PixelStatsProcessor
from dataeval.metrics.stats._visualstats import VisualStatsOutput, VisualStatsProcessor
from dataeval.typing import (
    ArrayLike,
    Dataset,
)


@dataclass(frozen=True)
class ImageStatsOutput(DimensionStatsOutput, PixelStatsOutput, VisualStatsOutput):
    """
    Output class for :func:`.imagestats` stats metric.

    This class represents the combined outputs of various stats functions against a
    single dataset, such that each index across all stat outputs are representative
    of the same source image. Modifying or mixing outputs will result in inaccurate
    outlier calculations if not created correctly.

    See Also
    --------
    DimensionStatsOutput, PixelStatsOutput, VisualStatsOutput
    """


@dataclass(frozen=True)
class ChannelStatsOutput(PixelStatsOutput, VisualStatsOutput):
    """
    Output class for :func:`.imagestats` stats metric.

    This class represents the outputs of various per-channel stats functions against
    a single dataset, such that each index across all stat outputs are representative
    of the same source image. Modifying or mixing outputs will result in inaccurate
    outlier calculations if not created correctly.

    See Also
    --------
    PixelStatsOutput, VisualStatsOutput
    """


@overload
def imagestats(
    dataset: Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]],
    *,
    per_box: bool = False,
    per_channel: Literal[True],
) -> ChannelStatsOutput: ...


@overload
def imagestats(
    dataset: Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]],
    *,
    per_box: bool = False,
    per_channel: Literal[False] = False,
) -> ImageStatsOutput: ...


@set_metadata
def imagestats(
    dataset: Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]],
    *,
    per_box: bool = False,
    per_channel: bool = False,
) -> ImageStatsOutput | ChannelStatsOutput:
    """
    Calculates various :term:`statistics<Statistics>` for each image.

    This function computes dimension, pixel and visual metrics
    on the images or individual bounding boxes for each image as
    well as label statistics if provided.

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
    ImageStatsOutput or ChannelStatsOutput
        Output class containing the outputs of various stats functions

    See Also
    --------
    dimensionstats, labelstats, pixelstats, visualstats, Outliers

    Examples
    --------
    Calculate dimension, pixel and visual statistics for a dataset containing 8
    images.

    >>> stats = imagestats(dataset)
    >>> print(stats.aspect_ratio)
    [1.    1.    1.333 1.    0.667 1.    1.    1.   ]

    >>> print(stats.sharpness)
    [20.23 20.23 23.33 20.23 77.06 20.23 20.23 20.23]

    Calculate the pixel and visual stats for a dataset containing 6 3-channel
    images and 2 1-channel images for a total of 20 channels.

    >>> ch_stats = imagestats(dataset, per_channel=True)
    >>> print(ch_stats.brightness)
    [0.027 0.152 0.277 0.127 0.135 0.142 0.259 0.377 0.385 0.392 0.508 0.626
     0.634 0.642 0.751 0.759 0.767 0.876 0.884 0.892]
    """
    if per_channel:
        processors = [PixelStatsProcessor, VisualStatsProcessor]
        output_cls = ChannelStatsOutput
    else:
        processors = [DimensionStatsProcessor, PixelStatsProcessor, VisualStatsProcessor]
        output_cls = ImageStatsOutput

    outputs = run_stats(dataset, per_box, per_channel, processors)
    return output_cls(**{k: v for d in outputs for k, v in d.dict().items()})
