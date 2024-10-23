from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from numpy.typing import ArrayLike

from dataeval._internal.metrics.stats.base import BaseStatsOutput, run_stats
from dataeval._internal.metrics.stats.dimensionstats import (
    DimensionStatsOutput,
    DimensionStatsProcessor,
)
from dataeval._internal.metrics.stats.labelstats import LabelStatsOutput, labelstats
from dataeval._internal.metrics.stats.pixelstats import PixelStatsOutput, PixelStatsProcessor
from dataeval._internal.metrics.stats.visualstats import VisualStatsOutput, VisualStatsProcessor
from dataeval._internal.output import OutputMetadata, set_metadata


@dataclass(frozen=True)
class DatasetStatsOutput(OutputMetadata):
    """
    Output class for :func:`datasetstats` stats metric

    This class represents the outputs of various stats functions against a single
    dataset, such that each index across all stat outputs are representative of
    the same source image.  Modifying or mixing outputs will result in inaccurate
    outlier calculations if not created correctly.

    Attributes
    ----------
    dimensionstats : DimensionStatsOutput
    pixelstats: PixelStatsOutput
    visualstats: VisualStatsOutput
    labelstats: LabelStatsOutput or None
    """

    dimensionstats: DimensionStatsOutput
    pixelstats: PixelStatsOutput
    visualstats: VisualStatsOutput
    labelstats: LabelStatsOutput | None = None

    def outputs(self) -> list[OutputMetadata]:
        return [s for s in (self.dimensionstats, self.pixelstats, self.visualstats, self.labelstats) if s is not None]

    def dict(self) -> dict[str, Any]:
        return {k: v for o in self.outputs() for k, v in o.dict().items()}

    def __post_init__(self):
        lengths = [len(s) for s in self.outputs() if isinstance(s, BaseStatsOutput)]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All StatsOutput classes must contain the same number of image sources.")


@dataclass(frozen=True)
class ChannelStatsOutput(OutputMetadata):
    """
    Output class for :func:`channelstats` stats metric

    This class represents the outputs of various per-channel stats functions against
    a single dataset, such that each index across all stat outputs are representative
    of the same source image.  Modifying or mixing outputs will result in inaccurate
    outlier calculations if not created correctly.

    Attributes
    ----------
    pixelstats: PixelStatsOutput
    visualstats: VisualStatsOutput
    """

    pixelstats: PixelStatsOutput
    visualstats: VisualStatsOutput

    def outputs(self) -> list[BaseStatsOutput]:
        return [self.pixelstats, self.visualstats]

    def dict(self) -> dict[str, Any]:
        return {**self.pixelstats.dict(), **self.visualstats.dict()}

    def __post_init__(self):
        lengths = [len(s) for s in self.outputs()]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All StatsOutput classes must contain the same number of image sources.")


@set_metadata("dataeval.metrics")
def datasetstats(
    images: Iterable[ArrayLike],
    bboxes: Iterable[ArrayLike] | None = None,
    labels: Iterable[ArrayLike] | None = None,
) -> DatasetStatsOutput:
    """
    Calculates various statistics for each image

    This function computes dimension, pixel and visual metrics
    on the images or individual bounding boxes for each image as
    well as label statistics if provided.

    Parameters
    ----------
    images : Iterable[ArrayLike]
        Images to perform calculations on
    bboxes : Iterable[ArrayLike] or None
        Bounding boxes in `xyxy` format for each image to perform calculations on
    labels : Iterable[ArrayLike] or None
        Labels of images or boxes to perform calculations on

    Returns
    -------
    DatasetStatsOutput
        Output class containing the outputs of various stats functions

    See Also
    --------
    dimensionstats, labelstats, pixelstats, visualstats, Outliers

    Examples
    --------
    Calculating the dimension, pixel and visual stats for a dataset with bounding boxes

    >>> stats = datasetstats(images, bboxes)
    >>> print(stats.dimensionstats.aspect_ratio)
    [ 0.864   0.5884 16.      1.143   1.692   0.5835  0.6665  2.555   1.3
      0.8335  1.      0.6     0.522  15.      3.834   1.75    0.75    0.7   ]
    >>> print(stats.visualstats.contrast)
    [1.744   1.946   0.1164  0.0635  0.0633  0.06274 0.0429  0.0317  0.0317
     0.02576 0.02081 0.02171 0.01915 0.01767 0.01799 0.01595 0.01433 0.01478]
    """
    outputs = run_stats(images, bboxes, False, [DimensionStatsProcessor, PixelStatsProcessor, VisualStatsProcessor])
    return DatasetStatsOutput(*outputs, labelstats=labelstats(labels) if labels else None)  # type: ignore


@set_metadata("dataeval.metrics")
def channelstats(
    images: Iterable[ArrayLike],
    bboxes: Iterable[ArrayLike] | None = None,
) -> ChannelStatsOutput:
    """
    Calculates various per-channel statistics for each image

    This function computes pixel and visual metrics on the images
    or individual bounding boxes for each image.

    Parameters
    ----------
    images : Iterable[ArrayLike]
        Images to perform calculations on
    bboxes : Iterable[ArrayLike] or None
        Bounding boxes in `xyxy` format for each image to perform calculations on

    Returns
    -------
    ChannelStatsOutput
        Output class containing the per-channel outputs of various stats functions

    See Also
    --------
    pixelstats, visualstats

    Examples
    --------
    Calculating the per-channel pixel and visual stats for a dataset

    >>> stats = channelstats(images)
    >>> print(stats.visualstats.darkness)
    [0.02124 0.1213  0.2212  0.1013  0.1076  0.11383 0.2013  0.2076  0.2139
     0.3013  0.3076  0.3137  0.4014  0.4075  0.4138  0.5015  0.508   0.5137
     0.6016  0.6074  0.614   0.701   0.7075  0.714   0.8013  0.8076  0.814
     0.9014  0.9077  0.914  ]
    """
    outputs = run_stats(images, bboxes, True, [PixelStatsProcessor, VisualStatsProcessor])
    return ChannelStatsOutput(*outputs)  # type: ignore
