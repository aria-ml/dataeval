from __future__ import annotations

__all__ = []

from dataclasses import dataclass
from typing import Any, Iterable

from numpy.typing import ArrayLike

from dataeval.metrics.stats.base import BaseStatsOutput, run_stats
from dataeval.metrics.stats.dimensionstats import (
    DimensionStatsOutput,
    DimensionStatsProcessor,
)
from dataeval.metrics.stats.labelstats import LabelStatsOutput, labelstats
from dataeval.metrics.stats.pixelstats import PixelStatsOutput, PixelStatsProcessor
from dataeval.metrics.stats.visualstats import VisualStatsOutput, VisualStatsProcessor
from dataeval.output import Output, set_metadata


@dataclass(frozen=True)
class DatasetStatsOutput(Output):
    """
    Output class for :func:`datasetstats` stats metric

    This class represents the outputs of various stats functions against a single
    dataset, such that each index across all stat outputs are representative of
    the same source image. Modifying or mixing outputs will result in inaccurate
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

    def _outputs(self) -> list[Output]:
        return [s for s in (self.dimensionstats, self.pixelstats, self.visualstats, self.labelstats) if s is not None]

    def dict(self) -> dict[str, Any]:
        return {k: v for o in self._outputs() for k, v in o.dict().items()}

    def __post_init__(self) -> None:
        lengths = [len(s) for s in self._outputs() if isinstance(s, BaseStatsOutput)]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All StatsOutput classes must contain the same number of image sources.")


@dataclass(frozen=True)
class ChannelStatsOutput(Output):
    """
    Output class for :func:`channelstats` stats metric

    This class represents the outputs of various per-channel stats functions against
    a single dataset, such that each index across all stat outputs are representative
    of the same source image. Modifying or mixing outputs will result in inaccurate
    outlier calculations if not created correctly.

    Attributes
    ----------
    pixelstats: PixelStatsOutput
    visualstats: VisualStatsOutput
    """

    pixelstats: PixelStatsOutput
    visualstats: VisualStatsOutput

    def _outputs(self) -> tuple[PixelStatsOutput, VisualStatsOutput]:
        return (self.pixelstats, self.visualstats)

    def dict(self) -> dict[str, Any]:
        return {**self.pixelstats.dict(), **self.visualstats.dict()}

    def __post_init__(self) -> None:
        lengths = [len(s) for s in self._outputs()]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All StatsOutput classes must contain the same number of image sources.")


@set_metadata
def datasetstats(
    images: Iterable[ArrayLike],
    bboxes: Iterable[ArrayLike] | None = None,
    labels: Iterable[ArrayLike] | None = None,
) -> DatasetStatsOutput:
    """
    Calculates various :term:`statistics<Statistics>` for each image

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

    >>> stats = datasetstats(stats_images, bboxes)
    >>> print(stats.dimensionstats.aspect_ratio)
    [ 0.864   0.5884 16.      1.143   1.692   0.5835  0.6665  2.555   1.3   ]
    >>> print(stats.visualstats.sharpness)
    [4.04   4.434  0.2778 4.957  5.145  5.22   4.957  3.076  2.855 ]
    """
    outputs = run_stats(images, bboxes, False, [DimensionStatsProcessor, PixelStatsProcessor, VisualStatsProcessor])
    return DatasetStatsOutput(*outputs, labelstats=labelstats(labels) if labels else None)  # type: ignore


@set_metadata
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

    >>> stats = channelstats(stats_images)
    >>> print(stats.visualstats.darkness)
    [0.1499 0.3499 0.55   0.2094 0.2219 0.2344 0.4194 0.6094 0.622  0.6343
     0.8154]
    """
    outputs = run_stats(images, bboxes, True, [PixelStatsProcessor, VisualStatsProcessor])
    return ChannelStatsOutput(*outputs)  # type: ignore
