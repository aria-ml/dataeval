from __future__ import annotations

__all__ = []

from dataclasses import dataclass
from typing import Any, Iterable

from dataeval._output import Output, set_metadata
from dataeval.metrics.stats._base import BaseStatsOutput, HistogramPlotMixin, _is_plottable, run_stats
from dataeval.metrics.stats._dimensionstats import DimensionStatsOutput, DimensionStatsProcessor
from dataeval.metrics.stats._labelstats import LabelStatsOutput, labelstats
from dataeval.metrics.stats._pixelstats import PixelStatsOutput, PixelStatsProcessor
from dataeval.metrics.stats._visualstats import VisualStatsOutput, VisualStatsProcessor
from dataeval.typing import ArrayLike
from dataeval.utils._plot import channel_histogram_plot


@dataclass(frozen=True)
class DatasetStatsOutput(Output, HistogramPlotMixin):
    """
    Output class for :func:`.datasetstats` stats metric.

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

    _excluded_keys = ["histogram", "percentiles"]

    def _outputs(self) -> list[Output]:
        return [s for s in (self.dimensionstats, self.pixelstats, self.visualstats, self.labelstats) if s is not None]

    def dict(self) -> dict[str, Any]:
        return {k: v for o in self._outputs() for k, v in o.dict().items()}

    def __post_init__(self) -> None:
        lengths = [len(s) for s in self._outputs() if isinstance(s, BaseStatsOutput)]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All StatsOutput classes must contain the same number of image sources.")


def _get_channels(cls, channel_limit: int | None = None, channel_index: int | Iterable[int] | None = None):
    raw_channels = max([si.channel for si in cls.dict()["source_index"]]) + 1
    if isinstance(channel_index, int):
        max_channels = 1 if channel_index < raw_channels else raw_channels
        ch_mask = cls.pixelstats.get_channel_mask(channel_index)
    elif isinstance(channel_index, Iterable) and all(isinstance(val, int) for val in list(channel_index)):
        max_channels = len(list(channel_index))
        ch_mask = cls.pixelstats.get_channel_mask(channel_index)
    elif isinstance(channel_limit, int):
        max_channels = channel_limit
        ch_mask = cls.pixelstats.get_channel_mask(None, channel_limit)
    else:
        max_channels = raw_channels
        ch_mask = None

    if max_channels > raw_channels:
        max_channels = raw_channels
    if ch_mask is not None and not any(ch_mask):
        ch_mask = None

    return max_channels, ch_mask


@dataclass(frozen=True)
class ChannelStatsOutput(Output):
    """
    Output class for :func:`.channelstats` stats metric.

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

    def plot(
        self, log: bool, channel_limit: int | None = None, channel_index: int | Iterable[int] | None = None
    ) -> None:
        max_channels, ch_mask = _get_channels(self, channel_limit, channel_index)
        data_dict = {k: v for k, v in self.dict().items() if _is_plottable(k, v, ("histogram", "percentiles"))}
        channel_histogram_plot(data_dict, log, max_channels, ch_mask)


@set_metadata
def datasetstats(
    images: Iterable[ArrayLike],
    bboxes: Iterable[ArrayLike] | None = None,
    labels: Iterable[ArrayLike] | None = None,
) -> DatasetStatsOutput:
    """
    Calculates various :term:`statistics<Statistics>` for each image.

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
    [ 0.864  0.588 16.     1.143  1.692  0.583  0.667  2.555  1.3  ]
    >>> print(stats.visualstats.sharpness)
    [4.04  4.434 0.278 4.957 5.145 5.22  4.957 3.076 2.855]
    """
    outputs = run_stats(images, bboxes, False, [DimensionStatsProcessor, PixelStatsProcessor, VisualStatsProcessor])
    return DatasetStatsOutput(*outputs, labelstats=labelstats(labels) if labels else None)  # type: ignore


@set_metadata
def channelstats(
    images: Iterable[ArrayLike],
    bboxes: Iterable[ArrayLike] | None = None,
) -> ChannelStatsOutput:
    """
    Calculates various per-channel :term:`statistics` for each image.

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
    [0.15  0.35  0.55  0.209 0.222 0.234 0.419 0.609 0.622 0.634 0.815]
    """
    outputs = run_stats(images, bboxes, True, [PixelStatsProcessor, VisualStatsProcessor])
    return ChannelStatsOutput(*outputs)  # type: ignore
