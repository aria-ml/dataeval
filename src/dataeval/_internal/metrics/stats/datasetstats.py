from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from numpy.typing import ArrayLike

from dataeval._internal.metrics.stats.base import BaseStatsOutput
from dataeval._internal.metrics.stats.dimensionstats import DimensionStatsOutput, dimensionstats
from dataeval._internal.metrics.stats.labelstats import LabelStatsOutput, labelstats
from dataeval._internal.metrics.stats.pixelstats import PixelStatsOutput, pixelstats
from dataeval._internal.metrics.stats.visualstats import VisualStatsOutput, visualstats
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
    dimensionstats : DimensionStatsOutput or None
    pixelstats: PixelStatsOutput or None
    visualstats: VisualStatsOutput or None
    labelstats: LabelStatsOutput or None, default None
    """

    dimensionstats: DimensionStatsOutput | None
    pixelstats: PixelStatsOutput | None
    visualstats: VisualStatsOutput | None
    labelstats: LabelStatsOutput | None = None

    def outputs(self) -> list[BaseStatsOutput]:
        return [s for s in (self.dimensionstats, self.pixelstats, self.visualstats) if s is not None]

    def __post_init__(self):
        lengths = [len(s) for s in self.outputs()]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All StatsOutput classes must contain the same number of image sources.")


@set_metadata("dataeval.metrics")
def datasetstats(
    images: Iterable[ArrayLike],
    bboxes: Iterable[ArrayLike] | None = None,
    labels: Iterable[ArrayLike] | None = None,
    use_dimension: bool = True,
    use_pixel: bool = True,
    use_visual: bool = True,
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
    return DatasetStatsOutput(
        dimensionstats(images, bboxes) if use_dimension else None,
        pixelstats(images, bboxes) if use_pixel else None,
        visualstats(images, bboxes) if use_visual else None,
        labelstats(labels) if labels else None,
    )
