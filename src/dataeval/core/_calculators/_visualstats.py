__all__ = []

from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.core._calculators._base import ALL_VIEWS, Calculator, Handler
from dataeval.core._calculators._cache import CalculatorCache
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.flags import ImageStats
from dataeval.utils._internal import EPSILON
from dataeval.utils.preprocessing import edge_filter

QUARTILES = (0, 25, 50, 75, 100)


@CalculatorRegistry.register(ImageStats)
class VisualStatCalculator(Calculator[ImageStats]):
    """Calculator for visual statistics like brightness, contrast, sharpness.

    Reads ``CalculatorCache.perceptual`` throughout rather than the raw or normalized view:
    these statistics stand in for how an image looks to a person, and that is a position
    between black and white rather than a value in whatever units the sensor wrote.
    """

    def __init__(self, datum: NDArray[Any], cache: "CalculatorCache", per_channel: bool = False) -> None:
        self.datum = datum
        self.cache = cache
        self.per_channel_mode = per_channel

    @cached_property
    def _unreadable(self) -> bool:
        """Whether there is no reading to take: nothing measured, or nothing to measure it against.

        The second half is what separates this family from the pixel one. A band carrying
        elevation or temperature has values but no full-scale reference, so *how bright is
        it* has no answer — reported as NaN rather than raised, unlike a histogram, which
        genuinely cannot be binned without an interval.
        """
        return self.cache.is_unmeasurable

    @cached_property
    def percentiles(self) -> NDArray[np.float64]:
        if self._unreadable:
            if self.per_channel_mode:
                return self.cache.nan_like((self.cache.channel_count, len(QUARTILES)))
            return self.cache.nan_like((len(QUARTILES),))
        # The perceptual view, never `scaled`: a visual statistic reports where values sit
        # between black and white, which `normalize_pixel_values` has no bearing on.
        if self.per_channel_mode:
            return np.nanpercentile(self.cache.per_channel_perceptual, q=QUARTILES, axis=1).T.astype(np.float64)
        return np.nanpercentile(self.cache.perceptual, q=QUARTILES).astype(np.float64)

    def get_applicable_flags(self) -> ImageStats:
        """Return which flags this calculator handles."""
        return ImageStats.VISUAL

    def _brightness(self) -> list[float]:
        if self.per_channel_mode:
            return self.percentiles[:, 1].tolist()
        return [float(self.percentiles[1])]

    def _contrast(self) -> list[float]:
        if self.per_channel_mode:
            return (
                (np.max(self.percentiles, axis=1) - np.min(self.percentiles, axis=1))
                / (np.mean(self.percentiles, axis=1) + EPSILON)
            ).tolist()
        return [float(np.max(self.percentiles) - np.min(self.percentiles)) / float(np.mean(self.percentiles) + EPSILON)]

    def _darkness(self) -> list[float]:
        if self.per_channel_mode:
            return self.percentiles[:, -2].tolist()
        return [float(self.percentiles[-2])]

    def _sharpness(self) -> list[float]:
        # Sharpness requires 2D spatial data; return NaN for low-dimensional or unreadable data
        if self._unreadable:
            return [np.nan] * self.cache.channel_count if self.per_channel_mode else [np.nan]
        if self.cache.image.ndim < 2:
            return [np.nan] if not self.per_channel_mode else [np.nan] * self.cache.channel_count
        # Edge magnitudes off the perceptual view, so the same picture at two bit depths
        # gives one answer. `edge_filter` clips to 0-255, which is the range this lands on.
        perceptual = self.cache.perceptual
        if self.cache.image.ndim == 2:
            # 2D data: treat as single-channel image
            return [float(np.nanstd(edge_filter(perceptual)))]
        # 3D+ data with channels
        if self.per_channel_mode:
            return np.nanstd(
                np.vectorize(edge_filter, signature="(m,n)->(m,n)")(perceptual),
                axis=(1, 2),
            ).tolist()
        return [float(np.nanstd(edge_filter(np.mean(perceptual, axis=0))))]

    def _percentiles(self) -> list[Any]:
        if self.per_channel_mode:
            return self.percentiles.tolist()
        return [self.percentiles.tolist()]

    def get_empty_values(self) -> dict[str, Any]:
        """Return empty values for visual statistics."""
        return {
            "percentiles": [np.nan] * 5,  # 5 percentiles: 0, 25, 50, 75, 100
        }

    def get_handlers(self) -> dict[ImageStats, Handler]:
        """Return mapping of flags to the statistic each produces.

        Percentiles and edge magnitudes are both taken NaN-aware, so a masked region has
        an answer; and a band group is a different picture, so it has one too — asking how
        bright the visible bands are separately from the near-infrared is the question this
        family exists to answer on multispectral data.
        """
        return {
            ImageStats.VISUAL_BRIGHTNESS: Handler("brightness", self._brightness, ALL_VIEWS),
            ImageStats.VISUAL_CONTRAST: Handler("contrast", self._contrast, ALL_VIEWS),
            ImageStats.VISUAL_DARKNESS: Handler("darkness", self._darkness, ALL_VIEWS),
            ImageStats.VISUAL_SHARPNESS: Handler("sharpness", self._sharpness, ALL_VIEWS),
            ImageStats.VISUAL_PERCENTILES: Handler("percentiles", self._percentiles, ALL_VIEWS),
        }
