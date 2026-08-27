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


def _percentiles_by_count(counts: NDArray[np.intp], quantiles: tuple[int, ...]) -> NDArray[np.float64]:
    """Percentiles read off value counts rather than out of the sorted values.

    `np.percentile` partitions the values, which over a megapixel image is a sort's worth
    of cache misses across the whole array and the single largest cost in this family. An
    8-bit view holds only 256 distinct values, so `CalculatorCache.display_counts` locates
    every order statistic at once and the values are never rearranged.

    Exact, not approximate: this reproduces NumPy's default ``linear`` method bit for bit,
    by resolving the same virtual index against the running totals and interpolating
    between the same two order statistics it would have partitioned to.
    """
    cumulative = np.cumsum(counts)
    # NumPy places quantile q at index q/100 * (n - 1) of the sorted values, and reads
    # between its neighbours when that lands off a whole index.
    virtual = np.asarray(quantiles, dtype=np.float64) / 100.0 * (cumulative[-1] - 1)
    below = np.floor(virtual)
    # The k-th smallest value is the first level whose running total has passed k.
    lower = np.searchsorted(cumulative, below, side="right").astype(np.float64)
    upper = np.searchsorted(cumulative, np.ceil(virtual), side="right").astype(np.float64)
    return lower + (virtual - below) * (upper - lower)


@CalculatorRegistry.register(ImageStats)
class VisualStatCalculator(Calculator[ImageStats]):
    """Calculator for visual statistics like brightness, contrast, sharpness.

    Reads ``CalculatorCache.perceptual`` throughout rather than the raw or normalized view:
    these statistics stand in for how an image looks to a person, and that is a position
    between black and white rather than a value in whatever units the sensor wrote.
    """

    def __init__(self, datum: NDArray[Any], cache: "CalculatorCache") -> None:
        self.datum = datum
        self.cache = cache

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
            return self.cache.nan_like((len(QUARTILES),))
        return self._whole_percentiles()

    def _whole_percentiles(self) -> NDArray[np.float64]:
        """Quartiles over every value in the view at once.

        Read off the perceptual view, never `scaled`: a visual statistic reports where
        values sit between black and white, which `normalize_pixel_values` has no bearing
        on.
        """
        counts = self.cache.display_counts
        if counts is not None:
            # Summed across bands, which is the same multiset the flat view would sort.
            return _percentiles_by_count(counts.sum(axis=0), QUARTILES)
        return np.nanpercentile(self.cache.perceptual, q=QUARTILES).astype(np.float64)

    def get_applicable_flags(self) -> ImageStats:
        """Return which flags this calculator handles."""
        return ImageStats.VISUAL

    def _brightness(self) -> list[float]:
        return [float(self.percentiles[1])]

    def _contrast(self) -> list[float]:
        return [float(np.max(self.percentiles) - np.min(self.percentiles)) / float(np.mean(self.percentiles) + EPSILON)]

    def _darkness(self) -> list[float]:
        return [float(self.percentiles[-2])]

    def _deviation(self, values: NDArray[Any], **kwargs: Any) -> Any:
        """Spread of `values`, NaN-aware only where the view it came from can hold NaN.

        `np.nanstd` scans for NaN before it reduces, which over a megapixel edge image is
        a second pass for an answer the view already knows.
        """
        return np.std(values, **kwargs) if self.cache.is_fully_measured else np.nanstd(values, **kwargs)

    def _sharpness(self) -> list[float]:
        # Sharpness requires 2D spatial data; return NaN for low-dimensional or unreadable data
        if self._unreadable:
            return [np.nan]
        grayscale = self.cache.display_grayscale
        if grayscale is None:
            return self._sharpness_from_view()
        # Read straight off the window. Reaching this through `perceptual` instead would
        # promote every band into a float64 copy of the whole datum, only to average it
        # back down to the one plane the filter runs over.
        return [float(self._deviation(edge_filter(grayscale)))]

    def _sharpness_from_view(self) -> list[float]:
        """Sharpness off :attr:`~CalculatorCache.perceptual`, for the views no window slice covers."""
        if self.cache.image.ndim < 2:
            return [np.nan]
        # Edge magnitudes off the perceptual view, so the same picture at two bit depths
        # gives one answer. `edge_filter` clips to 0-255, which is the range this lands on.
        perceptual = self.cache.perceptual
        if self.cache.image.ndim == 2:
            # 2D data: treat as single-channel image
            return [float(self._deviation(edge_filter(perceptual)))]
        # 3D+ data with channels
        return [float(self._deviation(edge_filter(np.mean(perceptual, axis=0))))]

    def _percentiles(self) -> list[Any]:
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
