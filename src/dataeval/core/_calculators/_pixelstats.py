__all__ = []

from collections.abc import Callable
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy, kurtosis, skew

from dataeval.core._calculators._base import Calculator
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.flags import ImageStats

if TYPE_CHECKING:
    from dataeval.core._calculate import CalculatorCache


@CalculatorRegistry.register(ImageStats)
class PixelStatCalculator(Calculator[ImageStats]):
    """Calculator for pixel-level statistics."""

    def __init__(self, datum: NDArray[Any], cache: "CalculatorCache", per_channel: bool = False) -> None:
        self.datum = datum
        self.cache = cache
        self.per_channel_mode = per_channel

    @cached_property
    def histogram(self) -> NDArray[np.float64]:
        if self.per_channel_mode:
            return np.apply_along_axis(lambda y: np.histogram(y, bins=256, range=(0, 1))[0], 1, self.cache.per_channel)
        return np.histogram(self.cache.scaled, bins=256, range=(0, 1))[0]

    def get_applicable_flags(self) -> ImageStats:
        """Return which flags this calculator handles."""
        return ImageStats.PIXEL

    def _mean(self) -> list[float]:
        if self.per_channel_mode:
            return np.nanmean(self.cache.per_channel, axis=1).tolist()
        return [float(np.nanmean(self.cache.scaled))]

    def _std(self) -> list[float]:
        if self.per_channel_mode:
            return np.nanstd(self.cache.per_channel, axis=1).tolist()
        return [float(np.nanstd(self.cache.scaled))]

    def _var(self) -> list[float]:
        if self.per_channel_mode:
            return np.nanvar(self.cache.per_channel, axis=1).tolist()
        return [float(np.nanvar(self.cache.scaled))]

    def _skew(self) -> list[float]:
        if self.per_channel_mode:
            return skew(self.cache.per_channel, axis=1, nan_policy="omit").tolist()
        return [float(skew(self.cache.scaled.ravel(), nan_policy="omit"))]

    def _kurtosis(self) -> list[float]:
        if self.per_channel_mode:
            return kurtosis(self.cache.per_channel, axis=1, nan_policy="omit").tolist()
        return [float(kurtosis(self.cache.scaled.ravel(), nan_policy="omit"))]

    def _entropy(self) -> list[float]:
        if self.per_channel_mode:
            return np.asarray(entropy(self.histogram, axis=1)).tolist()
        return [float(entropy(self.histogram))]

    def _missing(self) -> list[float]:
        if self.per_channel_mode:
            return (
                np.count_nonzero(np.isnan(self.cache.per_channel), axis=1) / self.cache.per_channel.shape[1]
            ).tolist()
        return [float(np.count_nonzero(np.isnan(self.cache.image)) / self.cache.image.size)]

    def _zeros(self) -> list[float]:
        if self.per_channel_mode:
            return (np.count_nonzero(self.cache.per_channel == 0, axis=1) / self.cache.per_channel.shape[1]).tolist()
        return [float(np.count_nonzero(self.cache.image == 0) / self.cache.image.size)]

    def _histogram(self) -> list[Any]:
        if self.per_channel_mode:
            return self.histogram.tolist()
        return [self.histogram.tolist()]

    def get_empty_values(self) -> dict[str, Any]:
        """Return empty values for pixel statistics."""
        return {
            "histogram": [np.nan] * 256,  # Histogram with 256 bins
        }

    def get_handlers(self) -> dict[ImageStats, tuple[str, Callable[[], list[Any]]]]:
        """Return mapping of flags to (stat_name, handler_function)."""
        return {
            ImageStats.PIXEL_MEAN: ("mean", self._mean),
            ImageStats.PIXEL_STD: ("std", self._std),
            ImageStats.PIXEL_VAR: ("var", self._var),
            ImageStats.PIXEL_SKEW: ("skew", self._skew),
            ImageStats.PIXEL_KURTOSIS: ("kurtosis", self._kurtosis),
            ImageStats.PIXEL_ENTROPY: ("entropy", self._entropy),
            ImageStats.PIXEL_MISSING: ("missing", self._missing),
            ImageStats.PIXEL_ZEROS: ("zeros", self._zeros),
            ImageStats.PIXEL_HISTOGRAM: ("histogram", self._histogram),
        }
