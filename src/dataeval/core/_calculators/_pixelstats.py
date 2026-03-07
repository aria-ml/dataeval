__all__ = []

from collections.abc import Callable
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.core._calculators._base import Calculator
from dataeval.core._calculators._cache import CalculatorCache
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.flags import ImageStats


@CalculatorRegistry.register(ImageStats)
class PixelStatCalculator(Calculator[ImageStats]):
    """Calculator for pixel-level statistics."""

    def __init__(self, datum: NDArray[Any], cache: "CalculatorCache", per_channel: bool = False) -> None:
        self.datum = datum
        self.cache = cache
        self.per_channel_mode = per_channel

    @cached_property
    def _has_nan(self) -> bool:
        """Check once whether the scaled data contains any NaN values."""
        if self.per_channel_mode:
            return bool(np.isnan(self.cache.per_channel).any())
        return bool(np.isnan(self.cache.scaled).any())

    def _mean_func(self, data: NDArray[Any], **kw: Any) -> Any:
        """Use fast .mean() when no NaN, fall back to nanmean."""
        return np.nanmean(data, **kw) if self._has_nan else np.mean(data, **kw)

    def _std_func(self, data: NDArray[Any], **kw: Any) -> Any:
        return np.nanstd(data, **kw) if self._has_nan else np.std(data, **kw)

    def _var_func(self, data: NDArray[Any], **kw: Any) -> Any:
        return np.nanvar(data, **kw) if self._has_nan else np.var(data, **kw)

    @cached_property
    def histogram(self) -> NDArray[np.float64]:
        if self.per_channel_mode:
            return np.apply_along_axis(lambda y: np.histogram(y, bins=256, range=(0, 1))[0], 1, self.cache.per_channel)
        return np.histogram(self.cache.scaled, bins=256, range=(0, 1))[0]

    def get_applicable_flags(self) -> ImageStats:
        """Return which flags this calculator handles."""
        return ImageStats.PIXEL

    def _nan_list(self) -> list[float]:
        """Return NaN values matching the expected output shape for all-NaN data."""
        if self.per_channel_mode:
            return [np.nan] * self.cache.image.shape[0]
        return [np.nan]

    def _mean(self) -> list[float]:
        if self.cache.is_all_nan:
            return self._nan_list()
        if self.per_channel_mode:
            return self._mean_func(self.cache.per_channel, axis=1).tolist()
        return [float(self._mean_func(self.cache.scaled))]

    def _std(self) -> list[float]:
        if self.cache.is_all_nan:
            return self._nan_list()
        if self.per_channel_mode:
            return self._std_func(self.cache.per_channel, axis=1).tolist()
        return [float(self._std_func(self.cache.scaled))]

    def _var(self) -> list[float]:
        if self.cache.is_all_nan:
            return self._nan_list()
        if self.per_channel_mode:
            return self._var_func(self.cache.per_channel, axis=1).tolist()
        return [float(self._var_func(self.cache.scaled))]

    @cached_property
    def _moments(self) -> tuple[Any, Any, Any]:
        """Compute variance (m2), 3rd central moment (m3), and 4th central moment (m4).

        Uses fast .mean() when no NaN, caches only scalars (or per-channel arrays),
        not full-image-sized intermediates.
        """
        mean_fn = self._mean_func
        if self.per_channel_mode:
            data = self.cache.per_channel
            d = data - mean_fn(data, axis=1, keepdims=True)
            d2 = d * d
            m2 = mean_fn(d2, axis=1)
            m3 = mean_fn(d2 * d, axis=1)
            np.multiply(d2, d2, out=d2)
            m4 = mean_fn(d2, axis=1)
            return m2, m3, m4
        data = self.cache.scaled.ravel()
        d = data - mean_fn(data)
        d2 = d * d
        m2 = float(mean_fn(d2))
        m3 = float(mean_fn(d2 * d))
        np.multiply(d2, d2, out=d2)
        m4 = float(mean_fn(d2))
        return m2, m3, m4

    def _skew(self) -> list[float]:
        if self.cache.is_all_nan:
            return self._nan_list()
        m2, m3, _ = self._moments
        if self.per_channel_mode:
            s3 = np.float_power(m2, 1.5)
            s3 = np.where(s3 == 0, 1.0, s3)
            return (m3 / s3).tolist()
        if m2 == 0:
            return [0.0]
        return [m3 / (m2**1.5)]

    def _kurtosis(self) -> list[float]:
        if self.cache.is_all_nan:
            return self._nan_list()
        m2, _, m4 = self._moments
        if self.per_channel_mode:
            s4 = m2 * m2
            s4_safe = np.where(s4 == 0, 1.0, s4)
            k = m4 / s4_safe - 3.0
            return np.where(s4 == 0, 0.0, k).tolist()
        if m2 == 0:
            return [0.0]
        return [m4 / (m2 * m2) - 3.0]

    def _entropy(self) -> list[float]:
        if self.per_channel_mode:
            h = self.histogram.astype(np.float64)
            totals = h.sum(axis=1, keepdims=True)
            totals = np.where(totals == 0, 1.0, totals)
            h = h / totals
            with np.errstate(divide="ignore", invalid="ignore"):
                return (-np.nansum(h * np.log(np.where(h > 0, h, 1.0)), axis=1) + 0.0).tolist()
        h = self.histogram.astype(np.float64)
        total = h.sum()
        if total == 0:
            return [0.0]
        h = h / total
        with np.errstate(divide="ignore", invalid="ignore"):
            return [float(-np.nansum(h * np.log(np.where(h > 0, h, 1.0))) + 0.0)]

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
