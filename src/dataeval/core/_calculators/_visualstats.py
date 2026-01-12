__all__ = []

from collections.abc import Callable
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from dataeval.config import EPSILON
from dataeval.core._calculators._base import Calculator
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.flags import ImageStats
from dataeval.utils.preprocessing import edge_filter

if TYPE_CHECKING:
    from dataeval.core._calculate import CalculatorCache

QUARTILES = (0, 25, 50, 75, 100)


@CalculatorRegistry.register(ImageStats)
class VisualStatCalculator(Calculator[ImageStats]):
    """Calculator for visual statistics like brightness, contrast, sharpness."""

    def __init__(self, datum: NDArray[Any], cache: "CalculatorCache", per_channel: bool = False) -> None:
        self.datum = datum
        self.cache = cache
        self.per_channel_mode = per_channel

    @cached_property
    def percentiles(self) -> NDArray[np.float64]:
        if self.per_channel_mode:
            return np.nanpercentile(self.cache.per_channel, q=QUARTILES, axis=1).T.astype(np.float64)
        return np.nanpercentile(self.cache.scaled, q=QUARTILES).astype(np.float64)

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
        # Sharpness requires 2D spatial data; return NaN for low-dimensional data
        if self.cache.image.ndim < 2:
            return [np.nan] if not self.per_channel_mode else [np.nan] * self.cache.image.shape[0]
        if self.cache.image.ndim == 2:
            # 2D data: treat as single-channel image
            return [float(np.nanstd(edge_filter(self.cache.image)))]
        # 3D+ data with channels
        if self.per_channel_mode:
            return np.nanstd(
                np.vectorize(edge_filter, signature="(m,n)->(m,n)")(self.cache.image), axis=(1, 2)
            ).tolist()
        return [float(np.nanstd(edge_filter(np.mean(self.cache.image, axis=0))))]

    def _percentiles(self) -> list[Any]:
        if self.per_channel_mode:
            return self.percentiles.tolist()
        return [self.percentiles.tolist()]

    def get_empty_values(self) -> dict[str, Any]:
        """Return empty values for visual statistics."""
        return {
            "percentiles": [np.nan] * 5,  # 5 percentiles: 0, 25, 50, 75, 100
        }

    def get_handlers(self) -> dict[ImageStats, tuple[str, Callable[[], list[Any]]]]:
        """Return mapping of flags to (stat_name, handler_function)."""
        return {
            ImageStats.VISUAL_BRIGHTNESS: ("brightness", self._brightness),
            ImageStats.VISUAL_CONTRAST: ("contrast", self._contrast),
            ImageStats.VISUAL_DARKNESS: ("darkness", self._darkness),
            ImageStats.VISUAL_SHARPNESS: ("sharpness", self._sharpness),
            ImageStats.VISUAL_PERCENTILES: ("percentiles", self._percentiles),
        }
