from __future__ import annotations

__all__ = []

from collections.abc import Callable
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy, kurtosis, skew

from dataeval.config import EPSILON
from dataeval.core._calculators._base import Calculator
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.core.flags import ImageStats
from dataeval.utils._image import edge_filter, get_bitdepth

if TYPE_CHECKING:
    from dataeval.core._calculate import CalculatorCache

QUARTILES = (0, 25, 50, 75, 100)


@CalculatorRegistry.register(ImageStats)
class PixelStatCalculator(Calculator[ImageStats]):
    """Calculator for pixel-level statistics."""

    def __init__(self, datum: NDArray[Any], calculator: CalculatorCache, per_channel: bool = False) -> None:
        self.datum = datum
        self.calculator = calculator
        self.per_channel_mode = per_channel

    @cached_property
    def histogram(self) -> NDArray[np.float64]:
        if self.per_channel_mode:
            return np.apply_along_axis(
                lambda y: np.histogram(y, bins=256, range=(0, 1))[0], 1, self.calculator.per_channel
            )
        return np.histogram(self.calculator.scaled, bins=256, range=(0, 1))[0]

    def get_applicable_flags(self) -> ImageStats:
        """Return which flags this calculator handles."""
        return ImageStats.PIXEL

    def _mean(self) -> list[float]:
        if self.per_channel_mode:
            return np.nanmean(self.calculator.per_channel, axis=1).tolist()
        return [float(np.nanmean(self.calculator.scaled))]

    def _std(self) -> list[float]:
        if self.per_channel_mode:
            return np.nanstd(self.calculator.per_channel, axis=1).tolist()
        return [float(np.nanstd(self.calculator.scaled))]

    def _var(self) -> list[float]:
        if self.per_channel_mode:
            return np.nanvar(self.calculator.per_channel, axis=1).tolist()
        return [float(np.nanvar(self.calculator.scaled))]

    def _skew(self) -> list[float]:
        if self.per_channel_mode:
            return skew(self.calculator.per_channel, axis=1, nan_policy="omit").tolist()
        return [float(skew(self.calculator.scaled.ravel(), nan_policy="omit"))]

    def _kurtosis(self) -> list[float]:
        if self.per_channel_mode:
            return kurtosis(self.calculator.per_channel, axis=1, nan_policy="omit").tolist()
        return [float(kurtosis(self.calculator.scaled.ravel(), nan_policy="omit"))]

    def _entropy(self) -> list[float]:
        if self.per_channel_mode:
            return np.asarray(entropy(self.histogram, axis=1)).tolist()
        return [float(entropy(self.histogram))]

    def _missing(self) -> list[float]:
        if self.per_channel_mode:
            return (
                np.count_nonzero(np.isnan(self.calculator.image), axis=(1, 2)) / np.prod(self.calculator.shape[-2:])
            ).tolist()
        return [float(np.count_nonzero(np.isnan(self.calculator.image)) / np.prod(self.calculator.shape[-2:]))]

    def _zeros(self) -> list[float]:
        if self.per_channel_mode:
            return (
                np.count_nonzero(self.calculator.image == 0, axis=(1, 2)) / np.prod(self.calculator.shape[-2:])
            ).tolist()
        return [
            float(np.count_nonzero(np.sum(self.calculator.image, axis=0) == 0) / np.prod(self.calculator.shape[-2:]))
        ]

    def _histogram(self) -> list[Any]:
        if self.per_channel_mode:
            return self.histogram.tolist()
        return [self.histogram.tolist()]

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


@CalculatorRegistry.register(ImageStats)
class VisualStatCalculator(Calculator[ImageStats]):
    """Calculator for visual statistics like brightness, contrast, sharpness."""

    def __init__(self, datum: NDArray[Any], calculator: CalculatorCache, per_channel: bool = False) -> None:
        self.datum = datum
        self.calculator = calculator
        self.per_channel_mode = per_channel

    @cached_property
    def percentiles(self) -> NDArray[np.float64]:
        if self.per_channel_mode:
            return np.nanpercentile(self.calculator.per_channel, q=QUARTILES, axis=1).T.astype(np.float64)
        return np.nanpercentile(self.calculator.scaled, q=QUARTILES).astype(np.float64)

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
        if self.per_channel_mode:
            return np.nanstd(
                np.vectorize(edge_filter, signature="(m,n)->(m,n)")(self.calculator.image), axis=(1, 2)
            ).tolist()
        return [float(np.nanstd(edge_filter(np.mean(self.calculator.image, axis=0))))]

    def _percentiles(self) -> list[Any]:
        if self.per_channel_mode:
            return self.percentiles.tolist()
        return [self.percentiles.tolist()]

    def get_handlers(self) -> dict[ImageStats, tuple[str, Callable[[], list[Any]]]]:
        """Return mapping of flags to (stat_name, handler_function)."""
        return {
            ImageStats.VISUAL_BRIGHTNESS: ("brightness", self._brightness),
            ImageStats.VISUAL_CONTRAST: ("contrast", self._contrast),
            ImageStats.VISUAL_DARKNESS: ("darkness", self._darkness),
            ImageStats.VISUAL_SHARPNESS: ("sharpness", self._sharpness),
            ImageStats.VISUAL_PERCENTILES: ("percentiles", self._percentiles),
        }


@CalculatorRegistry.register(ImageStats)
class DimensionStatCalculator(Calculator[ImageStats]):
    """Calculator for dimension and geometry statistics."""

    def __init__(self, datum: NDArray[Any], calculator: CalculatorCache, per_channel: bool = False) -> None:
        self.datum = datum
        self.calculator = calculator

    def get_applicable_flags(self) -> ImageStats:
        """Return which flags this calculator handles."""
        return ImageStats.DIMENSION

    def _offset_x(self) -> list[float]:
        return [self.calculator.box.x0]

    def _offset_y(self) -> list[float]:
        return [self.calculator.box.y0]

    def _width(self) -> list[float]:
        return [self.calculator.box.width]

    def _height(self) -> list[float]:
        return [self.calculator.box.height]

    def _channels(self) -> list[int]:
        return [self.calculator.shape[-3]]

    def _size(self) -> list[float]:
        return [self.calculator.box.width * self.calculator.box.height]

    def _aspect_ratio(self) -> list[float]:
        box = self.calculator.box
        return [0.0 if box.height == 0 else box.width / box.height]

    def _depth(self) -> list[int]:
        return [get_bitdepth(self.calculator.raw).depth]

    def _center(self) -> list[list[float]]:
        box = self.calculator.box
        return [[(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2]]

    def _distance_center(self) -> list[float]:
        box = self.calculator.box
        raw = self.calculator.raw
        return [
            float(
                np.sqrt(
                    np.square(((box.x0 + box.x1) / 2) - (raw.shape[-1] / 2))
                    + np.square(((box.y0 + box.y1) / 2) - (raw.shape[-2] / 2))
                )
            )
        ]

    def _distance_edge(self) -> list[float]:
        box = self.calculator.box
        raw = self.calculator.raw
        return [
            float(
                np.min(
                    [
                        np.abs(box.x0),
                        np.abs(box.y0),
                        np.abs(box.x1 - raw.shape[-1]),
                        np.abs(box.y1 - raw.shape[-2]),
                    ]
                )
            )
        ]

    def _invalid_box(self) -> list[bool]:
        return [not self.calculator.box.is_valid()]

    def get_handlers(self) -> dict[ImageStats, tuple[str, Callable[[], list[Any]]]]:
        """Return mapping of flags to (stat_name, handler_function)."""
        return {
            ImageStats.DIMENSION_OFFSET_X: ("offset_x", self._offset_x),
            ImageStats.DIMENSION_OFFSET_Y: ("offset_y", self._offset_y),
            ImageStats.DIMENSION_WIDTH: ("width", self._width),
            ImageStats.DIMENSION_HEIGHT: ("height", self._height),
            ImageStats.DIMENSION_CHANNELS: ("channels", self._channels),
            ImageStats.DIMENSION_SIZE: ("size", self._size),
            ImageStats.DIMENSION_ASPECT_RATIO: ("aspect_ratio", self._aspect_ratio),
            ImageStats.DIMENSION_DEPTH: ("depth", self._depth),
            ImageStats.DIMENSION_CENTER: ("center", self._center),
            ImageStats.DIMENSION_DISTANCE_CENTER: ("distance_center", self._distance_center),
            ImageStats.DIMENSION_DISTANCE_EDGE: ("distance_edge", self._distance_edge),
            ImageStats.DIMENSION_INVALID_BOX: ("invalid_box", self._invalid_box),
        }


@CalculatorRegistry.register(ImageStats)
class HashStatCalculator(Calculator):
    """Calculator for hash-based statistics."""

    def __init__(self, datum: NDArray[Any], calculator: CalculatorCache, per_channel: bool = False) -> None:
        self.datum = datum
        self.calculator = calculator

    def get_applicable_flags(self) -> ImageStats:
        """Return which flags this calculator handles."""
        return ImageStats.HASH

    def _xxhash(self) -> list[str]:
        from dataeval.core._hash import xxhash

        return [xxhash(self.calculator.raw)]

    def _pchash(self) -> list[str]:
        from dataeval.core._hash import pchash

        return [pchash(self.calculator.raw)]

    def get_handlers(self) -> dict[ImageStats, tuple[str, Callable[[], list[Any]]]]:
        """Return mapping of flags to (stat_name, handler_function)."""
        return {
            ImageStats.HASH_XXHASH: ("xxhash", self._xxhash),
            ImageStats.HASH_PCHASH: ("pchash", self._pchash),
        }
