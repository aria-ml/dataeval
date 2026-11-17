"""The noise of a view: the standard deviation and variance of additive white Gaussian noise, after Immerkaer."""

__all__ = []

import math
from functools import cached_property
from typing import Any, NamedTuple

import numpy as np

from dataeval.core._calculators._base import ALL_VIEWS, Calculator, Handler, ViewKind
from dataeval.core._calculators._cache import CalculatorCache
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.core._noise import _MIN_NEIGHBORHOODS, noise_immerkaer
from dataeval.flags import ImageStats


class _Reading(NamedTuple):
    """One view's noise variance and how it was read. The variance is NaN unless the status is ``ok``."""

    status: str
    var: float
    neighborhoods: int


@CalculatorRegistry.register(ImageStats)
class NoiseStatCalculator(Calculator[ImageStats]):
    """Calculator for the noise of the image, a box, the background, a band group and a box's surround."""

    def __init__(self, datum: Any, cache: "CalculatorCache") -> None:
        self.datum = datum
        self.cache = cache
        self.warnings: list[str] = []

    def get_applicable_flags(self) -> ImageStats:
        """Return which flags this calculator handles."""
        return ImageStats.NOISE

    def _unmeasured(self) -> _Reading | None:
        """Return the reading of a view that cannot be read, or None where it can, in the order of the statuses."""
        if self.cache.invalid_box:
            return _Reading("invalid_box", math.nan, -1)
        if self.cache.lacks_bands:
            return _Reading("missing_channel", math.nan, -1)
        if self.cache.normalize_pixel_values and self.cache.full_scale is None:
            self.warnings.append(
                "normalize_pixel_values=True and no value range was declared, so the noise statistics are NaN with the "
                "status unknown_range. Only unsigned 8-bit data, and floating-point data within [0, 1] or [0, 255], "
                "have a range without a declaration: pass value_range=(low, high), or declare it in the datum's "
                "metadata."
            )
            return _Reading("unknown_range", math.nan, -1)
        return None

    @cached_property
    def _reading(self) -> _Reading:
        """Read the view once, for both statistics and the details that describe them."""
        if (unmeasured := self._unmeasured()) is not None:
            return unmeasured
        image = self.cache.image
        if image.ndim < 2:
            return _Reading("insufficient_support", math.nan, 0)
        result = noise_immerkaer(image, min_neighborhoods=_MIN_NEIGHBORHOODS)
        # The estimate is affine-equivariant, so normalizing divides the raw reading by the full scale: declared, else
        # from the dtype, else one of the two float conventions, which do read the data's extremes.
        low, high = (self.cache.full_scale if self.cache.normalize_pixel_values else None) or (0.0, 1.0)
        std, support = result["noise_std"] / (high - low), result["support"]
        # A datum can have no channels, which leaves nothing to read.
        neighborhoods = int(support.min()) if support.size else 0
        if not std.size or np.isnan(std).any():
            return _Reading("insufficient_support", math.nan, neighborhoods)
        # ponytail: a standard deviation above about 1.8e19 stores an infinite float32 `noise_var` under `ok`, as
        # PIXEL_VAR does; check finiteness at float32 if data that large is ever measured.
        return _Reading("ok", float(np.mean(std**2)), neighborhoods)

    def _std(self) -> list[float]:
        return [math.sqrt(self._reading.var)]

    def _var(self) -> list[float]:
        return [self._reading.var]

    def details(self, flags: ImageStats, view: ViewKind = ViewKind.WHOLE) -> dict[str, list[Any]]:  # noqa: ARG002
        """Say how the view was read: its status and the fewest valid 3x3 neighborhoods of any of its channels."""
        if not flags & ImageStats.NOISE:
            return {}
        reading = self._reading
        return {"noise_status": [reading.status], "noise_neighborhoods": [reading.neighborhoods]}

    def get_empty_values(self) -> dict[str, Any]:
        """Return the values of the details on a row where no statistic is measured."""
        return {"noise_status": "not_applicable", "noise_neighborhoods": -1}

    def get_handlers(self) -> dict[ImageStats, Handler]:
        """Return mapping of flags to the statistic each produces.

        Defined over every view that has pixels, the surround included: a masked or NaN pixel creates no 3x3
        neighborhood, so a masked region is read from what it keeps, and a band group is a set of channels to read.
        """
        views = ALL_VIEWS | ViewKind.SURROUND
        return {
            ImageStats.PIXEL_NOISE_STD: Handler("noise_std", self._std, views),
            ImageStats.PIXEL_NOISE_VAR: Handler("noise_var", self._var, views),
        }
