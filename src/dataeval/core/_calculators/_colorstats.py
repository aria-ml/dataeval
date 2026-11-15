"""The color of a view: lightness, chroma, achromatic share and hue direction of three declared sRGB channels."""

__all__ = []

import math
from functools import cached_property
from typing import Any, NamedTuple

import numpy as np

from dataeval.core._calculators._base import Calculator, Handler, ViewKind
from dataeval.core._calculators._cache import CalculatorCache
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.core._color import ColorSums, block_sums
from dataeval.flags import ImageStats
from dataeval.types import MeasurementSpec
from dataeval.utils.preprocessing import normalize_image_shape

_OK = "ok"
_MISSING_BAND = "missing_band"
_UNKNOWN_RANGE = "unknown_range"
_INSUFFICIENT_SUPPORT = "insufficient_support"
_ACHROMATIC = "achromatic"
_ZERO = ColorSums(0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0)
# The window is read in blocks of about this many pixels, so the calculator's own temporaries do not grow with the
# image.
# ponytail: a constant and not a setting; make it one if a caller ever needs a smaller peak.
_BLOCK_PIXELS = 1 << 20
_UNITS = {
    "color_lightness_mean": "L*",
    "color_chroma_mean": "C*",
    "color_achromatic_fraction": "fraction",
    "color_hue_cos": "unit vector component",
    "color_hue_sin": "unit vector component",
}


class _Reading(NamedTuple):
    """One view's color sums and how it was measured. A count is -1 and a fraction is NaN where unknown."""

    status: str
    sums: ColorSums
    valid_fraction: float
    valid_pixels: int
    excluded_pixels: int


def _none(status: str, excluded: int) -> _Reading:
    """Return a view with nothing read: the status alone, and how many pixels the mask removes, which geometry gives."""
    return _Reading(status, _ZERO, math.nan, -1, excluded)


def _sums_over(
    window: Any, channels: tuple[int, int, int], interval: tuple[float, float], threshold: float
) -> ColorSums:
    """Sum the color of the valid pixels of ``(C, H, W)`` `window`, in row blocks that add to the whole."""
    low, high = interval
    height, width = window.shape[-2:]
    rows = max(1, _BLOCK_PIXELS // max(width, 1))
    sums = _ZERO
    with np.errstate(invalid="ignore", over="ignore"):
        for start in range(0, height, rows):
            block = window[list(channels), start : start + rows].astype(np.float64)
            # NaN fails both comparisons, so it is not valid; a value outside the range is not valid either, and is
            # never clipped.
            valid = ((block >= low) & (block <= high)).all(axis=0)
            if valid.any():
                sums = sums + block_sums((block[:, valid].T - low) / (high - low), threshold)
    return sums


def _interval(cache: CalculatorCache) -> tuple[float, float] | None:
    """Return the interval the view's values are mapped from to ``[0, 1]``, or None where there is none to use.

    The declared one: the call's or the datum's own metadata, and not a channel group's. Without a declaration only an
    unsigned 8-bit datum has one, ``(0, 255)`` by its dtype, whatever its maximum. The shared decode is not used: it
    reads the data's own maximum, so the same pixels would read as different colors.
    """
    if cache.has_declared_range:
        return float(cache.value_range.pmin), float(cache.value_range.pmax)
    if getattr(cache.raw, "dtype", None) == np.uint8:
        return 0.0, 255.0
    return None


@CalculatorRegistry.register(ImageStats)
class ColorStatCalculator(Calculator[ImageStats]):
    """Calculator for the color of the image, of a box and of the background, from three declared sRGB channels."""

    def __init__(self, datum: Any, cache: "CalculatorCache") -> None:
        self.datum = datum
        self.cache = cache
        self.warnings: list[str] = []

    def get_applicable_flags(self) -> ImageStats:
        """Return which flags this calculator handles."""
        return ImageStats.COLOR

    def _unmeasured(self, window: Any, excluded: int) -> _Reading | None:
        """Return the reading of a view that cannot be measured, or None where it can, in the order of the statuses."""
        channels = self.cache.settings.color.rgb_channels
        # The window is 2-D for a datum with no box and no mask, so channels come from the normalized shape.
        if channels is None or window.ndim < 2 or max(channels) >= normalize_image_shape(window).shape[0]:
            self.warnings.append(
                "the datum does not have the channels color_rgb_channels names, so the color statistics are NaN with "
                "the status missing_band. Name the channels this datum has, or drop the datum from the call."
            )
            return _none(_MISSING_BAND, excluded)
        if _interval(self.cache) is None:
            self.warnings.append(
                "no value range was declared, so the color statistics are NaN with the status unknown_range. Only an "
                "unsigned 8-bit datum has a range without a declaration: pass value_range=(low, high), or declare "
                "it in the datum's metadata."
            )
            return _none(_UNKNOWN_RANGE, excluded)
        return None

    @cached_property
    def _reading(self) -> _Reading:
        """Read the view once, for every statistic and the details that describe them."""
        settings, mask = self.cache.settings.color, self.cache.window_mask
        excluded = 0 if mask is None else int(np.count_nonzero(mask))
        image = self.cache.image
        if (unmeasured := self._unmeasured(image, excluded)) is not None:
            return unmeasured
        window = normalize_image_shape(image)
        sums = _sums_over(
            window,
            settings.rgb_channels or (0, 1, 2),
            _interval(self.cache) or (0.0, 1.0),
            settings.chroma_threshold,
        )
        # The pixels the view can measure: those of its window inside the datum, less what the mask removes.
        pixels = self.cache.overlap_pixels - excluded
        fraction = sums.pixels / pixels if pixels > 0 else math.nan
        short = sums.pixels < settings.min_pixels or not fraction >= settings.min_valid_fraction
        return _Reading(_INSUFFICIENT_SUPPORT if short else _OK, sums, fraction, sums.pixels, excluded)

    @classmethod
    def measurement_specs(
        cls,
        flags: ImageStats,
        view: ViewKind,
        settings: Any,
        declared_range: tuple[float, float] | None,
    ) -> dict[str, MeasurementSpec]:
        """Record the algorithm, the declared channels, the settings and the range the call or a group declared.

        Without a declared range the interval is each datum's own, its metadata's or 255 for an unsigned 8-bit one,
        and the spec reads the same for all of them, as it does for saturation: results of an 8-bit and a declared
        16-bit collection still combine when their other settings agree.
        """
        selected = flags & ImageStats.COLOR
        if not selected:
            return {}
        color = settings.color
        policy = ("declared", declared_range) if declared_range is not None else ("inferred", None)
        config = {
            "encoding": "srgb",
            "min_pixels": color.min_pixels,
            "min_valid_fraction": color.min_valid_fraction,
            "chroma_threshold": color.chroma_threshold,
        }
        return {
            handler.name: MeasurementSpec(
                algorithm="cielab_srgb_d65",
                version="1",
                units=_UNITS[handler.name],
                region="background" if ViewKind.MASK in view else "pixels",
                normalization="[0, 1]",
                config=config,
                range_policy=policy[0],  # type: ignore[arg-type]
                declared_range=policy[1],
                rgb_channels=color.rgb_channels,
            )
            for handler in cls._handlers_for(selected).values()
        }

    def _statistic(self, pick: Any) -> list[float]:
        reading = self._reading
        return [pick(reading.sums) if reading.status == _OK else math.nan]

    def _lightness(self) -> list[float]:
        return self._statistic(lambda s: s.lightness / s.pixels)

    def _chroma(self) -> list[float]:
        return self._statistic(lambda s: s.chroma / s.pixels)

    def _achromatic(self) -> list[float]:
        return self._statistic(lambda s: s.achromatic / s.pixels)

    def _hue_cos(self) -> list[float]:
        return self._statistic(lambda s: s.a_chromatic / s.chroma_chromatic if s.chromatic else math.nan)

    def _hue_sin(self) -> list[float]:
        return self._statistic(lambda s: s.b_chromatic / s.chroma_chromatic if s.chromatic else math.nan)

    def details(self, flags: ImageStats, view: ViewKind = ViewKind.WHOLE) -> dict[str, list[Any]]:  # noqa: ARG002
        """Say how the view was measured: its status, the valid pixels and the share they are of the window."""
        if not flags & ImageStats.COLOR:
            return {}
        reading = self._reading
        achromatic = reading.status == _OK and reading.sums.chromatic == 0
        return {
            "color_status": [reading.status],
            "color_valid_fraction": [reading.valid_fraction],
            "color_valid_pixels": [reading.valid_pixels],
            "color_excluded_pixels": [reading.excluded_pixels],
            "color_hue_reason": [_ACHROMATIC if achromatic else ""],
        }

    def get_empty_values(self) -> dict[str, Any]:
        """Return the values of the details on a row where no statistic is measured."""
        return {"color_status": "", "color_valid_pixels": -1, "color_excluded_pixels": -1, "color_hue_reason": ""}

    def get_handlers(self) -> dict[ImageStats, Handler]:
        """Return mapping of flags to the statistic each produces.

        Defined over the whole datum and over a masked region, as the issue lists the image, a box and the background,
        and not over a band group: `color_rgb_channels` indexes the datum's own channels.
        """
        views = ViewKind.WHOLE | ViewKind.MASK
        return {
            ImageStats.VISUAL_COLOR_LIGHTNESS_MEAN: Handler("color_lightness_mean", self._lightness, views),
            ImageStats.VISUAL_COLOR_CHROMA_MEAN: Handler("color_chroma_mean", self._chroma, views),
            ImageStats.VISUAL_COLOR_ACHROMATIC_FRACTION: Handler("color_achromatic_fraction", self._achromatic, views),
            ImageStats.VISUAL_COLOR_HUE_COS: Handler("color_hue_cos", self._hue_cos, views),
            ImageStats.VISUAL_COLOR_HUE_SIN: Handler("color_hue_sin", self._hue_sin, views),
        }
