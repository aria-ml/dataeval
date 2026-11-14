"""The surround of a box: the pixels in a widened frame around it, less the box and less every other valid box."""

__all__ = []

import math
from collections.abc import Sequence
from functools import cached_property
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from dataeval.core._calculators._base import Calculator, ContextNeed, Handler, ViewKind
from dataeval.core._calculators._cache import CalculatorCache
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.flags import ImageStats
from dataeval.types import MeasurementSpec
from dataeval.utils._annotations import AnnotationContext
from dataeval.utils.preprocessing import ValueRange, normalize_image_shape

IntBox = tuple[int, int, int, int]

_NOT_APPLICABLE = "not_applicable"
_INVALID_BOX = "invalid_box"
_UNKNOWN_RANGE = "unknown_range"
_EMPTY_RING = "empty_ring"
_INSUFFICIENT_SUPPORT = "insufficient_support"
_INCOMPLETE_GEOMETRY = "incomplete_geometry"
_OK = "ok"
# Why a ratio is NaN while its surround is measured: the target, whose pixels it reads, has too few valid ones.
_INSUFFICIENT_TARGET_SUPPORT = "insufficient_target_support"
# The statuses that carry a value: every other one is NaN.
_MEASURED = (_OK, _INCOMPLETE_GEOMETRY)
_RATIOS = ImageStats.CONTEXT_TARGET_SURROUND_CONTRAST | ImageStats.CONTEXT_SIGNAL_CLUTTER_RATIO


class Regions(NamedTuple):
    """The two pixel rectangles of one box, each ``(x0, y0, x1, y1)`` rounded outward and clipped to the image.

    Attributes
    ----------
    window : tuple[int, int, int, int]
        The box widened by the padding on every side. It contains `target`.
    target : tuple[int, int, int, int]
        The box itself.
    """

    window: IntBox
    target: IntBox


class Reading(NamedTuple):
    """What the surround's pixels hold.

    Attributes
    ----------
    measurable : int
        Retained surround pixels at which every selected channel is finite and within the range.
    mean : float
        The mean intensity of those pixels, in ``[0, 1]``, or NaN when there are none.
    std : float
        Their population (``ddof=0``) standard deviation, or NaN when there are none.
    target_valid_fraction : float
        The share of the target's pixels at which every selected channel is finite and within the range.
    target_mean : float
        The mean intensity of those target pixels, in ``[0, 1]``, or NaN when there are none.
    """

    measurable: int
    mean: float
    std: float
    target_valid_fraction: float
    target_mean: float


def _rounded(box: tuple[float, float, float, float], extent: tuple[int, int]) -> IntBox:
    """Clip a float box to the image, then round it outward: ``floor`` the low edges and ``ceil`` the high ones.

    Clipping comes first so that no input reaches ``floor`` or ``ceil`` as infinity: both raise on one. The result is
    the same as rounding first, since the image's edges are whole numbers.
    """
    height, width = extent
    x0, y0, x1, y1 = box
    return (
        math.floor(min(max(x0, 0.0), width)),
        math.floor(min(max(y0, 0.0), height)),
        math.ceil(min(max(x1, 0.0), width)),
        math.ceil(min(max(y1, 0.0), height)),
    )


def surround_regions(row: NDArray[np.float64], extent: tuple[int, int], padding: float) -> Regions:
    """
    Return a box's window and target rectangles.

    Parameters
    ----------
    row : NDArray[np.float64]
        A ``VALID`` box's coordinates as given, ``(x0, y0, x1, y1)``. Its width and height are the box's own, not the
        clipped ones, so the padding does not shrink where the box leaves the image.
    extent : tuple[int, int]
        The image's ``(H, W)``.
    padding : float
        The share of the box's width and of its height added on every side.

    Returns
    -------
    Regions
        The window ``[x0 - padding * w, y0 - padding * h, x1 + padding * w, y1 + padding * h]`` and the box, each
        rounded outward to whole pixels as :func:`~dataeval.utils.preprocessing.boxes_to_mask` rounds, and clipped.
    """
    x0, y0, x1, y1 = (float(v) for v in row)
    # Python floats, so a width that overflows is infinity and not a warning; `_rounded` clips it.
    pad_x, pad_y = padding * (x1 - x0), padding * (y1 - y0)
    return Regions(
        _rounded((x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y), extent),
        _rounded((x0, y0, x1, y1), extent),
    )


def read_surround(
    image: NDArray[Any],
    channels: Sequence[int],
    regions: Regions,
    retained: NDArray[np.bool_],
    low: float,
    high: float,
) -> Reading:
    """
    Read the intensity of a box's surround.

    Parameters
    ----------
    image : NDArray[Any]
        The whole image, ``(C, H, W)``.
    channels : Sequence[int]
        The channels to read, which the intensity is averaged over. Each must be within `image`.
    regions : Regions
        The box's window and target.
    retained : NDArray[np.bool_]
        Shaped like the window: True where a pixel is in the surround, that is, no box of the image covers it.
    low, high : float
        The interval the values are mapped from to ``[0, 1]``. ``low < high``, both finite.

    Returns
    -------
    Reading
        The count, mean and standard deviation of the measurable surround pixels, and the target's valid fraction and
        mean. A pixel is measurable only if every channel read is within ``[low, high]``, and a value outside is never
        clipped, so NaN and infinity, which no finite interval holds, are excluded by the same test. The target's mean
        is over the pixels that pass it too.
    """
    x0, y0, x1, y1 = regions.window
    tx0, ty0, tx1, ty1 = regions.target
    measurable = retained.copy()
    target_valid = np.ones((ty1 - ty0, tx1 - tx0), dtype=np.bool_)
    total = np.zeros(retained.shape, dtype=np.float64)
    channels = tuple(channels)
    scale = high - low
    # ponytail: one pass per channel over the window, so time and memory grow with the window and not the band count; an
    # integral image per channel if boxes near the frame's size ever dominate.
    with np.errstate(invalid="ignore", over="ignore"):
        for channel in channels:
            values = image[channel, y0:y1, x0:x1].astype(np.float64)
            measurable &= (values >= low) & (values <= high)
            total += (values - low) / scale
            inside = image[channel, ty0:ty1, tx0:tx1].astype(np.float64)
            target_valid &= (inside >= low) & (inside <= high)
    intensity = total[measurable] / len(channels)
    valid = int(np.count_nonzero(target_valid))
    fraction = valid / target_valid.size
    # `total` holds each pixel's intensity summed over the channels, the target's among them, since the window contains
    # the target, so its mean is read from it and costs no second mapping. A pixel that is not valid is left out.
    target_mean = math.nan
    if valid:
        target_total = total[ty0 - y0 : ty1 - y0, tx0 - x0 : tx1 - x0]
        # Where every pixel is valid, which is the usual case, the plain sum is about twice as fast as the masked one.
        target_sum = target_total.sum() if valid == target_valid.size else np.sum(target_total, where=target_valid)
        target_mean = float(target_sum) / (valid * len(channels))
    if intensity.size == 0:
        return Reading(0, math.nan, math.nan, fraction, target_mean)
    mean = float(intensity.mean())
    # Two passes, so a constant surround reads a spread of 0 to float64 precision and not a cancellation's residue.
    return Reading(int(intensity.size), mean, float(np.sqrt(np.mean((intensity - mean) ** 2))), fraction, target_mean)


class _Surround(NamedTuple):
    """One row's reading and how it was made. A count is -1 and a fraction is NaN where it does not apply."""

    status: str
    mean: float
    std: float
    candidate: int
    retained: int
    excluded: float
    measurable: int
    surround_valid: float
    target_valid: float
    target_mean: float
    low: float
    high: float


class _Ratios(NamedTuple):
    """One row's two ratios, whether each rests on its floor, and why they are NaN where the target is the cause."""

    contrast: float
    clutter: float
    contrast_floored: bool
    clutter_floored: bool
    reason: str


_NO_RATIOS = _Ratios(math.nan, math.nan, False, False, "")


class _Geometry(NamedTuple):
    """Where a box's surround is, and how much of it other boxes leave."""

    regions: Regions
    retained_mask: NDArray[np.bool_]
    candidate: int
    retained: int


def _none(status: str, low: float = math.nan, high: float = math.nan) -> _Surround:
    """Return a row with nothing counted: the status alone, and the interval where the view has one."""
    return _Surround(status, math.nan, math.nan, -1, -1, math.nan, -1, math.nan, math.nan, math.nan, low, high)


def _interval(value_range: ValueRange) -> tuple[float, float]:
    """Return the view's interval, or NaN for both where none could be established or its span overflows.

    A span of infinity would map every pixel to 0 or to NaN, which a status that claims a value must not carry.
    """
    low, high = float(value_range.pmin), float(value_range.pmax)
    return (low, high) if value_range.is_known and math.isfinite(high - low) else (math.nan, math.nan)


def _status(frame: str, candidate: int, measurable: int, min_pixels: int) -> str:
    """Name the row's status given the frame's: an empty candidate, then too little support, else the frame's."""
    if candidate == 0:
        return _EMPTY_RING
    return _INSUFFICIENT_SUPPORT if measurable < min_pixels else frame


@CalculatorRegistry.register(ImageStats)
class ContextStatCalculator(Calculator[ImageStats]):
    """Calculator for statistics of the pixels around a box, read against the other boxes of its image."""

    def __init__(self, datum: NDArray[Any], cache: "CalculatorCache") -> None:
        self.datum = datum
        self.cache = cache

    def get_applicable_flags(self) -> ImageStats:
        """Return which flags this calculator handles."""
        return ImageStats.CONTEXT

    def _focal(self) -> tuple[AnnotationContext, int] | None:
        """Return the context and the focal box, or None on a row with none: the image's own, or no spatial extent."""
        context, focal = self.cache.context, self.cache.focal
        if context is None or focal is None or len(self.cache.shape) < 2:
            return None
        return context, focal

    def _geometry(self, context: AnnotationContext, focal: int) -> _Geometry:
        """Place the focal box's window and count what the other boxes leave of it."""
        regions = surround_regions(context.raw[focal], context.extent, self.cache.settings.surround.padding)
        wx0, wy0, wx1, wy1 = regions.window
        tx0, ty0, tx1, ty1 = regions.target
        candidate = (wx1 - wx0) * (wy1 - wy0) - (tx1 - tx0) * (ty1 - ty0)
        # The focal box is in the coverage, so what is left of the window is the candidate less the other boxes.
        retained_mask = ~context.coverage[wy0:wy1, wx0:wx1]
        return _Geometry(regions, retained_mask, candidate, int(np.count_nonzero(retained_mask)))

    def _lacks_bands(self) -> bool:
        """Whether this view is a band group that names a band the datum does not carry."""
        bands = self.cache.bands
        return bands is not None and max(bands) >= normalize_image_shape(self.cache.raw).shape[0]

    def _unreadable(self, low: float) -> bool:
        """Whether the view has no interval to read against. A missing band comes first, as nothing was read for it."""
        return math.isnan(low) and not self._lacks_bands()

    def _read(self, geometry: _Geometry, low: float, high: float) -> Reading:
        """Read the surround's pixels over the channels this view selects."""
        if self._lacks_bands():
            # A group the datum cannot supply is measured all-or-nothing, as every other statistic does.
            return Reading(0, math.nan, math.nan, 0.0, math.nan)
        image = normalize_image_shape(self.cache.raw)
        bands = self.cache.bands
        channels = range(image.shape[0]) if bands is None else bands
        return read_surround(image, channels, geometry.regions, geometry.retained_mask, low, high)

    @cached_property
    def _surround(self) -> _Surround:
        """Read the row once, for both statistics and the details that describe them."""
        at = self._focal()
        if at is None:
            return _none(_NOT_APPLICABLE)
        context, focal = at
        low, high = _interval(self.cache.value_range)
        frame = context.status(focal)
        if frame == _INVALID_BOX:
            return _none(_INVALID_BOX, low, high)
        geometry = self._geometry(context, focal)
        excluded = (geometry.candidate - geometry.retained) / geometry.candidate if geometry.candidate else math.nan
        if self._unreadable(low):
            # The counts come from the geometry alone, so they are still true; nothing can be read against no interval.
            return _none(_UNKNOWN_RANGE)._replace(
                candidate=geometry.candidate, retained=geometry.retained, excluded=excluded
            )
        reading = self._read(geometry, low, high)
        status = _status(frame, geometry.candidate, reading.measurable, self.cache.settings.surround.min_pixels)
        measured = status in _MEASURED
        return _Surround(
            status,
            reading.mean if measured else math.nan,
            reading.std if measured else math.nan,
            geometry.candidate,
            geometry.retained,
            excluded,
            reading.measurable,
            reading.measurable / geometry.retained if geometry.retained else math.nan,
            reading.target_valid_fraction,
            reading.target_mean,
            low,
            high,
        )

    @cached_property
    def _ratios(self) -> _Ratios:
        """Read the two ratios from the surround and the target, once for both statistics and the details.

        The surround's status comes first: a ratio is NaN, with no reason, wherever the surround is not measured, since
        that is not the target's doing and `context_status` says what it is. Then the target must have enough valid
        pixels. A denominator under the floor is replaced by it and recorded, so that a ratio bounded by ``1 / floor``
        is not read as a measurement of the spread or the level alone.
        """
        row = self._surround
        if row.status not in _MEASURED:
            return _NO_RATIOS
        settings = self.cache.settings.surround
        if not row.target_valid >= settings.min_target_valid_fraction or math.isnan(row.target_mean):
            return _Ratios(math.nan, math.nan, False, False, _INSUFFICIENT_TARGET_SUPPORT)
        difference, floor = row.target_mean - row.mean, settings.intensity_floor
        return _Ratios(
            difference / max(row.mean, floor),
            difference / max(row.std, floor),
            row.mean < floor,
            row.std < floor,
            "",
        )

    @classmethod
    def measurement_specs(
        cls,
        flags: ImageStats,
        view: ViewKind,  # noqa: ARG003
        settings: Any,
        declared_range: tuple[float, float] | None,
    ) -> dict[str, MeasurementSpec]:
        """Record each statistic's settings, and the range the call or a channel group declared.

        The surround statistics record the padding and the support floor. A ratio also records the share of valid target
        pixels it needs and the floor under its denominator, which decide its value, so results of a ratio measured with
        different ones refuse to combine and results of the surround statistics still combine with older ones.

        Without a declared range each image's interval is its own metadata's, or what its dtype decodes to, and the spec
        reads the same for all of them, as it does for saturation: the interval each row used is in its details, which
        are not compared when results are combined.
        """
        selected = flags & ImageStats.CONTEXT
        if not selected:
            return {}
        surround = settings.surround
        policy = {
            "range_policy": "inferred" if declared_range is None else "declared",
            "declared_range": declared_range,
        }
        surround_spec = MeasurementSpec(
            algorithm="surround_intensity",
            version="1",
            units="normalized intensity",
            region="surround",
            normalization="[0, 1]",
            config={"padding": surround.padding, "min_pixels": surround.min_pixels},
            **policy,
        )
        ratio_config = {
            "padding": surround.padding,
            "min_pixels": surround.min_pixels,
            "min_target_valid_fraction": surround.min_target_valid_fraction,
            "intensity_floor": surround.intensity_floor,
        }
        specs: dict[str, MeasurementSpec] = {}
        for handler in cls._handlers_for(selected).values():
            if handler.name in ("surround_mean", "surround_std"):
                specs[handler.name] = surround_spec
            else:
                specs[handler.name] = MeasurementSpec(
                    algorithm=handler.name,
                    version="1",
                    units="ratio",
                    region="target and surround",
                    normalization="[0, 1]",
                    config=ratio_config,
                    **policy,
                )
        return specs

    def _mean(self) -> list[float]:
        return [self._surround.mean]

    def _std(self) -> list[float]:
        return [self._surround.std]

    def _contrast(self) -> list[float]:
        return [self._ratios.contrast]

    def _clutter(self) -> list[float]:
        return [self._ratios.clutter]

    def details(self, flags: ImageStats, view: ViewKind = ViewKind.WHOLE) -> dict[str, list[Any]]:
        """Return how the row was measured: its status, the pixel counts and fractions, and the interval.

        A request for a ratio adds the reason a ratio is NaN for want of target pixels, and whether each ratio selected
        rests on its floor. A request for neither returns the details it always did.
        """
        row = self._surround
        details: dict[str, list[Any]] = {"context_status": [row.status]}
        # The geometry is the same in every band group, so only the unnamed view reports it.
        if ViewKind.BAND not in view:
            details["surround_candidate_pixels"] = [row.candidate]
            details["surround_retained_pixels"] = [row.retained]
            details["ring_peer_excluded_fraction"] = [row.excluded]
        details["surround_measurable_pixels"] = [row.measurable]
        details["surround_valid_fraction"] = [row.surround_valid]
        details["target_valid_fraction"] = [row.target_valid]
        details["context_range_low"] = [row.low]
        details["context_range_high"] = [row.high]
        if flags & _RATIOS:
            ratios = self._ratios
            details["contrast_reason"] = [ratios.reason]
            if flags & ImageStats.CONTEXT_TARGET_SURROUND_CONTRAST:
                details["target_surround_contrast_floored"] = [ratios.contrast_floored]
            if flags & ImageStats.CONTEXT_SIGNAL_CLUTTER_RATIO:
                details["signal_clutter_ratio_floored"] = [ratios.clutter_floored]
        return details

    def get_empty_values(self) -> dict[str, Any]:
        """Return the values of the details on a row where no statistic is measured."""
        return {
            "context_status": _NOT_APPLICABLE,
            "surround_candidate_pixels": -1,
            "surround_retained_pixels": -1,
            "surround_measurable_pixels": -1,
            "contrast_reason": "",
            "target_surround_contrast_floored": False,
            "signal_clutter_ratio_floored": False,
        }

    def get_handlers(self) -> dict[ImageStats, Handler]:
        """Return mapping of flags to the statistic each produces.

        Defined over the whole image and over a band group, and not over a masked region: the background columns
        ride on the image's own row, which has no box. Every statistic reads the image's other boxes, so `image_stats`
        builds the annotation context.
        """
        views = ViewKind.WHOLE | ViewKind.BAND
        return {
            ImageStats.CONTEXT_SURROUND_MEAN: Handler("surround_mean", self._mean, views, ContextNeed.CONTEXT),
            ImageStats.CONTEXT_SURROUND_STD: Handler("surround_std", self._std, views, ContextNeed.CONTEXT),
            ImageStats.CONTEXT_TARGET_SURROUND_CONTRAST: Handler(
                "target_surround_contrast", self._contrast, views, ContextNeed.CONTEXT
            ),
            ImageStats.CONTEXT_SIGNAL_CLUTTER_RATIO: Handler(
                "signal_clutter_ratio", self._clutter, views, ContextNeed.CONTEXT
            ),
        }
