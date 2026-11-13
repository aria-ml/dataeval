__all__ = []

from functools import cached_property
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from dataeval.core._calculators._base import ALL_VIEWS, Calculator, Handler, ViewKind
from dataeval.core._calculators._cache import CalculatorCache
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.flags import ImageStats
from dataeval.types import MeasurementSpec


class _Saturation(NamedTuple):
    """One view's saturation reading and the facts that say how it was measured."""

    status: str
    saturation: float
    valid_fraction: float
    above_range_fraction: float
    upper: float
    rule: str


def _finite(image: NDArray[Any]) -> tuple[NDArray[np.bool_] | None, int]:
    """Return which values of `image` are finite, and how many.

    An integer or Boolean view holds no NaN or infinity, so every value is finite and there is nothing to scan: the mask
    is None.
    """
    if not np.issubdtype(image.dtype, np.inexact):
        return None, image.size
    mask = np.isfinite(image)
    return mask, int(np.count_nonzero(mask))


def _status(has_pixels: bool, finite: int, endpoint: tuple[float, str] | None) -> str:
    """Name why a view has no reading, or ``ok``: no pixels, then no finite value, then no endpoint."""
    if not has_pixels:
        return "empty_region"
    if finite == 0:
        return "no_finite_values"
    return "ok" if endpoint is not None else "unknown_upper"


def _counts_at(
    image: NDArray[Any],
    finite_mask: NDArray[np.bool_] | None,
    dtype: np.dtype[Any],
    upper: float,
    atol: float,
) -> tuple[int, int]:
    """Count the values within `atol` of `upper`, and the finite ones above that band.

    One comparison against Python floats for every dtype, which makes no float copy of an integer view. For a float
    datum the thresholds are rounded to its own dtype: a 3-D float32 datum reaches here as float64 and a 2-D one as
    float32, so a threshold left as given would match one and miss the other.
    """
    low, high = upper - atol, upper + atol
    if dtype.kind == "f":
        with np.errstate(over="ignore"):
            low, high = float(dtype.type(low)), float(dtype.type(high))
    above = (image > high) if finite_mask is None else (image > high) & finite_mask
    return int(np.count_nonzero((image >= low) & (image <= high))), int(np.count_nonzero(above))


@CalculatorRegistry.register(ImageStats)
class PixelStatCalculator(Calculator[ImageStats]):
    """Calculator for pixel-level statistics."""

    def __init__(self, datum: NDArray[Any], cache: "CalculatorCache") -> None:
        self.datum = datum
        self.cache = cache
        self.warnings: list[str] = []

    @cached_property
    def _has_nan(self) -> bool:
        """Check once whether the scaled data contains any NaN values."""
        values = self.cache.scaled
        if not np.issubdtype(values.dtype, np.inexact):
            # Integer pixels carry no NaN, so every reduction below can take its fast path
            # without the scan. Reachable whenever the view is an in-bounds window of an
            # integer image and `normalize_pixel_values` left it in its own dtype.
            return False
        return bool(np.isnan(values).any())

    def _mean_func(self, data: NDArray[Any], **kw: Any) -> Any:
        """Use fast .mean() when no NaN, fall back to nanmean."""
        return np.nanmean(data, **kw) if self._has_nan else np.mean(data, **kw)

    def _std_func(self, data: NDArray[Any], **kw: Any) -> Any:
        return np.nanstd(data, **kw) if self._has_nan else np.std(data, **kw)

    def _var_func(self, data: NDArray[Any], **kw: Any) -> Any:
        return np.nanvar(data, **kw) if self._has_nan else np.var(data, **kw)

    @cached_property
    def _unmeasured(self) -> bool:
        """Whether there is nothing to reduce: no pixels measured, or no scale to place them on.

        The second half only bites when normalizing, which is the one thing a pixel
        statistic needs an interval for. A mean of raw values is a perfectly good mean
        whether or not the data carries an encoding, so it is not gated here.
        """
        return self.cache.is_all_nan or (self.cache.normalize_pixel_values and not self.cache.value_range.is_known)

    @property
    def _unbinnable(self) -> bool:
        """Whether there is no interval to divide into bins.

        Exactly ``CalculatorCache.is_unmeasurable``: a histogram is 256 buckets spanning
        *something*, so unlike :attr:`_unmeasured` it needs an interval whether or not the
        values are being normalized, and the normalization term folds away. Reported as NaN
        rather than raised — the same answer every other unmeasurable view gives — because
        the caller may have declared ranges for the band groups they care about and never
        asked for this view at all.
        """
        return self.cache.is_unmeasurable

    @cached_property
    def _histogram_range(self) -> tuple[float, float]:
        if self.cache.normalize_pixel_values:
            return (0.0, 1.0)
        # The whole datum's range, not this view's, so that a box's or a background's
        # histogram is binned over the same interval as the image it is compared against.
        value_range = self.cache.value_range
        return (float(value_range.pmin), float(value_range.pmax))

    def _report_uncounted(self, counted: int, values: NDArray[Any], r: tuple[float, float]) -> None:
        """Say how many values `np.histogram` dropped for falling outside its range.

        It drops them silently, and the bins give no sign of it — so a histogram binned over
        an interval narrower than the data looks like an ordinary one, and the entropy
        derived from it reads as the entropy of the whole region. Only a *declared* range
        can be narrower than what it describes; a decoded one covers its data by
        construction, so this never fires for ordinary imagery.

        Counted off the bin totals rather than by re-scanning the values, which costs
        nothing beyond a sum over 256 bins.
        """
        measurable = int(np.count_nonzero(~np.isnan(values)))
        dropped = measurable - int(counted)
        if dropped > 0:
            self.warnings.append(
                f"{dropped} of {measurable} values fall outside the histogram range "
                f"[{r[0]:g}, {r[1]:g}] and are not counted, so the histogram and the entropy "
                f"derived from it describe only the values inside it. Widen value_range to "
                f"cover the data, or read the excluded values as out-of-range."
            )

    @cached_property
    def histogram(self) -> NDArray[np.float64]:
        r = self._histogram_range
        counts = np.histogram(self.cache.scaled, bins=256, range=r)[0]
        self._report_uncounted(int(counts.sum()), self.cache.scaled, r)
        return counts

    def get_applicable_flags(self) -> ImageStats:
        """Return which flags this calculator handles."""
        return ImageStats.PIXEL

    def _nan_list(self) -> list[float]:
        """Return NaN in the shape every statistic here returns: one value for the row."""
        return [np.nan]

    def _mean(self) -> list[float]:
        if self._unmeasured:
            return self._nan_list()
        return [float(self._mean_func(self.cache.scaled))]

    def _std(self) -> list[float]:
        if self._unmeasured:
            return self._nan_list()
        return [float(self._std_func(self.cache.scaled))]

    def _var(self) -> list[float]:
        if self._unmeasured:
            return self._nan_list()
        return [float(self._var_func(self.cache.scaled))]

    @cached_property
    def _moments(self) -> tuple[Any, Any, Any]:
        """Compute variance (m2), 3rd central moment (m3), and 4th central moment (m4).

        Uses fast .mean() when no NaN, and caches only the three scalars rather than any
        full-image-sized intermediate.
        """
        mean_fn = self._mean_func
        data = self.cache.scaled.ravel()
        d = data - mean_fn(data)
        d2 = d * d
        m2 = float(mean_fn(d2))
        m3 = float(mean_fn(d2 * d))
        np.multiply(d2, d2, out=d2)
        m4 = float(mean_fn(d2))
        return m2, m3, m4

    def _skew(self) -> list[float]:
        if self._unmeasured:
            return self._nan_list()
        m2, m3, _ = self._moments
        if m2 == 0:
            return [0.0]
        return [m3 / (m2**1.5)]

    def _kurtosis(self) -> list[float]:
        if self._unmeasured:
            return self._nan_list()
        m2, _, m4 = self._moments
        if m2 == 0:
            return [0.0]
        return [m4 / (m2 * m2) - 3.0]

    def _entropy(self) -> list[float]:
        # Data that is entirely NaN was not measured, and "not measured" is NaN — the
        # same answer _mean, _skew, _kurtosis and the visual percentiles give. Falling
        # through would instead read an all-zero histogram as a distribution with no
        # spread and report 0.0, which is a legitimate-looking extreme rather than an
        # absence: an outlier search would flag such an image as genuinely low-entropy.
        if self._unbinnable:
            return self._nan_list()
        h = self.histogram.astype(np.float64)
        total = h.sum()
        if total == 0:
            return [0.0]
        h = h / total
        with np.errstate(divide="ignore", invalid="ignore"):
            return [float(-np.nansum(h * np.log(np.where(h > 0, h, 1.0))) + 0.0)]

    def _as_fraction(self, counted: Any) -> list[float]:
        """Express a count of pixels as a fraction of the pixels measured.

        Pixels the cache excluded are in neither half of the ratio: they are not
        measurements, and — although they are NaN in the image, since NaN is how the
        exclusion is carried — they are not missing data either. Subtracting them from
        both the count and the total is what keeps ``missing`` and ``zeros`` reporting
        on the region actually being reduced over rather than on the mask's size.
        """
        total = self.cache.image.size - self.cache.excluded_total
        if total <= 0:
            return [np.nan]
        return [float(counted / total)]

    def _missing(self) -> list[float]:
        # Deliberately has no all-NaN guard, unlike every other statistic here. This one
        # measures the *presence* of data rather than the data, so it is precisely the
        # statistic that still has an answer when nothing was measured: an out-of-bounds
        # box, or a band group the datum could not supply, reports 1.0. That is the signal
        # a reader needs, and NaN would erase it.
        #
        # Read off the raw view. `scaled` is all-NaN wherever no interval could be
        # established, so counting NaNs in it would report fully present data as wholly
        # missing the moment `normalize_pixel_values` met a range it could not read.
        return self._as_fraction(np.count_nonzero(np.isnan(self.cache.image)) - self.cache.excluded_total)

    def _zeros(self) -> list[float]:
        # "None of these values are zero" is a claim about values, and there are none to
        # make it about — the denominator is a count of pixels, not of measurements, so the
        # ratio comes out 0.0 and reads as a genuine observation of a band that is absent.
        if self._unmeasured:
            return self._nan_list()
        # Excluded pixels are NaN, never 0, so only the denominator needs correcting here.
        #
        # Read off the raw view, as `_missing` is. A zero is a property of the stored
        # value, and `scaled` shifts by `pmin` — against a declared range that does not
        # start at zero, a raw 0 lands somewhere in the middle of [0, 1] and the count
        # would answer a question nobody asked.
        return self._as_fraction(np.count_nonzero(self.cache.image == 0))

    def _endpoint(self) -> tuple[float, str] | None:
        """Return the clipping endpoint and what supplied it, or None when nothing did.

        In the order the statistic defines: a range declared for this view, by the call, a channel group or the image's
        own metadata, then the maximum of the datum's unsigned integer dtype, 1 for a Boolean one. Never the observed
        maximum, and never `value_range` when it was decoded from the data, which would read 12-bit values in uint16 as
        4095 and so call a clipped image unclipped, or the reverse.
        """
        if self.cache.has_declared_range:
            return float(self.cache.value_range.pmax), "declared"
        dtype = self.cache.raw.dtype
        if dtype.kind == "b":
            return 1.0, "dtype"
        # ponytail: uint64 is counted in float64, which cannot tell values within 2**11 of its maximum apart, so it has
        # no dtype endpoint. No sensor writes it. Upgrade is an integer comparison path for that dtype.
        if dtype.kind == "u" and dtype.itemsize <= 4:
            return float(np.iinfo(dtype).max), "dtype"
        return None

    @cached_property
    def _saturation(self) -> _Saturation:
        """Read the view once, for both the statistic and the details that describe it."""
        image = self.cache.image
        finite_mask, finite = _finite(image)
        positions = image.size - self.cache.excluded_total
        valid = finite / positions if positions > 0 else np.nan
        endpoint = self._endpoint()
        upper, rule = endpoint or (np.nan, "")
        status = _status(self.cache.has_pixels, finite, endpoint)
        if status != "ok":
            return _Saturation(status, np.nan, valid, np.nan, upper, rule)
        saturated, above = _counts_at(
            image, finite_mask, self.cache.raw.dtype, upper, self.cache.settings.saturation.atol
        )
        return _Saturation(status, saturated / finite, valid, above / finite, upper, rule)

    @classmethod
    def measurement_specs(
        cls,
        flags: ImageStats,
        view: ViewKind,
        settings: Any,
        declared_range: tuple[float, float] | None,
    ) -> dict[str, MeasurementSpec]:
        """Record the saturation setting and the range the call or a channel group declared.

        Without a declared range the endpoint is left to each image, from its own metadata or its dtype. ponytail: that
        endpoint differs between images, and between a uint8 and a uint16 collection, while the spec reads the same
        for all of them, since the per-row endpoint is in the details, which are not compared when results are
        combined. Upgrade is recording what each image declared, which would refuse any two collections that differ.
        """
        if not flags & ImageStats.PIXEL_SATURATION:
            return {}
        saturation = settings.saturation
        if declared_range is not None:
            policy, declared = "declared", declared_range
        else:
            policy, declared = "inferred", None
        return {
            "saturation": MeasurementSpec(
                algorithm="saturation",
                version="1",
                units="fraction",
                region="background" if ViewKind.MASK in view else "pixels",
                config={"atol": saturation.atol},
                range_policy=policy,
                declared_range=declared,
            )
        }

    def _saturation_fraction(self) -> list[float]:
        return [self._saturation.saturation]

    def details(self, flags: ImageStats, view: ViewKind = ViewKind.WHOLE) -> dict[str, list[Any]]:  # noqa: ARG002
        """Say how the saturation was measured, when it was asked for."""
        if not flags & ImageStats.PIXEL_SATURATION:
            return {}
        reading = self._saturation
        return {
            "saturation_status": [reading.status],
            "saturation_valid_fraction": [reading.valid_fraction],
            "saturation_above_range_fraction": [reading.above_range_fraction],
            "saturation_upper": [reading.upper],
            "saturation_upper_rule": [reading.rule],
        }

    def _histogram(self) -> list[Any]:
        # As _entropy: an all-zero histogram over unmeasured data would read as a real
        # distribution rather than an absent one.
        if self._unbinnable:
            return [[np.nan] * 256]
        return [self.histogram.tolist()]

    def get_empty_values(self) -> dict[str, Any]:
        """Return empty values for pixel statistics."""
        return {
            "histogram": [np.nan] * 256,  # Histogram with 256 bins
            # A row that does not measure it, such as a box row of a background pass, has no status and no rule. Without
            # a string default the padded column would hold the text 'nan'.
            "saturation_status": "",
            "saturation_upper_rule": "",
        }

    def get_handlers(self) -> dict[ImageStats, Handler]:
        """Return mapping of flags to the statistic each produces.

        Every pixel statistic is a NaN-aware reduction over values, so a masked region is
        simply not counted and a band subset is a different set of values to reduce. All
        three views therefore have an answer.
        """
        return {
            ImageStats.PIXEL_MEAN: Handler("mean", self._mean, ALL_VIEWS),
            ImageStats.PIXEL_STD: Handler("std", self._std, ALL_VIEWS),
            ImageStats.PIXEL_VAR: Handler("var", self._var, ALL_VIEWS),
            ImageStats.PIXEL_SKEW: Handler("skew", self._skew, ALL_VIEWS),
            ImageStats.PIXEL_KURTOSIS: Handler("kurtosis", self._kurtosis, ALL_VIEWS),
            ImageStats.PIXEL_ENTROPY: Handler("entropy", self._entropy, ALL_VIEWS),
            ImageStats.PIXEL_MISSING: Handler("missing", self._missing, ALL_VIEWS),
            ImageStats.PIXEL_ZEROS: Handler("zeros", self._zeros, ALL_VIEWS),
            ImageStats.PIXEL_HISTOGRAM: Handler("histogram", self._histogram, ALL_VIEWS),
            ImageStats.PIXEL_SATURATION: Handler("saturation", self._saturation_fraction, ALL_VIEWS),
        }
