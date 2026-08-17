__all__ = []

from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.core._calculators._base import ALL_VIEWS, Calculator, Handler
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
        self.warnings: list[str] = []

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
        if self.per_channel_mode:
            counts = np.apply_along_axis(lambda y: np.histogram(y, bins=256, range=r)[0], 1, self.cache.per_channel)
            self._report_uncounted(int(counts.sum()), self.cache.per_channel, r)
            return counts
        counts = np.histogram(self.cache.scaled, bins=256, range=r)[0]
        self._report_uncounted(int(counts.sum()), self.cache.scaled, r)
        return counts

    def get_applicable_flags(self) -> ImageStats:
        """Return which flags this calculator handles."""
        return ImageStats.PIXEL

    def _nan_list(self) -> list[float]:
        """Return NaN values matching the expected output shape for all-NaN data."""
        if self.per_channel_mode:
            return [np.nan] * self.cache.channel_count
        return [np.nan]

    def _mean(self) -> list[float]:
        if self._unmeasured:
            return self._nan_list()
        if self.per_channel_mode:
            return self._mean_func(self.cache.per_channel, axis=1).tolist()
        return [float(self._mean_func(self.cache.scaled))]

    def _std(self) -> list[float]:
        if self._unmeasured:
            return self._nan_list()
        if self.per_channel_mode:
            return self._std_func(self.cache.per_channel, axis=1).tolist()
        return [float(self._std_func(self.cache.scaled))]

    def _var(self) -> list[float]:
        if self._unmeasured:
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
        if self._unmeasured:
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
        if self._unmeasured:
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
        # Data that is entirely NaN was not measured, and "not measured" is NaN — the
        # same answer _mean, _skew, _kurtosis and the visual percentiles give. Falling
        # through would instead read an all-zero histogram as a distribution with no
        # spread and report 0.0, which is a legitimate-looking extreme rather than an
        # absence: an outlier search would flag such an image as genuinely low-entropy.
        if self._unbinnable:
            return self._nan_list()
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

    def _as_fraction(self, counted: Any) -> list[float]:
        """Express a per-channel or whole-image count as a fraction of the pixels measured.

        Pixels the cache excluded are in neither half of the ratio: they are not
        measurements, and — although they are NaN in the image, since NaN is how the
        exclusion is carried — they are not missing data either. Subtracting them from
        both the count and the total is what keeps ``missing`` and ``zeros`` reporting
        on the region actually being reduced over rather than on the mask's size.
        """
        if self.per_channel_mode:
            total = self.cache.per_channel_raw.shape[1] - self.cache.excluded_per_channel
            if total <= 0:
                return [np.nan] * self.cache.channel_count
            return (np.asarray(counted, dtype=np.float64) / total).tolist()
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
        # Read off the raw view in both modes. `scaled` is all-NaN wherever no interval
        # could be established, so counting NaNs in it would report fully present data as
        # wholly missing the moment `normalize_pixel_values` met a range it could not read.
        if self.per_channel_mode:
            nans = np.count_nonzero(np.isnan(self.cache.per_channel_raw), axis=1) - self.cache.excluded_per_channel
            return self._as_fraction(nans)
        return self._as_fraction(np.count_nonzero(np.isnan(self.cache.image)) - self.cache.excluded_total)

    def _zeros(self) -> list[float]:
        # "None of these values are zero" is a claim about values, and there are none to
        # make it about — the denominator is a count of pixels, not of measurements, so the
        # ratio comes out 0.0 and reads as a genuine observation of a band that is absent.
        if self._unmeasured:
            return self._nan_list()
        # Excluded pixels are NaN, never 0, so only the denominator needs correcting here.
        #
        # Read off the raw view in both modes, as `_missing` is. A zero is a property of
        # the stored value, and `scaled` shifts by `pmin` — against a declared range that
        # does not start at zero, a raw 0 lands somewhere in the middle of [0, 1] and the
        # count would answer a question nobody asked.
        if self.per_channel_mode:
            return self._as_fraction(np.count_nonzero(self.cache.per_channel_raw == 0, axis=1))
        return self._as_fraction(np.count_nonzero(self.cache.image == 0))

    def _histogram(self) -> list[Any]:
        # As _entropy: an all-zero histogram over unmeasured data would read as a real
        # distribution rather than an absent one.
        if self._unbinnable:
            empty = [np.nan] * 256
            return [empty] * self.cache.channel_count if self.per_channel_mode else [empty]
        if self.per_channel_mode:
            return self.histogram.tolist()
        return [self.histogram.tolist()]

    def get_empty_values(self) -> dict[str, Any]:
        """Return empty values for pixel statistics."""
        return {
            "histogram": [np.nan] * 256,  # Histogram with 256 bins
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
        }
