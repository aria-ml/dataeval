__all__ = []

from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.utils.preprocessing import (
    BoundingBox,
    ValueRange,
    crop_with_fill,
    get_value_range,
    normalize_image_shape,
    rescale,
)

#: Bit depth of the range a visual statistic is read against — see `CalculatorCache.perceptual`.
_DISPLAY_DEPTH = 8

#: Distinct values that range holds, which is how many bins count all of them.
_DISPLAY_LEVELS = 2**_DISPLAY_DEPTH


def _reduces_widely(dtype: np.dtype[Any]) -> bool:
    """Whether reducing over `dtype` is at least as accurate as reducing over a float64 copy of it.

    The gate on `CalculatorCache._windowed`'s slice. That slice replaced a float64 copy, so
    it may only be taken where the narrower dtype costs no precision. NumPy accumulates a
    mean or a variance over integers in float64 and over float64 in float64, so both are
    safe; over float32 it accumulates in float32, and a constant float32 image would come
    back with a variance of 1e-14 where the copy answered 0.
    """
    return dtype == np.float64 or dtype.kind in "biu"


class CalculatorCache:
    """
    A calculator cache for a single datum (image, text, etc.).

    Provides preprocessing and cached transformations of the raw datum.
    This class adapts based on the data type passed in.

    Parameters
    ----------
    datum : Any
        The raw data element statistics are computed on.
    box : BoundingBox or None, default None
        Region of the datum to reduce over. None reduces over the whole datum.
    per_channel : bool, default False
        Whether the consuming calculators are computing per-channel statistics.
    normalize_pixel_values : bool, default False
        Whether to rescale values to [0, 1] before a *pixel* statistic is computed. Visual
        statistics ignore it — they resolve their own reference, always; see
        :attr:`perceptual`.
    exclude : NDArray[np.bool_] or None, default None
        A ``(H, W)`` mask over the *whole datum* marking pixels to leave out of every
        statistic, True where a pixel is excluded. Excluded pixels are set to NaN in
        :attr:`image`, which the calculators' NaN-aware reductions then skip — the
        same path an out-of-bounds bounding box already takes. Applied after the crop,
        so it composes with `box`: masking the boxes of an image while cropping to a
        widened box reduces over that box's surroundings only. Ignored for data with
        fewer than three dimensions, where box geometry does not apply.
    bands : tuple[int, ...] or None, default None
        Indices of the channel axis to restrict every statistic to, or None for all of
        them. The band-axis counterpart of `box` and `exclude`: those narrow the spatial
        axes, this narrows the channel axis, and the three compose. Applied *after* the
        crop, so a group of a 224-band cube costs one box-sized copy per row rather than a
        full-resolution copy of the selected bands. A datum that cannot supply every index
        is measured over an all-NaN slice of the right shape rather than skipped — see
        :attr:`image`.
    value_range : ValueRange or None, default None
        The datum's value range, if the caller already established it. Every view of one
        datum shares the interval and establishing it costs a scan of the whole datum, so
        a caller building one cache per row — as :func:`~dataeval.core.compute_stats`
        does, once per box — should read it once and pass it here. Also how a caller
        passes a *declared* range down, which is the only way float data outside the two
        conventional image spellings gets one at all. Found on demand when None.
    """

    def __init__(
        self,
        datum: Any,
        box: BoundingBox | None = None,
        per_channel: bool = False,
        normalize_pixel_values: bool = False,
        exclude: NDArray[np.bool_] | None = None,
        value_range: ValueRange | None = None,
        bands: tuple[int, ...] | None = None,
    ) -> None:
        is_spatial = len(datum.shape) >= 2
        self.raw = datum
        # Assume image data for now (will be generic in future)
        self.width: int = datum.shape[-1] if is_spatial else 0
        self.height: int = datum.shape[-2] if is_spatial else 0
        self.shape: tuple[int, ...] = datum.shape
        self.per_channel_mode = per_channel
        self.normalize_pixel_values = normalize_pixel_values
        self.has_box = box is not None
        self.exclude = exclude
        self._value_range = value_range
        self.bands = bands

        # Ensure bounding box
        self.box = BoundingBox(0, 0, self.width, self.height, image_shape=datum.shape) if box is None else box

    @cached_property
    def value_range(self) -> ValueRange:
        """Range of the datum, read off the whole datum rather than any view of it.

        Anchored on :attr:`raw` so that every view of one datum — the full image, each
        box, the background — is scaled and binned against the same interval, which is
        what makes their statistics comparable. Reading it off a view instead lets the
        view's own extremes pick the range: a background whose bright pixels all sat
        inside the boxes, or a dark box within a bright image, would infer a lower depth
        and land its histogram in a different range than the image it is compared against.

        Read once per datum by the caller where there is one, since a scan of the whole
        datum repeated once per box is the cost this property would otherwise carry.

        May be :attr:`~dataeval.utils.preprocessing.ValueRange.is_known` False, which is
        not an error here — :attr:`scaled` and the histogram raise on it, while
        :attr:`~dataeval.flags.ImageStats.DIMENSION_DEPTH` reports the ``nan`` depth as
        the answer it is.
        """
        return self._value_range if self._value_range is not None else get_value_range(self.raw)

    @cached_property
    def window_mask(self) -> NDArray[np.bool_] | None:
        """:attr:`exclude` cropped to the same window as :attr:`image`, or None when unset.

        Cropped through the same call as the pixels so the two cannot fall out of
        alignment, with out-of-image pixels filled False — a pixel that is not part of
        the datum is not part of the excluded region either; it is already NaN from the
        image's own fill and is counted as such.
        """
        if self.exclude is None:
            return None
        window = self.box.xyxy_int
        if self._has_geometry and self._is_interior(window, self.exclude.shape):
            # Wholly inside, so there is nothing to fill and the mask is already a slice.
            x0, y0, x1, y1 = window
            return self.exclude[y0:y1, x0:x1]
        # A leading axis makes the (H, W) mask a one-channel image for the crop, which is
        # the only shape crop_with_fill windows; dropped again on the way out.
        return crop_with_fill(self.exclude[None, ...], window, fill=False)[0][0]

    @cached_property
    def channel_count(self) -> int:
        """Number of channels :attr:`image` presents, matching :attr:`per_channel`'s leading axis."""
        return self.image.shape[0] if self.image.ndim >= 3 else 1

    @cached_property
    def excluded_per_channel(self) -> int:
        """Pixels per channel that :attr:`exclude` removes from the window. 0 when unset."""
        return 0 if self.window_mask is None else int(np.count_nonzero(self.window_mask))

    @cached_property
    def excluded_total(self) -> int:
        """Pixel values across all channels that :attr:`exclude` removes from the window.

        The denominator correction for any statistic expressed as a fraction of the
        datum: an excluded pixel is neither a hole in the data nor a measurement, so it
        belongs in neither half of such a ratio.
        """
        return self.excluded_per_channel * self.channel_count

    @cached_property
    def _has_geometry(self) -> bool:
        """Whether box geometry applies to this datum at all.

        Bounding-box geometry assumes channels-first dimensionality >= 3. An exclusion mask
        is spatial in the same way, so it brings a datum in here too — that is what gets a
        2-D single-channel image masked rather than silently returned whole, its background
        row carrying no box of its own to bring it in.
        """
        return self.has_box or self.raw.ndim >= 3 or self.exclude is not None

    @cached_property
    def _window_slice(self) -> NDArray[Any] | None:
        """The window as a plain slice of the datum, or None where it is not one.

        None when the window leaves the image and some of it must be filled, when the datum
        carries no box geometry, or when its dtype is one the reductions need widened —
        see :func:`_reduces_widely`. Says nothing about `exclude`: this is the datum's own
        pixels, which is what a *count* of them reads, and the masking happens on the copy
        :attr:`_windowed` makes.
        """
        if not self._has_geometry:
            return None
        image = normalize_image_shape(self.raw)
        window = self.box.xyxy_int
        if not _reduces_widely(image.dtype) or not self._is_interior(window, image.shape):
            return None
        x0, y0, x1, y1 = window
        # A read-only view rather than a copy, since nothing here writes pixels.
        return image[:, y0:y1, x0:x1]

    @cached_property
    def _integer_window(self) -> NDArray[Any] | None:
        """:attr:`_window_slice` band-selected, where its pixels are integers — otherwise None.

        What both :attr:`display_counts` and :attr:`is_all_nan` read, and they read it for
        the same reason: an integer window holds no NaN of its own, so every absent pixel
        in a view built from it is one `exclude` masked out, and every present one is a
        value that can be counted rather than sorted. None wherever that reasoning does not
        hold — a float datum, a window that leaves the image, or a band group this datum
        cannot supply, which :attr:`image` answers with NaN instead.
        """
        window = self._window_slice
        if window is None or not window.size or not np.issubdtype(window.dtype, np.integer):
            return None
        if self.bands is None:
            return window
        if window.ndim >= 3 and max(self.bands) < window.shape[0]:
            return window[list(self.bands)]
        return None

    @cached_property
    def _display_window(self) -> NDArray[np.uint8] | None:
        """:attr:`_integer_window` where :attr:`perceptual` is simply those pixels, else None.

        The one place the perceptual view's identity path is restated. `perceptual` takes
        that path when the datum's range is already the display range, at which point it
        hands :attr:`image` back untouched — so where the window is the datum's own 8-bit
        pixels, those pixels *are* the perceptual view, and a statistic can read them
        without the view ever being built. None for a float datum, a padded window, a band
        group the datum cannot supply, or a range `perceptual` would have to rescale.

        Pinned to `perceptual` by test rather than by construction: asking `perceptual`
        directly would mean building the array the readers below exist to avoid.
        """
        window = self._integer_window
        if window is None or window.dtype != np.uint8:
            return None
        if not self.value_range.is_known or self.value_range.depth != _DISPLAY_DEPTH:
            return None
        return window

    @cached_property
    def is_fully_measured(self) -> bool:
        """Whether every pixel of the view holds a measurement, known without scanning for one.

        A licence to take the plain reductions instead of the NaN-aware ones, which each
        scan the whole view to discover the same thing. False wherever it cannot be
        answered outright — including views that do happen to hold no NaN, since this
        makes no claim about the ones it declines.
        """
        return self._integer_window is not None and self.window_mask is None

    @cached_property
    def display_counts(self) -> NDArray[np.intp] | None:
        """How many measured pixels hold each 8-bit display value, one row per band.

        Shape ``(bands, 256)``, or None wherever :attr:`_display_window` is.

        Read off the datum and the mask rather than off :attr:`perceptual`, which is what
        makes it worth having: a background view is float64 with a NaN hole wherever a box
        was, and every percentile of it otherwise costs a compaction and a partition of the
        whole image. The masked pixels are counted and *subtracted* rather than stepped
        around, because the boxes are a small part of an image and counting them is far
        cheaper than counting around them.
        """
        window = self._display_window
        if window is None:
            return None
        counts = np.stack([np.bincount(band.ravel(), minlength=_DISPLAY_LEVELS) for band in window])
        mask = self.window_mask
        if mask is not None:
            counts -= np.stack([np.bincount(band[mask], minlength=_DISPLAY_LEVELS) for band in window])
        return counts

    @cached_property
    def display_grayscale(self) -> NDArray[np.float64] | None:
        """:attr:`perceptual` averaged down to one ``(H, W)`` band, or None wherever it cannot be.

        The same values ``np.mean(perceptual, axis=0)`` gives, to the bit: the datum's own
        8-bit pixels are exact in float64, so summing them across bands reaches the same
        total whether they were promoted first or not, and a masked pixel is masked in
        every band so it averages to NaN either way.

        Worth reading separately because of what it lets the caller *not* build. Averaging
        `perceptual` means promoting every band of the window into one float64 array —
        eight times the datum, and the largest allocation in a statistics run over
        megapixel imagery — which sharpness is the only statistic ever to have wanted.
        """
        window = self._display_window
        if window is None or window.ndim != 3:
            return None
        bands = window.shape[0]
        # Totalled in uint16 rather than accumulated in float64 wherever the brightest
        # possible band stack still fits it. The total is a small whole number either way
        # and dividing it reaches the same float64 the wider accumulator would have, for
        # most of a pass over the window less; a stack deep enough to overflow falls back.
        fits = bands * np.iinfo(np.uint8).max <= np.iinfo(np.uint16).max
        summed = window.sum(axis=0, dtype=np.uint16 if fits else np.float64)
        grayscale = summed / bands
        mask = self.window_mask
        if mask is not None:
            grayscale[mask] = np.nan
        return grayscale

    @cached_property
    def _windowed(self) -> NDArray[Any]:
        """The datum narrowed on its spatial axes: cropped to :attr:`box`, masked by `exclude`."""
        if not self._has_geometry:
            # For data with < 3 dimensions, don't normalize or clip
            return self.raw
        window_slice = self._window_slice
        if window_slice is not None and self.exclude is None:
            # Nothing to fill and nothing to mask, so the window is already a slice of the
            # datum. Skipping `crop_with_fill` here skips the copy *and* the float
            # promotion its NaN fill carries: an 8-bit image stays 8-bit, which is eight
            # times less to sort, convolve and scan for every statistic below.
            return window_slice
        cropped = crop_with_fill(normalize_image_shape(self.raw), self.box.xyxy_int)[0]
        window_mask = self.window_mask
        if window_mask is not None:
            # In place: crop_with_fill has already allocated a fresh float array (its
            # NaN fill promotes any integer image), so masking costs no further copy.
            cropped[:, window_mask] = np.nan
        return cropped

    @staticmethod
    def _is_interior(window: tuple[int, int, int, int], shape: tuple[int, ...]) -> bool:
        """Whether `window` lies wholly inside a ``(C, H, W)`` datum and encloses real pixels.

        The condition under which `crop_with_fill` would fill nothing, so slicing gives the
        same pixels for none of the cost. Degenerate windows are excluded deliberately:
        `crop_with_fill` widens an empty one to a single all-fill pixel, and a slice would
        instead produce a zero-sized array that the reductions below answer differently.
        """
        x0, y0, x1, y1 = window
        height, width = shape[-2:]
        return 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height

    @cached_property
    def image(self) -> NDArray[Any]:
        """The region every statistic here reduces over, on all three axes.

        Bands are taken last so the copy is of the window rather than of the whole datum.
        A group the datum cannot fully supply is *substituted* with NaN, never skipped:
        skipping means the calculators never run, so the column is never produced for this
        datum, and the aggregation appends by name — a name missing from one datum silently
        shortens its array and misaligns it against the source index. All-or-nothing rather
        than reduced over whichever bands are present, so one column name means one thing.
        """
        windowed = self._windowed
        if self.bands is None:
            return windowed
        if windowed.ndim >= 3 and max(self.bands) < windowed.shape[0]:
            return windowed[list(self.bands)]
        hw = windowed.shape[-2:] if windowed.ndim >= 2 else (1, 1)
        return self.nan_like((len(self.bands), *hw))

    @cached_property
    def is_all_nan(self) -> bool:
        """Check if the image data is entirely NaN (e.g. from an out-of-bounds bounding box)."""
        if self._integer_window is not None:
            # An integer window brings no NaN of its own, so the view is wholly absent
            # exactly when `exclude` covered all of it. Read off the one-band mask rather
            # than scanned out of the float64 copy every band was promoted into.
            mask = self.window_mask
            return mask is not None and bool(mask.all())
        image = self.image
        if image.size and not np.issubdtype(image.dtype, np.inexact):
            # An integer or boolean view cannot hold NaN, so the answer is known without
            # the scan — which is over every pixel, and reached once per statistic family.
            return False
        return bool(np.isnan(image).all())

    @cached_property
    def is_unmeasurable(self) -> bool:
        """Whether this view supports no reading at all: no values, or no interval to place them on.

        The one question two calculator families were each answering for themselves, in the
        same two terms this class already owns. Refining what counts as unmeasurable — a
        zero-span declared range, a mask covering the whole window — belongs here, where
        both families see it at once, rather than in whichever module is edited first.
        """
        return self.is_all_nan or not self.value_range.is_known

    def nan_like(self, shape: tuple[int, ...] | None = None) -> NDArray[np.float64]:
        """Return an all-NaN float64 array standing in for a view that could not be measured.

        Absence is reported as float64 NaN rather than raised or sentinelled, and the dtype
        is load-bearing: `np.nanpercentile`, `np.histogram` and `edge_filter` all behave
        differently on a float32 or object array. Kept in one place so the policy can change
        in one place.
        """
        return np.full(self.image.shape if shape is None else shape, np.nan, dtype=np.float64)

    @cached_property
    def scaled(self) -> NDArray[Any]:
        """The view normalized to [0, 1], or all-NaN where there is no interval to divide by.

        Anchored on the whole datum's range rather than this view's, so that the image,
        each box and the background all scale onto one interval — see :attr:`value_range`.

        All-NaN rather than an error where no interval could be established, matching
        :attr:`perceptual` and everything else here: an unmeasurable view reports absence,
        it does not stop the run. :func:`~dataeval.utils.preprocessing.rescale` still
        raises when called directly, where an unknown range is a mistake at the call site
        rather than a property of one datum among many.
        """
        if not self.normalize_pixel_values:
            return self.image
        if not self.value_range.is_known:
            return self.nan_like()
        return rescale(self.image, value_range=self.value_range)

    @cached_property
    def perceptual(self) -> NDArray[Any]:
        """The view on the 0–255 display range, or all-NaN where no reference exists.

        What a *visual* statistic reads, and never what a pixel statistic reads. Brightness
        is a claim about how an image looks to a person, which is only meaningful against a
        full-scale reference: a median of 15677 says nothing until you know whether the
        sensor's ceiling is 65535 or 40000. Resolving that reference is the whole
        difference between the two families — a pixel statistic reports the values as
        stored, a visual one reports where they sit between black and white.

        0–255 rather than [0, 1] because that is the range perception is actually
        calibrated against, and everything downstream already assumes it —
        :func:`~dataeval.utils.preprocessing.edge_filter` clips there, and
        :func:`~dataeval.utils.preprocessing.to_canonical_grayscale` composites there. It
        also costs the common case nothing: `rescale` returns its input untouched when the
        source depth is already the target, so an 8-bit image and an 8-bit image held in a
        float array both take this property's identity path.

        All-NaN where :attr:`value_range` established none, which is the answer rather than
        an error — a band carrying elevation or temperature has no perceptual reading to
        report, whereas :attr:`scaled` and the histogram genuinely cannot proceed and so
        raise instead.
        """
        if not self.value_range.is_known:
            return self.nan_like()
        perceptual = rescale(self.image, depth=_DISPLAY_DEPTH, value_range=self.value_range)
        if perceptual is self.image:
            # The identity path, taken only when the source depth is already 8-bit, where
            # the values are on the display range by construction and clipping would cost
            # a copy for nothing.
            return perceptual
        # A *declared* interval can be narrower than the data it describes, which maps the
        # values outside it past 0-255 — at which point "brightness" is no longer a display
        # value. `edge_filter` already clips there, so without this the family disagrees
        # with itself: sharpness saturates while brightness and darkness run free.
        return np.clip(perceptual, 0.0, 2**_DISPLAY_DEPTH - 1)

    @cached_property
    def per_channel(self) -> NDArray[Any]:
        # For data with >= 3 dimensions, reshape as (channels, -1)
        # For data with < 3 dimensions, treat as single channel
        return self._flatten_channels(self.scaled)

    @cached_property
    def per_channel_perceptual(self) -> NDArray[Any]:
        """:attr:`perceptual` in :attr:`per_channel`'s layout, for per-channel visual statistics."""
        return self._flatten_channels(self.perceptual)

    @cached_property
    def per_channel_raw(self) -> NDArray[Any]:
        """:attr:`image` in :attr:`per_channel`'s layout, untouched by any rescaling.

        What a statistic about the *presence* of values rather than about the values
        themselves reads — ``missing`` counts NaNs and ``zeros`` counts exact zeros, and
        neither question is asked of the normalized copy. :attr:`scaled` is not answerable
        for either: it substitutes all-NaN where no interval could be established, which
        would read as wholly missing data, and it shifts by ``pmin``, which moves a raw
        zero off zero for any range that does not start there.
        """
        return self._flatten_channels(self.image)

    def _flatten_channels(self, values: NDArray[Any]) -> NDArray[Any]:
        """Reshape to ``(channels, -1)``, giving lower-dimensional data a channel axis of 1."""
        return values.reshape(self.channel_count, -1)
