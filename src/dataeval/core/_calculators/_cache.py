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
        # A leading axis makes the (H, W) mask a one-channel image for the crop, which is
        # the only shape crop_with_fill windows; dropped again on the way out.
        return crop_with_fill(self.exclude[None, ...], self.box.xyxy_int, fill=False)[0][0]

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
    def _windowed(self) -> NDArray[Any]:
        """The datum narrowed on its spatial axes: cropped to :attr:`box`, masked by `exclude`."""
        # Crop/pad to the bounding box (a full-image default when none was given), but only for
        # image-like data: bounding-box geometry assumes channels-first dimensionality >= 3.
        # An exclusion mask is spatial in the same way and takes the same path, so that a
        # 2-D single-channel image gets masked rather than silently returned whole — its
        # background row carries no box of its own to bring it in here.
        if self.has_box or self.raw.ndim >= 3 or self.exclude is not None:
            cropped = crop_with_fill(normalize_image_shape(self.raw), self.box.xyxy_int)[0]
            window_mask = self.window_mask
            if window_mask is not None:
                # In place: crop_with_fill has already allocated a fresh float array (its
                # NaN fill promotes any integer image), so masking costs no further copy.
                cropped[:, window_mask] = np.nan
            return cropped
        # For data with < 3 dimensions, don't normalize or clip
        return self.raw

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
        return bool(np.isnan(self.image).all())

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
