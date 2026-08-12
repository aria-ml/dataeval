__all__ = []

from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.utils.preprocessing import (
    BitDepth,
    BoundingBox,
    crop_with_fill,
    get_bitdepth,
    normalize_image_shape,
    rescale,
)


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
        Whether to rescale pixel values to [0, 1] before any statistic is computed.
    exclude : NDArray[np.bool_] or None, default None
        A ``(H, W)`` mask over the *whole datum* marking pixels to leave out of every
        statistic, True where a pixel is excluded. Excluded pixels are set to NaN in
        :attr:`image`, which the calculators' NaN-aware reductions then skip — the
        same path an out-of-bounds bounding box already takes. Applied after the crop,
        so it composes with `box`: masking the boxes of an image while cropping to a
        widened box reduces over that box's surroundings only. Ignored for data with
        fewer than three dimensions, where box geometry does not apply.
    bitdepth : BitDepth or None, default None
        The datum's bit depth, if the caller already found it. Every view of one datum
        shares the value and finding it costs a scan of the whole datum, so a caller
        building one cache per row — as :func:`~dataeval.core.compute_stats` does, once
        per box — should read it once and pass it here. Found on demand when None.
    """

    def __init__(
        self,
        datum: Any,
        box: BoundingBox | None = None,
        per_channel: bool = False,
        normalize_pixel_values: bool = False,
        exclude: NDArray[np.bool_] | None = None,
        bitdepth: BitDepth | None = None,
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
        self._bitdepth = bitdepth

        # Ensure bounding box
        self.box = BoundingBox(0, 0, self.width, self.height, image_shape=datum.shape) if box is None else box

    @cached_property
    def bitdepth(self) -> BitDepth:
        """Bit depth of the datum, read off the whole datum rather than any view of it.

        Anchored on :attr:`raw` so that every view of one datum — the full image, each
        box, the background — is scaled and binned against the same range, which is what
        makes their statistics comparable. Reading it off a view instead lets the view's
        own extremes pick the range: a background whose bright pixels all sat inside the
        boxes, or a dark box within a bright image, would infer a lower depth and land
        its histogram in a different range than the image it is being compared against.

        Read once per datum by the caller where there is one, since a scan of the whole
        datum repeated once per box is the cost this property would otherwise carry.
        """
        return self._bitdepth if self._bitdepth is not None else get_bitdepth(self.raw)

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
    def image(self) -> NDArray[Any]:
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
    def is_all_nan(self) -> bool:
        """Check if the image data is entirely NaN (e.g. from an out-of-bounds bounding box)."""
        return bool(np.isnan(self.image).all())

    @cached_property
    def scaled(self) -> NDArray[Any]:
        if not self.normalize_pixel_values:
            return self.image
        # Anchored on the whole datum's bit depth rather than this view's, so that the
        # image, each box and the background all scale onto one range — see `bitdepth`.
        return rescale(self.image, bitdepth=self.bitdepth)

    @cached_property
    def per_channel(self) -> NDArray[Any]:
        # For data with >= 3 dimensions, reshape as (channels, -1)
        # For data with < 3 dimensions, treat as single channel
        if self.image.ndim >= 3:
            return self.scaled.reshape(self.image.shape[0], -1)
        # For lower-dimensional data, add a channel dimension
        return self.scaled.reshape(1, -1)
