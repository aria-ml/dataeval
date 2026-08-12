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
        self._bitdepth = bitdepth

        # Ensure bounding box
        self.box = BoundingBox(0, 0, self.width, self.height, image_shape=datum.shape) if box is None else box

    @cached_property
    def bitdepth(self) -> BitDepth:
        """Bit depth of the datum, read off the whole datum rather than any view of it.

        Anchored on :attr:`raw` so that every view of one datum — the full image, each
        box — is scaled and binned against the same range, which is what makes their
        statistics comparable. Reading it off a view instead lets the view's own extremes
        pick the range: a dark box within a bright image would infer a lower depth and
        land its histogram in a different range than the image it is being compared
        against.

        Read once per datum by the caller where there is one, since a scan of the whole
        datum repeated once per box is the cost this property would otherwise carry.
        """
        return self._bitdepth if self._bitdepth is not None else get_bitdepth(self.raw)

    @cached_property
    def image(self) -> NDArray[Any]:
        # Crop/pad to the bounding box (a full-image default when none was given), but only for
        # image-like data: bounding-box geometry assumes channels-first dimensionality >= 3.
        if self.has_box or self.raw.ndim >= 3:
            return crop_with_fill(normalize_image_shape(self.raw), self.box.xyxy_int)[0]
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
        # image and each box scale onto one range — see `bitdepth`.
        return rescale(self.image, bitdepth=self.bitdepth)

    @cached_property
    def per_channel(self) -> NDArray[Any]:
        # For data with >= 3 dimensions, reshape as (channels, -1)
        # For data with < 3 dimensions, treat as single channel
        if self.image.ndim >= 3:
            return self.scaled.reshape(self.image.shape[0], -1)
        # For lower-dimensional data, add a channel dimension
        return self.scaled.reshape(1, -1)
