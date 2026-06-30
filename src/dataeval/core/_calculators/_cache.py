__all__ = []

from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.utils.preprocessing import (
    BoundingBox,
    crop_with_fill,
    normalize_image_shape,
    rescale,
)


class CalculatorCache:
    """
    A calculator cache for a single datum (image, text, etc.).

    Provides preprocessing and cached transformations of the raw datum.
    This class adapts based on the data type passed in.
    """

    def __init__(
        self,
        datum: Any,
        box: BoundingBox | None = None,
        per_channel: bool = False,
        normalize_pixel_values: bool = False,
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

        # Ensure bounding box
        self.box = BoundingBox(0, 0, self.width, self.height, image_shape=datum.shape) if box is None else box

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
        if self.normalize_pixel_values:
            return rescale(self.image)
        return self.image

    @cached_property
    def per_channel(self) -> NDArray[Any]:
        # For data with >= 3 dimensions, reshape as (channels, -1)
        # For data with < 3 dimensions, treat as single channel
        if self.image.ndim >= 3:
            return self.scaled.reshape(self.image.shape[0], -1)
        # For lower-dimensional data, add a channel dimension
        return self.scaled.reshape(1, -1)
