__all__ = []

from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval.core._calculators._base import Calculator, Handler, ViewKind
from dataeval.core._calculators._cache import CalculatorCache
from dataeval.core._calculators._registry import CalculatorRegistry
from dataeval.flags import ImageStats


@CalculatorRegistry.register(ImageStats)
class DimensionStatCalculator(Calculator[ImageStats]):
    """Calculator for dimension and geometry statistics."""

    def __init__(self, datum: NDArray[Any], cache: "CalculatorCache") -> None:
        self.datum = datum
        self.cache = cache
        # Check if this is spatial data (has width and height dimensions)
        self.is_spatial = len(cache.shape) >= 2

    def get_applicable_flags(self) -> ImageStats:
        """Return which flags this calculator handles."""
        return ImageStats.DIMENSION

    def _offset_x(self) -> list[float]:
        # For non-spatial data, offset concepts don't apply
        return [self.cache.box.x0 if self.is_spatial else np.nan]

    def _offset_y(self) -> list[float]:
        # For non-spatial data, offset concepts don't apply
        return [self.cache.box.y0 if self.is_spatial else np.nan]

    def _width(self) -> list[float]:
        # For non-spatial data, return the total data length instead of width
        if not self.is_spatial:
            return [float(self.cache.shape[-1]) if len(self.cache.shape) >= 1 else np.nan]
        return [self.cache.box.width]

    def _height(self) -> list[float]:
        # For non-spatial data (1D), height concept doesn't apply
        if not self.is_spatial:
            return [np.nan]
        return [self.cache.box.height]

    def _channels(self) -> list[int]:
        # For data with >= 3 dimensions, return the channel dimension
        # For lower-dimensional data, return 1 (single channel)
        if len(self.cache.shape) >= 3:
            return [self.cache.shape[-3]]
        return [1]

    def _size(self) -> list[float]:
        # For non-spatial data, return total number of elements
        if not self.is_spatial:
            return [float(np.prod(self.cache.shape))]
        return [self.cache.box.width * self.cache.box.height]

    def _aspect_ratio(self) -> list[float]:
        # Normalized aspect ratio only makes sense for spatial data
        if not self.is_spatial:
            return [np.nan]
        box = self.cache.box
        # Wide is positive - tall is negative
        mult, divisor, dividend = (-1, box.width, box.height) if box.height > box.width else (1, box.height, box.width)
        return [float("nan") if dividend == 0 else mult * (1 - (divisor / dividend))]

    def _depth(self) -> list[float]:
        # Through the cache rather than off `raw` directly, so a declared range is honored
        # and the answer is the same one `scaled` and the histogram were computed against.
        # NaN where the data carries no encoding to read — see `ValueRange`.
        return [self.cache.value_range.depth]

    def _center(self) -> list[list[float]]:
        # Center only makes sense for spatial data
        if not self.is_spatial:
            return [[np.nan, np.nan]]
        box = self.cache.box
        return [[(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2]]

    def _distance_center(self) -> list[float]:
        # Distance from center only makes sense for spatial data
        if not self.is_spatial:
            return [np.nan]
        box = self.cache.box
        raw = self.cache.raw
        return [
            float(
                np.sqrt(
                    np.square(((box.x0 + box.x1) / 2) - (raw.shape[-1] / 2))
                    + np.square(((box.y0 + box.y1) / 2) - (raw.shape[-2] / 2)),
                ),
            ),
        ]

    def _distance_edge(self) -> list[float]:
        # Distance from edge only makes sense for spatial data
        if not self.is_spatial:
            return [np.nan]
        box = self.cache.box
        raw = self.cache.raw
        return [
            float(
                np.min(
                    [
                        np.abs(box.x0),
                        np.abs(box.y0),
                        np.abs(box.x1 - raw.shape[-1]),
                        np.abs(box.y1 - raw.shape[-2]),
                    ],
                ),
            ),
        ]

    def _invalid_box(self) -> list[bool]:
        return [not self.cache.box.is_valid()]

    def get_empty_values(self) -> dict[str, Any]:
        """Return empty values for dimension statistics."""
        return {
            "center": [np.nan, np.nan],  # 2D coordinate array
        }

    def get_handlers(self) -> dict[ImageStats, Handler]:
        """Return mapping of flags to the statistic each produces.

        This is the calculator that makes the declaration belong per statistic rather than
        per class. Geometry is invariant under both other views — masking pixels does not
        move a box and dropping bands does not narrow one, so ``background_width`` and
        ``rgb_width`` would restate the plain value under a new name. ``depth`` is not:
        bands of one cube can carry different encodings, so a group's depth is genuinely
        its own and must be reported beside the group's other statistics rather than
        contradicted by the whole cube's. ``channels`` stays whole-only by the same logic
        in reverse — a group's size is a property of the caller's own argument.
        """
        return {
            # Geometry takes `Handler`'s default of WHOLE alone; only depth opts into BAND.
            ImageStats.DIMENSION_OFFSET_X: Handler("offset_x", self._offset_x),
            ImageStats.DIMENSION_OFFSET_Y: Handler("offset_y", self._offset_y),
            ImageStats.DIMENSION_WIDTH: Handler("width", self._width),
            ImageStats.DIMENSION_HEIGHT: Handler("height", self._height),
            ImageStats.DIMENSION_CHANNELS: Handler("channels", self._channels),
            ImageStats.DIMENSION_SIZE: Handler("size", self._size),
            ImageStats.DIMENSION_ASPECT_RATIO: Handler("aspect_ratio", self._aspect_ratio),
            ImageStats.DIMENSION_DEPTH: Handler("depth", self._depth, ViewKind.WHOLE | ViewKind.BAND),
            ImageStats.DIMENSION_CENTER: Handler("center", self._center),
            ImageStats.DIMENSION_DISTANCE_CENTER: Handler("distance_center", self._distance_center),
            ImageStats.DIMENSION_DISTANCE_EDGE: Handler("distance_edge", self._distance_edge),
            ImageStats.DIMENSION_INVALID_BOX: Handler("invalid_box", self._invalid_box),
        }
