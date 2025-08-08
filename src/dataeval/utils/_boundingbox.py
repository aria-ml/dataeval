from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from enum import Enum


class BoundingBoxFormat(Enum):
    XYXY = "xyxy"
    XYWH = "xywh"
    CXCYWH = "cxcywh"
    YOLO = "yolo"


class BoundingBox:
    def __init__(
        self,
        v1: float,
        v2: float,
        v3: float,
        v4: float,
        *,
        bbox_format: BoundingBoxFormat = BoundingBoxFormat.XYXY,
        image_shape: tuple[int, ...] | None = None,
    ) -> None:
        """Initialize bounding box.

        Parameters
        ----------
        v1 : float
            First coordinate value
        v2 : float
            Second coordinate value
        v3 : float
            Third coordinate value
        v4 : float
            Fourth coordinate value
        format : BBoxFormat, default BBoxFormat.XYXY
            Input format of the coordinates
        image_shape : tuple[int, ...] or None, default None
            Shape of the image in CHW format
        """
        self._image_shape = image_shape

        # Convert input to internal XYXY format
        if bbox_format == BoundingBoxFormat.XYXY:
            if v1 > v3 or v2 > v4:
                warnings.warn(f"Invalid bounding box coordinates: {(v1, v2, v3, v4)} - swapping invalid coordinates.")
            self._x0 = min(v1, v3)
            self._y0 = min(v2, v4)
            self._x1 = max(v1, v3)
            self._y1 = max(v2, v4)
        elif bbox_format == BoundingBoxFormat.XYWH:
            self._x0, self._y0 = v1, v2
            self._x1, self._y1 = v1 + v3, v2 + v4
        elif bbox_format == BoundingBoxFormat.CXCYWH:
            center_x, center_y, w, h = v1, v2, v3, v4
            self._x0 = center_x - w / 2
            self._y0 = center_y - h / 2
            self._x1 = center_x + w / 2
            self._y1 = center_y + h / 2
        elif bbox_format == BoundingBoxFormat.YOLO:
            h, w = self.image_hw
            center_x, center_y, w, h = v1 * w, v2 * h, v3 * w, v4 * h
            self._x0 = center_x - w / 2
            self._y0 = center_y - h / 2
            self._x1 = center_x + w / 2
            self._y1 = center_y + h / 2
        else:
            raise ValueError(f"Unknown format: {bbox_format}")

    @property
    def x0(self) -> float:
        """Get x0 coordinate."""
        return self._x0

    @property
    def y0(self) -> float:
        """Get y0 coordinate."""
        return self._y0

    @property
    def x1(self) -> float:
        """Get x1 coordinate."""
        return self._x1

    @property
    def y1(self) -> float:
        """Get y1 coordinate."""
        return self._y1

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Get coordinates in XYXY format (x0, y0, x1, y1)."""
        return (self._x0, self._y0, self._x1, self._y1)

    @property
    def xyxy_int(self) -> tuple[int, int, int, int]:
        """Get coordinates in XYXY format as int (x0, y0, x1, y1)."""
        return math.floor(self._x0), math.floor(self._y0), math.ceil(self._x1), math.ceil(self._y1)

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        """Get coordinates in XYWH format (x, y, width, height)."""
        return (self._x0, self._y0, self._x1 - self._x0, self._y1 - self._y0)

    @property
    def cxcywh(self) -> tuple[float, float, float, float]:
        """Get coordinates in CXCYWH format (center_x, center_y, width, height)."""
        center_x = (self._x0 + self._x1) / 2
        center_y = (self._y0 + self._y1) / 2
        width = self._x1 - self._x0
        height = self._y1 - self._y0
        return (center_x, center_y, width, height)

    @property
    def yolo(self) -> tuple[float, float, float, float]:
        """Get coordinates in YOLO format (center_x, center_y, width, height) normalized to [0, 1]."""
        h, w = self.image_hw
        center_x = (self._x0 + self._x1) / 2 / w
        center_y = (self._y0 + self._y1) / 2 / h
        width = (self._x1 - self._x0) / w
        height = (self._y1 - self._y0) / h
        return (center_x, center_y, width, height)

    @property
    def width(self) -> float:
        """Get bounding box width."""
        return self._x1 - self._x0

    @property
    def height(self) -> float:
        """Get bounding box height."""
        return self._y1 - self._y0

    @property
    def image_hw(self) -> tuple[int, int]:
        """Get image height and width."""
        if self._image_shape is None:
            raise ValueError("Image shape is required for bounds checking and YOLO format.")

        return self._image_shape[-2], self._image_shape[-1]

    def area(self) -> float:
        """Calculate bounding box area."""
        return (self._x1 - self._x0) * (self._y1 - self._y0)

    def center(self) -> tuple[float, float]:
        """Get center coordinates (x, y)."""
        return ((self._x0 + self._x1) / 2, (self._y0 + self._y1) / 2)

    def is_inside(self) -> bool:
        """Check if bounding box is within image bounds."""
        h, w = self.image_hw
        return self._x0 >= 0 and self._y0 >= 0 and self._x1 <= w and self._y1 <= h

    def is_outside(self) -> bool:
        """Check if bounding box is outside image bounds."""
        h, w = self.image_hw
        return self._x1 <= 0 or self._y1 <= 0 or self._x0 >= w or self._y0 >= h

    def is_partial(self) -> bool:
        """Check if bounding box is partially inside image bounds."""
        return not self.is_inside() and not self.is_outside()

    def is_valid(self) -> bool:
        """Check if bounding box is valid (not empty)."""
        return self._x0 < self._x1 and self._y0 < self._y1 and not self.is_outside()

    def is_clippable(self) -> bool:
        """Check if bounding box can be clipped to image bounds."""
        return is_valid_box(clip_box(self.image_hw, self.xyxy_int))

    @classmethod
    def from_boxlike(cls, boxlike: BoxLike, image_shape: tuple[int, ...] | None = None) -> BoundingBox:
        if isinstance(boxlike, BoundingBox):
            return boxlike
        try:
            if isinstance(boxlike, tuple | list) and len(boxlike) == 4:
                return BoundingBox(boxlike[0], boxlike[1], boxlike[2], boxlike[3], image_shape=image_shape)
            if isinstance(boxlike, Iterable):
                return BoundingBox(*boxlike, image_shape=image_shape)
            if isinstance(image_shape, tuple) and len(image_shape) > 2:
                return BoundingBox(0, 0, image_shape[-2], image_shape[-1], image_shape=image_shape)
        except (TypeError, ValueError):
            warnings.warn(
                f"Invalid bounding box format: {boxlike}. Expected a BoundingBox or a tuple/list of 4 numbers."
            )

        return BoundingBox(0, 0, 0, 0, image_shape=image_shape)


IntBox = tuple[int, int, int, int]
"""Bounding box as tuple of integers in xyxy format."""

FloatBox = tuple[float, float, float, float]
"""Bounding box as tuple of floats in xyxy format."""

Box = IntBox | FloatBox
BoxLike = BoundingBox | Box | Iterable[int | float] | None


def to_int_box(box: Box) -> IntBox:
    """
    Converts a bounding box from float to int format.
    """
    return (
        int(math.floor(box[0])),
        int(math.floor(box[1])),
        int(math.ceil(box[2])),
        int(math.ceil(box[3])),
    )


def clip_box(image_shape: tuple[int, ...], box: Box) -> IntBox:
    """
    Clip the box to inside the provided image dimensions.
    """
    x0, y0, x1, y1 = to_int_box(box)
    h, w = image_shape[-2:]

    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def is_valid_box(box: Box) -> bool:
    """
    Check if the box dimensions provided are a valid image.
    """
    return box[2] > box[0] and box[3] > box[1]
