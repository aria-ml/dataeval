"""
Utility functions for preprocessing images and bounding boxes.
"""

__all__ = [
    "BoundingBox",
    "BoundingBoxFormat",
    "normalize_image_shape",
    "to_canonical_grayscale",
    "resize",
    "clip_and_pad",
    "rescale",
    "get_bitdepth",
    "edge_filter",
]

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import zoom
from scipy.signal import convolve2d

try:
    from PIL import Image
except ImportError:
    Image = None

_logger = logging.getLogger(__name__)

EDGE_KERNEL = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.int8)
BIT_DEPTH = (1, 8, 12, 16, 32)


# ===========================
# Bounding Box Classes
# ===========================


class BoundingBoxFormat(Enum):
    """Supported bounding box coordinate formats."""

    XYXY = "xyxy"
    XYWH = "xywh"
    CXCYWH = "cxcywh"
    YOLO = "yolo"


class BoundingBox:
    """
    A bounding box representation that supports multiple coordinate formats.

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
    bbox_format : BoundingBoxFormat, default BoundingBoxFormat.XYXY
        Input format of the coordinates
    image_shape : tuple[int, ...] or None, default None
        Shape of the image in CHW format

    Examples
    --------
    Create a bounding box in XYXY format:

    >>> bbox = BoundingBox(10, 20, 100, 150, bbox_format=BoundingBoxFormat.XYXY)
    >>> bbox.xyxy
    (10.0, 20.0, 100.0, 150.0)

    Convert to different formats:

    >>> bbox.xywh
    (10.0, 20.0, 90.0, 130.0)
    >>> bbox.cxcywh
    (55.0, 85.0, 90.0, 130.0)

    With image shape for YOLO format:

    >>> bbox = BoundingBox(0.5, 0.5, 0.2, 0.3, bbox_format=BoundingBoxFormat.YOLO, image_shape=(3, 224, 224))
    >>> bbox.xyxy
    (89.6, 78.4, 134.4, 145.6)
    """

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
        self._image_shape = image_shape
        v1, v2, v3, v4 = float(v1), float(v2), float(v3), float(v4)

        # Convert input to internal XYXY format
        if bbox_format == BoundingBoxFormat.XYXY:
            if v1 > v3 or v2 > v4:
                _logger.warning(f"Invalid bounding box coordinates: {(v1, v2, v3, v4)} - swapping invalid coordinates.")
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


IntBox = tuple[int, int, int, int]
"""Bounding box as tuple of integers in xyxy format."""

FloatBox = tuple[float, float, float, float]
"""Bounding box as tuple of floats in xyxy format."""

Box = IntBox | FloatBox
BoxLike = BoundingBox | Box | Iterable[int | float] | None


def to_bounding_box(boxlike: BoxLike, image_shape: tuple[int, ...] | None = None) -> BoundingBox:
    """
    Converts a box-like input to a BoundingBox instance.

    Parameters
    ----------
    boxlike : BoxLike
        Box-like object to convert
    image_shape : tuple[int, ...] or None, default None
        Shape of the image in CHW format

    Returns
    -------
    BoundingBox
        BoundingBox instance
    """
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
        _logger.warning(f"Invalid bounding box format: {boxlike}. Expected a BoundingBox or a tuple/list of 4 numbers.")

    return BoundingBox(0, 0, 0, 0, image_shape=image_shape)


def to_int_box(box: Box) -> IntBox:
    """
    Converts a bounding box from float to int format.

    Parameters
    ----------
    box : Box
        Bounding box in XYXY format

    Returns
    -------
    IntBox
        Bounding box with integer coordinates
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

    Parameters
    ----------
    image_shape : tuple[int, ...]
        Image shape (supports CHW or HW format)
    box : Box
        Bounding box to clip

    Returns
    -------
    IntBox
        Clipped bounding box
    """
    x0, y0, x1, y1 = to_int_box(box)
    h, w = image_shape[-2:]

    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def is_valid_box(box: Box) -> bool:
    """
    Check if the box dimensions provided are valid (non-empty).

    Parameters
    ----------
    box : Box
        Bounding box to validate

    Returns
    -------
    bool
        True if box is valid, False otherwise
    """
    return box[2] > box[0] and box[3] > box[1]


# ===========================
# Image Processing Functions
# ===========================


@dataclass
class BitDepth:
    """
    Dataclass representing image bit depth information.

    Attributes
    ----------
    depth : int
        Bit depth (1, 8, 12, 16, or 32)
    pmin : float or int
        Minimum pixel value
    pmax : float or int
        Maximum pixel value
    """

    depth: int
    pmin: float | int
    pmax: float | int


def get_bitdepth(image: NDArray[Any]) -> BitDepth:
    """
    Approximates the bit depth of the image using the min and max pixel values.

    Parameters
    ----------
    image : NDArray
        Input image array

    Returns
    -------
    BitDepth
        Bit depth information
    """
    pmin, pmax = np.nanmin(image), np.nanmax(image)
    if pmin < 0:
        return BitDepth(0, pmin, pmax)
    depth = ([x for x in BIT_DEPTH if 2**x > pmax] or [max(BIT_DEPTH)])[0]
    return BitDepth(depth, 0, 2**depth - 1)


def rescale(image: NDArray[Any], depth: int = 1) -> NDArray[Any]:
    """
    Rescales the image using the bit depth provided.

    Parameters
    ----------
    image : NDArray
        Input image array
    depth : int, default 1
        Target bit depth

    Returns
    -------
    NDArray
        Rescaled image
    """
    bitdepth = get_bitdepth(image)
    if bitdepth.depth == depth:
        return image
    normalized = (image + bitdepth.pmin) / (bitdepth.pmax - bitdepth.pmin)
    return normalized * (2**depth - 1)


def normalize_image_shape(image: NDArray[Any]) -> NDArray[Any]:
    """
    Normalizes the image shape into (C,H,W) format.

    Parameters
    ----------
    image : NDArray
        Input image array

    Returns
    -------
    NDArray
        Image in CHW format

    Raises
    ------
    ValueError
        If image has less than 2 dimensions
    """
    ndim = image.ndim
    if ndim == 2:
        return np.expand_dims(image, axis=0)
    if ndim == 3:
        return image
    if ndim > 3:
        # Slice all but the last 3 dimensions
        return image[(0,) * (ndim - 3)]
    raise ValueError("Images must have 2 or more dimensions.")


def edge_filter(image: NDArray[Any], offset: float = 0.5) -> NDArray[np.uint8]:
    """
    Returns the image filtered using a 3x3 edge detection kernel.

    The kernel used is:
        [[ -1, -1, -1 ],
         [ -1,  8, -1 ],
         [ -1, -1, -1 ]]

    Parameters
    ----------
    image : NDArray
        Input image array (2D)
    offset : float, default 0.5
        Offset to add after convolution

    Returns
    -------
    NDArray[np.uint8]
        Edge-filtered image
    """
    edges = convolve2d(image, EDGE_KERNEL, mode="same", boundary="symm") + offset
    np.clip(edges, 0, 255, edges)
    return edges


def clip_and_pad(image: NDArray[Any], box: Box) -> NDArray[Any]:
    """
    Extract a region from an image based on a bounding box, clipping to image boundaries
    and padding out-of-bounds areas with np.nan.

    Parameters
    ----------
    image : NDArray
        Input image array in format C, H, W (channels first)
    box : Box
        Bounding box coordinates as (x0, y0, x1, y1) where (x0, y0) is top-left
        and (x1, y1) is bottom-right

    Returns
    -------
    NDArray
        The extracted region with out-of-bounds areas padded with np.nan
    """
    # Create output array filled with NaN with a minimum size of 1x1
    box = to_int_box(box)
    bw, bh = max(1, box[2] - box[0]), max(1, box[3] - box[1])

    output = np.full((image.shape[-3] if image.ndim > 2 else 1, bh, bw), np.nan)

    # Calculate source box
    sbox = clip_box(image.shape, box)

    # Calculate destination box
    x0, y0 = sbox[0] - box[0], sbox[1] - box[1]
    x1, y1 = x0 + (sbox[2] - sbox[0]), y0 + (sbox[3] - sbox[1])

    # Copy the source if valid from the image to the output
    if is_valid_box(sbox):
        output[:, y0:y1, x0:x1] = image[:, sbox[1] : sbox[3], sbox[0] : sbox[2]]

    return output


def resize(image: NDArray[np.uint8], resize_dim: int, use_pil: bool = True) -> NDArray[np.uint8]:
    """
    Resizes a grayscale (HxW) 8-bit image using PIL or scipy.ndimage.zoom.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Grayscale image to resize
    resize_dim : int
        Target dimension (output will be resize_dim x resize_dim)
    use_pil : bool, default True
        Whether to use PIL for resizing (if available)

    Returns
    -------
    NDArray[np.uint8]
        Resized image
    """
    # Use PIL if available, otherwise resize and resample with scipy.ndimage.zoom
    if use_pil and Image is not None:
        return np.array(Image.fromarray(image).resize((resize_dim, resize_dim), Image.Resampling.LANCZOS))

    zoom_factors = (resize_dim / image.shape[0], resize_dim / image.shape[1])
    return np.clip(np.asarray(zoom(image, zoom_factors, order=5, mode="reflect")), 0, 255, dtype=np.uint8)


def to_canonical_grayscale(image: NDArray[Any]) -> NDArray[np.uint8]:
    """
    Converts an image of arbitrary channels (CHW) to a single-channel
    uint8 grayscale image (HW) using color-space-aware heuristics.

    Parameters
    ----------
    image : NDArray
        Input array in CHW format

    Returns
    -------
    NDArray[np.uint8]
        2D grayscale array (HW) of type np.uint8
    """
    channels = image.shape[0]

    # --- Case 1: Single Channel (Already Grayscale) ---
    if channels == 1:
        return image[0].astype(np.uint8)

    # --- Case 2: RGB (3 Channels) ---
    if channels == 3:
        # Rec. 601 Luma coefficients
        weights = np.array([0.299, 0.587, 0.114]).reshape(3, 1, 1)
        grayscale = np.sum(image.astype(float) * weights, axis=0)
        return np.clip(grayscale, 0, 255).astype(np.uint8)

    # --- Case 3: 4 Channels (RGBA or CMYK) ---
    if channels == 4:
        # Statistical heuristic to detect CMYK vs RGBA
        # Sample pixels for efficiency
        sample = image[:, ::4, ::4].reshape(4, -1).astype(float)
        c4_mean = np.mean(sample[3])
        c4_std = np.std(sample[3])

        # Heuristic: CMYK 'K' channel usually has high variance and detail.
        # RGBA 'Alpha' is usually mostly 255 (opaque) or 0 (transparent).
        # If std dev is high and mean isn't pinned to the extremes, guess CMYK.
        is_cmyk = c4_std > 35 and (40 < c4_mean < 215)

        if is_cmyk:
            # CMYK to RGB (Subtractive)
            c, m, y, k = image.astype(float) / 255.0
            r = 255 * (1 - c) * (1 - k)
            g = 255 * (1 - m) * (1 - k)
            b = 255 * (1 - y) * (1 - k)
            # Convert resulting RGB to Grayscale
            grayscale = (0.299 * r) + (0.587 * g) + (0.114 * b)
        else:
            # RGBA to RGB (Composite over White background)
            rgb_raw = image[:3].astype(float)
            alpha = image[3].astype(float) / 255.0
            # Composite formula: Source * Alpha + Background * (1 - Alpha)
            r = (rgb_raw[0] * alpha) + (255.0 * (1 - alpha))
            g = (rgb_raw[1] * alpha) + (255.0 * (1 - alpha))
            b = (rgb_raw[2] * alpha) + (255.0 * (1 - alpha))
            grayscale = (0.299 * r) + (0.587 * g) + (0.114 * b)

        return np.clip(grayscale, 0, 255).astype(np.uint8)

    # --- Case 4: Arbitrary Channels (Fallback) ---
    # For 2, 5, or more channels, we simply average all information.
    return np.mean(image, axis=0).astype(np.uint8)
