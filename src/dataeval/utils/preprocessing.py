"""Utility functions for preprocessing images and bounding boxes."""

__all__ = [
    "BitDepth",
    "BoundingBox",
    "BoundingBoxFormat",
    "Box",
    "BoxLike",
    "ChannelGroup",
    "ChannelGroupLike",
    "FloatBox",
    "IntBox",
    "ValueRange",
    "boxes_to_mask",
    "clip_box",
    "compute_iou",
    "crop_with_fill",
    "edge_filter",
    "get_bitdepth",
    "get_value_range",
    "is_valid_box",
    "normalize_image_shape",
    "rescale",
    "resize",
    "to_bounding_box",
    "to_canonical_grayscale",
    "to_channel_group",
    "to_int_box",
]

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray
from scipy.ndimage import zoom
from scipy.signal import convolve2d

from dataeval._experimental import deprecated
from dataeval._log import get_logger
from dataeval.exceptions import ShapeMismatchError

try:
    from PIL import Image
except ImportError:
    Image = None

_logger = get_logger(__name__)

_EDGE_KERNEL = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.int8)
_BIT_DEPTH = (1, 8, 12, 16, 32)
_EIGHT_BIT_MAX = 255


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

    def __init__(  # noqa: C901
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
        """The x0 coordinate."""
        return self._x0

    @property
    def y0(self) -> float:
        """The y0 coordinate."""
        return self._y0

    @property
    def x1(self) -> float:
        """The x1 coordinate."""
        return self._x1

    @property
    def y1(self) -> float:
        """The y1 coordinate."""
        return self._y1

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Coordinates in XYXY format (x0, y0, x1, y1)."""
        return (self._x0, self._y0, self._x1, self._y1)

    @property
    def xyxy_int(self) -> tuple[int, int, int, int]:
        """Coordinates in XYXY format as int (x0, y0, x1, y1)."""
        return math.floor(self._x0), math.floor(self._y0), math.ceil(self._x1), math.ceil(self._y1)

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        """Coordinates in XYWH format (x, y, width, height)."""
        return (self._x0, self._y0, self._x1 - self._x0, self._y1 - self._y0)

    @property
    def cxcywh(self) -> tuple[float, float, float, float]:
        """Coordinates in CXCYWH format (center_x, center_y, width, height)."""
        center_x = (self._x0 + self._x1) / 2
        center_y = (self._y0 + self._y1) / 2
        width = self._x1 - self._x0
        height = self._y1 - self._y0
        return (center_x, center_y, width, height)

    @property
    def yolo(self) -> tuple[float, float, float, float]:
        """Coordinates in YOLO format (center_x, center_y, width, height) normalized to [0, 1]."""
        h, w = self.image_hw
        center_x = (self._x0 + self._x1) / 2 / w
        center_y = (self._y0 + self._y1) / 2 / h
        width = (self._x1 - self._x0) / w
        height = (self._y1 - self._y0) / h
        return (center_x, center_y, width, height)

    @property
    def width(self) -> float:
        """Bounding box width."""
        return self._x1 - self._x0

    @property
    def height(self) -> float:
        """Bounding box height."""
        return self._y1 - self._y0

    @property
    def image_hw(self) -> tuple[int, int]:
        """Image height and width."""
        if self._image_shape is None:
            raise ValueError("Image shape is required for bounds checking and YOLO format.")

        return self._image_shape[-2], self._image_shape[-1]

    def area(self) -> float:
        """Compute bounding box area."""
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


def to_bounding_box(boxlike: BoxLike, image_shape: tuple[int, ...] | None = None) -> BoundingBox:  # noqa: C901
    """
    Convert a box-like input to a BoundingBox instance.

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
    Convert a bounding box from float to int format.

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


def boxes_to_mask(image_shape: tuple[int, ...], boxes: Iterable[BoxLike]) -> NDArray[np.bool_]:
    r"""
    Paint a set of bounding boxes into a boolean coverage mask.

    Produces the union of the boxes as a per-pixel mask over the image's spatial
    extent, which is what separates an image's annotated foreground from the rest
    of the scene behind it.

    Parameters
    ----------
    image_shape : tuple[int, ...]
        Shape of the image the boxes belong to (supports CHW or HW format; only the
        trailing two dimensions are read).
    boxes : Iterable[BoxLike]
        Boxes to paint. Each is converted with :func:`to_bounding_box`, rounded
        outwards to whole pixels (:attr:`BoundingBox.xyxy_int`, matching how
        :func:`crop_with_fill` windows a box) and clipped to the image. Boxes that
        are degenerate or fall entirely outside the image contribute nothing.
        Overlapping boxes are unioned, so a pixel covered twice is still one pixel.

    Returns
    -------
    NDArray[np.bool\_]
        A ``(H, W)`` array that is True wherever at least one box covers the pixel.
        Channels are deliberately absent: box geometry is spatial, and a caller that
        needs to apply the mask across channels broadcasts it.

    Notes
    -----
    Rounding outwards means the mask covers *at least* every pixel a box touches, so
    using it to exclude foreground over-excludes rather than under-excludes. That is
    the conservative direction for background analysis — a retained pixel is
    background with high confidence, at the cost of discarding a boundary ring of
    genuine background along with the object.

    That choice is not shared by :class:`~dataeval.data.DetectionCrops`, whose
    ``region="surround"`` masks the object out with round-to-nearest instead, matching
    the crop window it is masking within. The two therefore disagree by up to a
    one-pixel ring: a background measured by ``compute_stats(per_background=True)`` and
    one embedded from ``DetectionCrops(region="surround")`` are not the same region, and
    a comparison between them carries that difference.

    Examples
    --------
    Two overlapping boxes over a 4x4 image:

    >>> mask = boxes_to_mask((3, 4, 4), [(0, 0, 2, 2), (1, 1, 3, 3)])
    >>> mask.astype(int)
    array([[1, 1, 0, 0],
           [1, 1, 1, 0],
           [0, 1, 1, 0],
           [0, 0, 0, 0]])

    A box reaching past the image edge is clipped rather than rejected:

    >>> boxes_to_mask((2, 2), [(1, 1, 99, 99)]).astype(int)
    array([[0, 0],
           [0, 1]])
    """
    height, width = image_shape[-2], image_shape[-1]
    mask = np.zeros((height, width), dtype=np.bool_)
    for boxlike in boxes:
        box = to_bounding_box(boxlike, image_shape=image_shape)
        x0, y0, x1, y1 = clip_box(image_shape, box.xyxy_int)
        if is_valid_box((x0, y0, x1, y1)):
            mask[y0:y1, x0:x1] = True
    return mask


def compute_iou(boxes1: NDArray[Any], boxes2: NDArray[Any]) -> NDArray[np.float64]:
    """
    Compute Intersection over Union (IoU) between two sets of boxes.

    Parameters
    ----------
    boxes1 : NDArray[Any]
        Boxes of shape (N, 4) in XYXY format.
    boxes2 : NDArray[Any]
        Boxes of shape (M, 4) in XYXY format.

    Returns
    -------
    NDArray[np.float64]
        IoU matrix of shape (N, M).
    """
    # Ensure 2D arrays
    if boxes1.ndim == 1:
        boxes1 = boxes1[np.newaxis, :]
    if boxes2.ndim == 1:
        boxes2 = boxes2[np.newaxis, :]

    # Extract coordinates
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    # Compute intersection coordinates
    x_a = np.maximum(x11, np.transpose(x21))
    y_a = np.maximum(y11, np.transpose(y21))
    x_b = np.minimum(x12, np.transpose(x22))
    y_b = np.minimum(y12, np.transpose(y22))

    # Compute intersection area
    inter_area = np.maximum(0, x_b - x_a) * np.maximum(0, y_b - y_a)

    # Compute individual areas
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)

    # Compute IoU
    union_area = box1_area + np.transpose(box2_area) - inter_area
    return np.divide(inter_area, union_area, out=np.zeros_like(inter_area, dtype=np.float64), where=union_area > 0)


# ===========================
# Image Processing Functions
# ===========================


@dataclass
class BitDepth:
    """
    Dataclass representing image bit depth information.

    .. deprecated:: 1.1
        Use :class:`ValueRange`, which separates the interval statistics are computed
        against from the encoding depth it was decoded from. Will be removed in v1.2.

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


@dataclass(frozen=True)
class ValueRange:
    """
    The interval a datum's values occupy, and the encoding depth that interval was decoded from.

    Two separate things, which :class:`BitDepth` conflated. **Encoding depth** is how the
    data was stored, and only exists where there is an encoding to decode — integer image
    formats genuinely are power-of-two, so reading one off is a decode rather than a
    guess. **Value range** is the interval :func:`rescale` divides by and a histogram bins
    over, and is what every consumer actually needs.

    Floating point data has no encoding to decode, so outside the two conventional float
    spellings of an ordinary image its depth is ``nan`` and its interval must be declared
    — see :func:`get_value_range`.

    Attributes
    ----------
    depth : float
        Bit depth the interval was decoded from — one of 1, 8, 12, 16 or 32 — or ``nan``
        where the values carry no encoding to read. Reported as
        :attr:`~dataeval.flags.ImageStats.DIMENSION_DEPTH`.
    pmin : float
        Lower bound of the interval, or ``nan`` when no interval could be established.
    pmax : float
        Upper bound of the interval, or ``nan`` when no interval could be established.

    See Also
    --------
    get_value_range : establish the range for an array
    rescale : the interval's principal consumer

    Examples
    --------
    An 8-bit image's interval is its encoding's, not the pixels' own extremes:

    >>> get_value_range(np.array([[0, 100]], dtype=np.uint8))
    ValueRange(depth=8, pmin=0.0, pmax=255.0)

    Float data outside the conventional spellings has neither:

    >>> get_value_range(np.array([[-50.0, 50.0]]))
    ValueRange(depth=nan, pmin=nan, pmax=nan)
    """

    depth: float
    pmin: float
    pmax: float

    @property
    def is_known(self) -> bool:
        """Whether an interval was established, and statistics needing one can be computed."""
        return not (math.isnan(self.pmin) or math.isnan(self.pmax))

    @classmethod
    def observed(cls, values: NDArray[Any]) -> "ValueRange":
        """
        Read the interval the values themselves span, carrying no encoding depth.

        An explicit request for a per-array stretch. Use it where each array is meant to
        be normalized against its own extremes — as a feature extractor does, since a
        descriptor reads contrast rather than absolute level — and *not* where arrays are
        to be compared against each other, which is what an inferred or declared range
        exists for.

        Parameters
        ----------
        values : NDArray
            Array to read the extremes off. NaNs are skipped.

        Returns
        -------
        ValueRange
            The observed interval with ``depth`` of ``nan``, or an unknown range when
            `values` is empty or entirely NaN.
        """
        if values.size == 0 or np.isnan(values).all():
            return cls(np.nan, np.nan, np.nan)
        return cls(np.nan, float(np.nanmin(values)), float(np.nanmax(values)))


#: The answer where no interval could be established — see :func:`get_value_range` Notes.
_UNKNOWN_RANGE = ValueRange(np.nan, np.nan, np.nan)


@deprecated(since="1.1", removal="1.2", alternative="get_value_range")
def get_bitdepth(image: NDArray[Any]) -> BitDepth:
    """
    Approximates the bit depth of the image using the min and max pixel values.

    .. deprecated:: 1.1
        Use :func:`get_value_range`, which does not fabricate a depth for float data and
        does not discard the observed range of data holding negative values. Will be
        removed in v1.2.

    Parameters
    ----------
    image : NDArray
        Input image array

    Returns
    -------
    BitDepth
        Bit depth information
    """
    if image.size == 0 or np.isnan(image).all():
        return BitDepth(0, np.nan, np.nan)
    pmin, pmax = np.nanmin(image), np.nanmax(image)
    if pmin < 0:
        return BitDepth(0, pmin, pmax)
    depth = ([x for x in _BIT_DEPTH if 2**x > pmax] or [max(_BIT_DEPTH)])[0]
    return BitDepth(depth, 0, 2**depth - 1)


def get_value_range(values: NDArray[Any], *, declared: tuple[float, float] | None = None) -> ValueRange:
    """
    Establish the interval an array's values are measured against.

    Parameters
    ----------
    values : NDArray
        Array to establish the range for. NaNs are skipped.
    declared : tuple[float, float] or None, default None
        The interval, stated by the caller as ``(low, high)``. Takes precedence over
        every inference below: a declaration is the caller saying these are physical
        values with a known span, not an encoded image, so no depth is implied from it.

    Returns
    -------
    ValueRange
        The interval and — where the data was encoded rather than measured — the depth it
        was decoded from. An unknown range (every field ``nan``) where none could be
        established; see Notes.

    Raises
    ------
    ValueError
        If `declared` is not an ordered pair of finite numbers.

    Notes
    -----
    Resolved in order, first match winning:

    1. `declared`, if given. Depth is ``nan`` — a declaration carries no encoding.
    2. Empty or all-NaN input: unknown.
    3. Non-negative integer dtype: the power-of-two encoding is **decoded** from the
       maximum. This is what makes two images comparable — every 8-bit image divides by
       255 rather than by its own brightest pixel.
    4. Float within ``[0, 1]``: already normalized, so the interval is ``[0, 1]``.
    5. Float within ``[0, 255]``: the float spelling of an 8-bit image — what a
       ``ToTensor``-style pipeline or a resize leaves behind — so the interval is
       ``[0, 255]``.
    6. Anything else: **unknown**. Float data spanning more than 255, and any data
       holding negative values, carries no encoding to decode and no convention to
       fall back on.

    Cases 4 and 5 are the two ordinary float spellings of visible imagery and are
    deliberately not fabrications: both are conventions with one reading. Their order
    matters — testing ``[0, 1]`` first means a binary mask reads as normalized rather
    than as a degenerate 8-bit image.

    Case 6 is the honest answer for elevation below sea level, mean-centred reflectance,
    temperature in Celsius, or any band whose dynamic range is a property of the sensor
    rather than of a file format. Declare the interval for these; the alternative is a
    number derived from an arbitrary maximum, which looks like a measurement and is not.

    Examples
    --------
    An integer encoding is decoded, so a dark image still scales by its format's range:

    >>> get_value_range(np.array([[0, 3000]], dtype=np.uint16))
    ValueRange(depth=12, pmin=0.0, pmax=4095.0)

    A declaration wins, and implies no depth:

    >>> get_value_range(np.array([[-50.0, 50.0]]), declared=(-100.0, 100.0))
    ValueRange(depth=nan, pmin=-100.0, pmax=100.0)
    """
    if declared is not None:
        low, high = _validate_declared_range(declared)
        return ValueRange(np.nan, low, high)

    if values.size == 0 or np.isnan(values).all():
        return _UNKNOWN_RANGE

    pmin, pmax = float(np.nanmin(values)), float(np.nanmax(values))
    if pmin < 0:
        # Neither a power-of-two decode nor either float convention covers a signed span.
        return _UNKNOWN_RANGE
    return _decode_range(values.dtype, pmax)


def _decode_range(dtype: DTypeLike, pmax: float) -> ValueRange:
    """Read the encoding off non-negative values, or answer unknown where there is none."""
    if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
        depth = ([x for x in _BIT_DEPTH if 2**x > pmax] or [max(_BIT_DEPTH)])[0]
        return ValueRange(depth, 0.0, float(2**depth - 1))
    # Float: the two conventional spellings of an ordinary image, [0, 1] first so a
    # binary mask reads as normalized rather than as a degenerate 8-bit image.
    if pmax <= 1.0:
        return ValueRange(1, 0.0, 1.0)
    if pmax <= _EIGHT_BIT_MAX:
        return ValueRange(8, 0.0, float(_EIGHT_BIT_MAX))
    return _UNKNOWN_RANGE


def _validate_declared_range(declared: tuple[float, float]) -> tuple[float, float]:
    """Check a caller-declared interval, which nothing downstream can sanity-check for itself.

    Unpacked rather than indexed so that a longer sequence is rejected rather than
    silently truncated to its first two entries — ``(0, 1, 99)`` is a caller who meant
    something this function cannot represent, not a ``(0, 1)`` with a typo after it.
    """
    try:
        low_raw, high_raw = declared
        low, high = (float(low_raw), float(high_raw))
    except (TypeError, ValueError, IndexError, KeyError) as e:
        raise ValueError(f"value range must be a (low, high) pair of numbers; got {declared!r}.") from e
    if not (math.isfinite(low) and math.isfinite(high)) or low >= high:
        raise ValueError(f"value range must be a finite, ordered (low, high) pair; got {declared!r}.")
    return low, high


class ChannelGroup:
    """
    A named set of an image's bands, measured jointly.

    The unit a band-wise statistic is computed over. ``ChannelGroup([0, 1, 2])`` on an
    RGB+NIR image reduces over the three visible bands *together* — the ordinary
    all-channel behavior, restricted to a subset — so the group's mean is one number
    describing the visible part rather than three describing each band. That is what makes
    it scale: a 224-band cube is asked about as three band groups, not 224 columns.

    Most callers never construct one. A bare index, sequence or range passed to
    :func:`~dataeval.core.compute_stats`'s ``channels`` is converted automatically;
    reach for the class when a group needs `value_range`.

    Parameters
    ----------
    indices : int or Sequence[int] or range
        Which bands belong to the group, as zero-based indices into the channel axis.
        Order is irrelevant — a group is a set being reduced over, not a rearrangement,
        so the indices are stored ascending however they were written — and repeats are
        rejected, since a repeated index would silently double-weight that band in every
        reduction.
    value_range : tuple[float, float] or None, default None
        The interval this group's values should be measured against, as ``(low, high)``.

        Bands of one cube are different measurements with different dynamic ranges, and
        stacking them into a single array leaves the dtype describing none of them
        individually. So a group whose values carry no encoding to decode — a reflectance,
        elevation or temperature band — declares its interval here, independently of its
        neighbours. Leave as None for bands that are ordinary image data; see
        :func:`get_value_range`.

    Raises
    ------
    ValueError
        If `indices` is empty, holds a repeat, or holds anything but non-negative
        integers; or if `value_range` is not a finite, ordered pair.

    See Also
    --------
    :class:`~dataeval.data.SelectChannels` : narrow a whole dataset to chosen bands

    Notes
    -----
    Deliberately not the same type as ``SelectChannels``' channel selection, which shares
    the spelling but not the semantics. That one permits repeats, accepts ``"gray"`` as a
    luminance *mix* rather than a slice, and rejects any channel count but 1 or 3 — all
    correct for producing a transformed image, all wrong for naming a set of bands to
    reduce over.

    A group is all-or-nothing. Where an image cannot supply every index the group names,
    every statistic for that group is NaN rather than reduced over the bands that are
    present — one column name has to mean one thing, and a datum missing bands it should
    have is a defect that should read as absent.

    Examples
    --------
    >>> ChannelGroup(3)
    ChannelGroup((3,))

    >>> ChannelGroup(range(30, 70))
    ChannelGroup((30, 31, ..., 69))

    A band whose range is the sensor's rather than a file format's:

    >>> ChannelGroup([4], value_range=(-40.0, 60.0))
    ChannelGroup((4,), value_range=(-40.0, 60.0))
    """

    def __init__(
        self,
        indices: int | Sequence[int | np.integer] | range | NDArray[Any],
        *,
        value_range: tuple[float, float] | None = None,
    ) -> None:
        self.indices: tuple[int, ...] = _validate_channel_indices(indices)
        self.value_range: tuple[float, float] | None = (
            None if value_range is None else _validate_declared_range(value_range)
        )

    def __repr__(self) -> str:
        """Render the group, eliding a long index run to its endpoints."""
        indices = (
            f"({self.indices[0]}, {self.indices[1]}, ..., {self.indices[-1]})"
            if len(self.indices) > 4
            else repr(self.indices)
        )
        suffix = "" if self.value_range is None else f", value_range={self.value_range!r}"
        return f"{type(self).__name__}({indices}{suffix})"

    def __eq__(self, other: object) -> bool:
        """Compare by bands and declared range; order of the indices is not part of identity."""
        if not isinstance(other, ChannelGroup):
            return NotImplemented
        return self.indices == other.indices and self.value_range == other.value_range

    def __hash__(self) -> int:
        """Hash the bands and declared range, matching :meth:`__eq__`."""
        return hash((self.indices, self.value_range))


ChannelGroupLike = int | Sequence[int | np.integer] | range | NDArray[Any] | ChannelGroup
"""What a channel group may be written as. Coerced by :func:`to_channel_group`."""


def _is_index(value: Any) -> bool:
    """Whether a value names a band.

    ``bool`` is rejected despite being an ``int`` subclass: numpy reads a list of bools as
    a mask rather than as indices, so ``[True, False, True]`` would silently select bands
    0 and 2 instead of the requested 1, 0, 1. ``np.bool_`` is rejected for the same reason,
    while ``np.integer`` is accepted — ``list(np.arange(3))`` holds ``np.int64``, and that
    is an ordinary way to spell a band selection.
    """
    return isinstance(value, int | np.integer) and not isinstance(value, bool | np.bool_)


def _normalize_index_candidates(indices: Any) -> Any:
    """Bring the accepted spellings of a band selection to one sequence form.

    A bare index becomes a one-element tuple, and a numpy array becomes a list —
    `np.arange(3)` and `np.where(...)[0]` are ordinary ways to spell a selection in a
    numpy-first library, and neither is a `Sequence`. Only a 1-D integer array names a set
    of bands; a float or 2-D array is a mistake worth reporting as one rather than coercing.
    """
    if _is_index(indices):
        return (indices,)
    if isinstance(indices, np.ndarray):
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError(
                f"channel indices given as an array must be 1-D and of integer dtype; got "
                f"{indices.ndim}-D {indices.dtype}."
            )
        return indices.tolist()
    return indices


def _validate_index_selection(indices: Any) -> tuple[int, ...]:
    """Check that a value names bands, and narrow it to built-in ints.

    The half two callers share. Repeats and order are *left alone* here, because the two
    disagree about them: `SelectChannels` reorders and duplicates bands deliberately, while
    a `ChannelGroup` is reduced over jointly and can do neither. Only what counts as an
    index at all is common, which is the part worth having one answer to.

    Narrowed to built-in ints so a group's stored tuple hashes and reprs the same however
    the caller spelled it — ``np.int64(3)`` and ``3`` name one band, not two groups.
    """
    candidates = _normalize_index_candidates(indices)
    if not isinstance(candidates, Sequence | range) or isinstance(candidates, str):
        raise ValueError(f"channel indices must be an int, a sequence of ints, or a range; got {indices!r}.")
    if not candidates or not all(_is_index(i) and i >= 0 for i in candidates):
        raise ValueError(f"channel indices must be a non-empty selection of non-negative ints; got {indices!r}.")
    return tuple(int(i) for i in candidates)


def _validate_channel_indices(indices: Any) -> tuple[int, ...]:
    """Check a band selection for a group, which nothing downstream can sanity-check itself."""
    resolved = _validate_index_selection(indices)
    if len(set(resolved)) != len(resolved):
        raise ValueError(
            f"channel indices must not repeat; got {indices!r}. A group is reduced over jointly, so a "
            "repeated band would be weighted twice in every statistic."
        )
    # Canonicalized so that order really is irrelevant, as the class documents. Kept as
    # written, the order would reach the band slice and so change the answer: a hash runs
    # a grayscale conversion over the bands in the order given, and equality and hashing
    # would call ``[0, 1, 2]`` and ``[2, 1, 0]`` two different groups.
    return tuple(sorted(resolved))


def to_channel_group(group: ChannelGroupLike) -> ChannelGroup:
    """
    Convert a band selection to a :class:`ChannelGroup`.

    Parameters
    ----------
    group : int or Sequence[int] or range or ChannelGroup
        The selection. A `ChannelGroup` is returned unchanged.

    Returns
    -------
    ChannelGroup
        The selection as a group.

    Examples
    --------
    >>> to_channel_group([0, 1, 2])
    ChannelGroup((0, 1, 2))
    """
    return group if isinstance(group, ChannelGroup) else ChannelGroup(group)


def rescale(image: NDArray[Any], depth: int = 1, value_range: ValueRange | None = None) -> NDArray[Any]:
    """
    Rescales the image using the value range provided.

    Parameters
    ----------
    image : NDArray
        Input image array
    depth : int, default 1
        Target bit depth
    value_range : ValueRange or None, default None
        Source range to rescale *from*. Read off `image` itself with
        :func:`get_value_range` when None.

        Pass it to scale several views of one datum onto a common range. A crop, or a
        region with part of it masked out, has its own extremes, and letting each infer
        its own source range scales them against different denominators — which is
        exactly what makes their statistics look different when the pixels do not. Hand
        in ``get_value_range(whole_image)`` and every view lands on one scale.

    Returns
    -------
    NDArray
        Rescaled image

    Raises
    ------
    ValueError
        If the range is unknown — see :func:`get_value_range`. There is no interval to
        divide by, so scaling would be arithmetic on an arbitrary maximum. Declare one,
        or ask for :meth:`ValueRange.observed` if a per-array stretch is what you want.

    Examples
    --------
    A crop dark enough to read as 1-bit is left alone when it infers its own range:

    >>> crop = np.array([[0, 1]], dtype=np.uint8)
    >>> rescale(crop)
    array([[0, 1]], dtype=uint8)

    Anchored on the 8-bit image it was cut from, it stays dark:

    >>> rescale(crop, value_range=get_value_range(np.array([[0, 255]], dtype=np.uint8)))
    array([[0.   , 0.004]])
    """
    value_range = get_value_range(image) if value_range is None else value_range
    if not value_range.is_known:
        raise ValueError(
            "Cannot rescale: no value range could be established for this data. Float data spanning "
            "more than [0, 255], and any data holding negative values, carries no encoding to decode. "
            "Declare the range it should be scaled against, or pass "
            "value_range=ValueRange.observed(image) to stretch this array's own extremes."
        )
    if value_range.depth == depth:
        return image
    span = value_range.pmax - value_range.pmin
    if span == 0:
        # A zero-width interval only reaches here from `ValueRange.observed` of a constant
        # array — every decoded or declared range is ordered. Dividing by it would answer
        # NaN with a RuntimeWarning, which reads as unmeasured data rather than as the
        # uniform array it is; 0 is min-max normalization's answer for a constant.
        #
        # NaN is carried through rather than flattened to 0. `observed` skips NaN when it
        # reads the extremes, so a constant array with holes in it reaches here, and a hole
        # is an absent measurement rather than a value equal to the constant.
        return np.where(np.isnan(image), np.nan, 0.0).astype(np.float64)
    normalized = (image - value_range.pmin) / span
    return normalized * (2**depth - 1)


def normalize_image_shape(image: NDArray[Any]) -> NDArray[Any]:
    """
    Normalize the image shape into (C,H,W) format.

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
    raise ShapeMismatchError("Images must have 2 or more dimensions.")


def edge_filter(image: NDArray[Any], offset: float = 0.5) -> NDArray[np.uint8]:
    """
    Return the image filtered using a 3x3 edge detection kernel.

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
    edges = convolve2d(image, _EDGE_KERNEL, mode="same", boundary="symm") + offset
    np.clip(edges, 0, 255, edges)
    return edges


def crop_with_fill(
    image: NDArray[Any],
    window: Box,
    fill: float | NDArray[Any] | Callable[[NDArray[Any]], NDArray[Any]] = np.nan,
    *,
    dtype: DTypeLike | None = None,
) -> tuple[NDArray[Any], tuple[int, int]]:
    """
    Extract a window from an image into an output of the window's size, filling out-of-bounds pixels with ``fill``.

    The window may extend past the image edges; pixels outside the image take the fill value.

    Parameters
    ----------
    image : NDArray
        Input image array in format C, H, W (channels first)
    window : Box
        Window to extract as (x0, y0, x1, y1); may extend past the image edges
    fill : float, NDArray, or callable, default np.nan
        Value for out-of-bounds pixels. A scalar or per-channel array is broadcast
        directly; a callable is passed the in-bounds region (C, H, W; possibly
        empty) and must return a scalar or per-channel array.
    dtype : DTypeLike or None, default None
        Output dtype. Defaults to a dtype that can hold both the image pixels and the
        fill, so ``np.nan`` fill promotes an integer image to float automatically.

    Returns
    -------
    tuple[NDArray, tuple[int, int]]
        The extracted window and its integer (x0, y0) origin in image coordinates
        (maps an output coordinate back to the source image).
    """
    x0, y0, x1, y1 = to_int_box(window)
    out_w, out_h = max(1, x1 - x0), max(1, y1 - y0)
    channels = image.shape[-3] if image.ndim > 2 else 1

    # In-bounds source pixels: the window clipped to the image (may be empty).
    sbox = clip_box(image.shape, (x0, y0, x1, y1))
    region = image[..., sbox[1] : sbox[3], sbox[0] : sbox[2]]
    fill_value = fill(region) if callable(fill) else fill

    if dtype is None:
        # Hold both the real pixels and the fill, so NaN fill promotes an integer image to float.
        dtype = np.result_type(image.dtype, np.asarray(fill_value))
    output = np.empty((channels, out_h, out_w), dtype=dtype)
    output[:] = np.reshape(fill_value, (channels, 1, 1)) if np.ndim(fill_value) else fill_value

    # Paste the real pixels at their offset within the (window-sized) output.
    if is_valid_box(sbox):
        dx, dy = sbox[0] - x0, sbox[1] - y0
        output[..., dy : dy + (sbox[3] - sbox[1]), dx : dx + (sbox[2] - sbox[0])] = region

    return output, (x0, y0)


@deprecated(since="1.1", removal="1.2")
def clip_and_pad(image: NDArray[Any], box: Box) -> NDArray[Any]:
    """
    Extract a region from an image based on a bounding box.

    Clips to image boundaries and pads out-of-bounds areas with np.nan.

    .. deprecated:: 1.1
        Use :func:`crop_with_fill` and pass ``fill=np.nan`` and take the
        first tuple value for the equivalent functionality.  Will be
        removed in v1.2.

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
    return crop_with_fill(image, box, fill=np.nan, dtype=float)[0]


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


def to_canonical_grayscale(image: NDArray[Any]) -> NDArray[np.uint8]:  # noqa: C901
    """
    Convert an image of arbitrary channels (CHW) to a single-channel uint8 grayscale image (HW).

    Uses color-space-aware heuristics.

    Parameters
    ----------
    image : NDArray
        Input array in CHW format

    Returns
    -------
    NDArray[np.uint8]
        2D grayscale array (HW) of type np.uint8
    """
    # Rescale normalized [0, 1] float images to [0, 255] range
    if np.issubdtype(image.dtype, np.floating) and image.size > 0:
        pmin, pmax = np.nanmin(image), np.nanmax(image)
        if pmax <= 1.0 and pmin >= 0.0:
            image = image * 255.0

    channels = image.shape[0]

    # --- Case 1: Single Channel (Already Grayscale) ---
    if channels == 1:
        return np.clip(np.nan_to_num(image[0], nan=0.0), 0, 255).astype(np.uint8)

    # --- Case 2: RGB (3 Channels) ---
    if channels == 3:
        # Rec. 601 Luma coefficients
        weights = np.array([0.299, 0.587, 0.114]).reshape(3, 1, 1)
        grayscale = np.sum(image.astype(float) * weights, axis=0)
        return np.clip(np.nan_to_num(grayscale, nan=0.0), 0, 255).astype(np.uint8)

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

        return np.clip(np.nan_to_num(grayscale, nan=0.0), 0, 255).astype(np.uint8)

    # --- Case 4: Arbitrary Channels (Fallback) ---
    # For 2, 5, or more channels, we simply average all information.
    return np.clip(np.nan_to_num(np.mean(image, axis=0), nan=0.0), 0, 255).astype(np.uint8)
