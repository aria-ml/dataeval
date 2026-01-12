__all__ = []

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import zoom
from scipy.signal import convolve2d

try:
    from PIL import Image
except ImportError:
    Image = None

from dataeval.utils._boundingbox import Box, clip_box, is_valid_box, to_int_box

EDGE_KERNEL = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.int8)
BIT_DEPTH = (1, 8, 12, 16, 32)


@dataclass
class BitDepth:
    depth: int
    pmin: float | int
    pmax: float | int


def get_bitdepth(image: NDArray[Any]) -> BitDepth:
    """
    Approximates the bit depth of the image using the
    min and max pixel values.
    """
    pmin, pmax = np.nanmin(image), np.nanmax(image)
    if pmin < 0:
        return BitDepth(0, pmin, pmax)
    depth = ([x for x in BIT_DEPTH if 2**x > pmax] or [max(BIT_DEPTH)])[0]
    return BitDepth(depth, 0, 2**depth - 1)


def rescale(image: NDArray[Any], depth: int = 1) -> NDArray[Any]:
    """
    Rescales the image using the bit depth provided.
    """
    bitdepth = get_bitdepth(image)
    if bitdepth.depth == depth:
        return image
    normalized = (image + bitdepth.pmin) / (bitdepth.pmax - bitdepth.pmin)
    return normalized * (2**depth - 1)


def normalize_image_shape(image: NDArray[Any]) -> NDArray[Any]:
    """
    Normalizes the image shape into (C,H,W).
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
    Returns the image filtered using a 3x3 edge detection kernel:
    [[ -1, -1, -1 ],
    [ -1,  8, -1 ],
    [ -1, -1, -1 ]]
    """
    edges = convolve2d(image, EDGE_KERNEL, mode="same", boundary="symm") + offset
    np.clip(edges, 0, 255, edges)
    return edges


def clip_and_pad(image: NDArray[Any], box: Box) -> NDArray[Any]:
    """
    Extract a region from an image based on a bounding box, clipping to image boundaries
    and padding out-of-bounds areas with np.nan.

    Parameters:
    -----------
    image : NDArray[Any]
        Input image array in format C, H, W (channels first)
    box : Box
        Bounding box coordinates as (x0, y0, x1, y1) where (x0, y0) is top-left and (x1, y1) is bottom-right

    Returns:
    --------
    NDArray[Any]
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
    """Resizes a grayscale (HxW) 8-bit image using PIL or scipy.ndimage.zoom."""

    # Use PIL if available, otherwise resize and resample with scipy.ndimage.zoom
    if use_pil and Image is not None:
        return np.array(Image.fromarray(image).resize((resize_dim, resize_dim), Image.Resampling.LANCZOS))

    zoom_factors = (resize_dim / image.shape[0], resize_dim / image.shape[1])
    return np.clip(zoom(image, zoom_factors, order=5, mode="reflect"), 0, 255, dtype=np.uint8)


def to_canonical_grayscale(image: NDArray[Any]) -> NDArray[np.uint8]:
    """
    Converts an image of arbitrary channels (CHW) to a single-channel
    uint8 grayscale image (HW) using color-space-aware heuristics.

    Parameters:
    -----------
    image : NDArray[Any]
        Input array in CHW format.

    Returns:
    --------
    np.ndarray
        2D array (HW) of type np.uint8.
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
