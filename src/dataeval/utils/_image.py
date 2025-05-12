from __future__ import annotations

__all__ = []

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import convolve2d

EDGE_KERNEL = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.int8)
BIT_DEPTH = (1, 8, 12, 16, 32)

Box = tuple[int, int, int, int]
"""Bounding box as tuple of integers in x0, y0, x1, y1 format."""


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


def clip_box(image: NDArray[Any], box: Box) -> Box:
    """
    Clip the box to inside the provided image dimensions.
    """
    x0, y0, x1, y1 = box
    h, w = image.shape[-2:]

    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def is_valid_box(box: Box) -> bool:
    """
    Check if the box dimensions provided are a valid image.
    """
    return box[2] > box[0] and box[3] > box[1]


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
    bw, bh = max(1, box[2] - box[0]), max(1, box[3] - box[1])

    output = np.full((image.shape[-3] if image.ndim > 2 else 1, bh, bw), np.nan)

    # Calculate source box
    sbox = clip_box(image, box)

    # Calculate destination box
    x0, y0 = sbox[0] - box[0], sbox[1] - box[1]
    x1, y1 = x0 + (sbox[2] - sbox[0]), y0 + (sbox[3] - sbox[1])

    # Copy the source if valid from the image to the output
    if is_valid_box(sbox):
        output[:, y0:y1, x0:x1] = image[:, sbox[1] : sbox[3], sbox[0] : sbox[2]]

    return output
