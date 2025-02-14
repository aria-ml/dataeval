from __future__ import annotations

__all__ = []

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import convolve2d

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
    pmin, pmax = np.min(image), np.max(image)
    if pmin < 0:
        return BitDepth(0, pmin, pmax)
    else:
        depth = ([x for x in BIT_DEPTH if 2**x > pmax] or [max(BIT_DEPTH)])[0]
        return BitDepth(depth, 0, 2**depth - 1)


def rescale(image: NDArray[Any], depth: int = 1) -> NDArray[Any]:
    """
    Rescales the image using the bit depth provided.
    """
    bitdepth = get_bitdepth(image)
    if bitdepth.depth == depth:
        return image
    else:
        normalized = (image + bitdepth.pmin) / (bitdepth.pmax - bitdepth.pmin)
        return normalized * (2**depth - 1)


def normalize_image_shape(image: NDArray[Any]) -> NDArray[Any]:
    """
    Normalizes the image shape into (C,H,W).
    """
    ndim = image.ndim
    if ndim == 2:
        return np.expand_dims(image, axis=0)
    elif ndim == 3:
        return image
    elif ndim > 3:
        # Slice all but the last 3 dimensions
        return image[(0,) * (ndim - 3)]
    else:
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
