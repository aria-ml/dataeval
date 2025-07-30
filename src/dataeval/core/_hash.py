from __future__ import annotations

__all__ = []

import warnings
from typing import Any

import numpy as np
import xxhash as xxh
from numpy.typing import NDArray
from scipy.fftpack import dct

from dataeval.utils._image import normalize_image_shape, rescale, resize

HASH_SIZE = 8
MAX_FACTOR = 4


def pchash(image: NDArray[Any]) -> str:
    """
    Performs a perceptual hash on an image by resizing to a square NxN image
    using the Lanczos algorithm where N is 32x32 or the largest multiple of
    8 that is smaller than the input image dimensions. The resampled image
    is compressed using a discrete cosine transform and the lowest frequency
    component is encoded as a bit array of greater or less than median value
    and returned as a hex string.

    Parameters
    ----------
    image : NDArray[Any]
        An image as a numpy array in CxHxW format

    Returns
    -------
    str
        The hex string hash of the image using perceptual hashing, or empty
        string if the image is too small to be hashed
    """
    # Verify that the image is at least larger than an 8x8 image
    min_dim = min(image.shape[-2:])
    if min_dim < HASH_SIZE + 1:
        warnings.warn(f"Image must be larger than {HASH_SIZE}x{HASH_SIZE} for fuzzy hashing.")
        return ""

    # Calculates the dimensions of the resized square image
    resize_dim = HASH_SIZE * min((min_dim - 1) // HASH_SIZE, MAX_FACTOR)

    # Normalizes the image to CxHxW and takes the mean over all the channels
    normalized = np.mean(normalize_image_shape(image), axis=0).squeeze()

    # Rescales the pixel values to an 8-bit 0-255 image
    rescaled = rescale(normalized, 8).astype(np.uint8)

    # Resizes the image using the Lanczos algorithm to a square image
    im = resize(rescaled, resize_dim)

    # Performs discrete cosine transforms to compress the image information and takes the lowest frequency component
    transform = dct(dct(im.T).T)[:HASH_SIZE, :HASH_SIZE]

    # Encodes the transform as a bit array over the median value
    diff = transform > np.median(transform)

    # Pads the front of the bit array to a multiple of 8 with False
    padded = np.full(int(np.ceil(diff.size / 8) * 8), False)
    padded[-diff.size :] = diff.ravel()

    # Converts the bit array to a hex string and strips leading 0s
    hash_hex = np.packbits(padded).tobytes().hex().lstrip("0")
    return hash_hex if hash_hex else "0"


def xxhash(image: NDArray[Any]) -> str:
    """
    Performs a fast non-cryptographic hash using the xxhash algorithm
    (xxhash.com) against the image as a flattened bytearray. The hash
    is returned as a hex string.

    Parameters
    ----------
    image : NDArray[Any]
        An image as a numpy array

    Returns
    -------
    str
        The hex string hash of the image using the xxHash algorithm
    """
    return xxh.xxh3_64_hexdigest(image.ravel().tobytes())
