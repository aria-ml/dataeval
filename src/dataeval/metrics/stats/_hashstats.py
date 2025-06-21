from __future__ import annotations

import warnings

__all__ = []

from collections.abc import Callable
from typing import Any

import numpy as np
import xxhash as xxh
from numpy.typing import NDArray
from scipy.fftpack import dct
from scipy.ndimage import zoom

from dataeval.metrics.stats._base import StatsProcessor, run_stats
from dataeval.outputs import HashStatsOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import ArrayLike, Dataset
from dataeval.utils._array import as_numpy
from dataeval.utils._image import normalize_image_shape, rescale

try:
    from PIL import Image
except ImportError:
    Image = None

HASH_SIZE = 8
MAX_FACTOR = 4


def _resize(image: NDArray[np.uint8], resize_dim: int, use_pil: bool = True) -> NDArray[np.uint8]:
    """Resizes a grayscale (HxW) 8-bit image using PIL or scipy.ndimage.zoom."""

    # Use PIL if available, otherwise resize and resample with scipy.ndimage.zoom
    if use_pil and Image is not None:
        return np.array(Image.fromarray(image).resize((resize_dim, resize_dim), Image.Resampling.LANCZOS))

    zoom_factors = (resize_dim / image.shape[0], resize_dim / image.shape[1])
    return np.clip(zoom(image, zoom_factors, order=5, mode="reflect"), 0, 255, dtype=np.uint8)


def pchash(image: ArrayLike) -> str:
    """
    Performs a perceptual hash on an image by resizing to a square NxN image
    using the Lanczos algorithm where N is 32x32 or the largest multiple of
    8 that is smaller than the input image dimensions. The resampled image
    is compressed using a discrete cosine transform and the lowest frequency
    component is encoded as a bit array of greater or less than median value
    and returned as a hex string.

    Parameters
    ----------
    image : ArrayLike
        An image as a numpy array in CxHxW format

    Returns
    -------
    str
        The hex string hash of the image using perceptual hashing, or empty
        string if the image is too small to be hashed
    """
    # Verify that the image is at least larger than an 8x8 image
    arr = as_numpy(image)
    min_dim = min(arr.shape[-2:])
    if min_dim < HASH_SIZE + 1:
        warnings.warn(f"Image must be larger than {HASH_SIZE}x{HASH_SIZE} for fuzzy hashing.")
        return ""

    # Calculates the dimensions of the resized square image
    resize_dim = HASH_SIZE * min((min_dim - 1) // HASH_SIZE, MAX_FACTOR)

    # Normalizes the image to CxHxW and takes the mean over all the channels
    normalized = np.mean(normalize_image_shape(arr), axis=0).squeeze()

    # Rescales the pixel values to an 8-bit 0-255 image
    rescaled = rescale(normalized, 8).astype(np.uint8)

    # Resizes the image using the Lanczos algorithm to a square image
    im = _resize(rescaled, resize_dim)

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


def xxhash(image: ArrayLike) -> str:
    """
    Performs a fast non-cryptographic hash using the xxhash algorithm
    (xxhash.com) against the image as a flattened bytearray. The hash
    is returned as a hex string.

    Parameters
    ----------
    image : ArrayLike
        An image as a numpy array

    Returns
    -------
    str
        The hex string hash of the image using the xxHash algorithm
    """
    return xxh.xxh3_64_hexdigest(as_numpy(image).ravel().tobytes())


class HashStatsProcessor(StatsProcessor[HashStatsOutput]):
    output_class: type = HashStatsOutput
    image_function_map: dict[str, Callable[[StatsProcessor[HashStatsOutput]], str]] = {
        "xxhash": lambda x: xxhash(x.image),
        "pchash": lambda x: pchash(x.image),
    }


@set_metadata
def hashstats(
    dataset: Dataset[ArrayLike] | Dataset[tuple[ArrayLike, Any, Any]],
    *,
    per_box: bool = False,
) -> HashStatsOutput:
    """
    Calculates hashes for each image.

    This function computes hashes from the images including exact hashes and perception-based
    hashes. These hash values can be used to determine if images are exact or near matches.

    Parameters
    ----------
    dataset : Dataset
        Dataset to perform calculations on.
    per_box : bool, default False
        If True, perform calculations on each bounding box.

    Returns
    -------
    HashStatsOutput
        A dictionary-like object containing the computed hashes for each image.

    See Also
    --------
    Duplicates

    Examples
    --------
    Calculate the hashes of a dataset of images, whose shape is (C, H, W)

    >>> results = hashstats(dataset)
    >>> print(results.xxhash[:5])
    ['69b50a5f06af238c', '5a861d7a23d1afe7', '7ffdb4990ad44ac6', '4f0c366a3298ceac', 'c5519e36ac1f8839']
    >>> print(results.pchash[:5])
    ['e666999999266666', 'e666999999266666', 'e666999966666299', 'e666999999266666', '96e91656e91616e9']
    """
    return run_stats(dataset, per_box, False, [HashStatsProcessor])[0]
