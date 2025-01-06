from __future__ import annotations

__all__ = []

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import xxhash as xxh
from numpy.typing import ArrayLike
from PIL import Image
from scipy.fftpack import dct

from dataeval.interop import as_numpy
from dataeval.metrics.stats.base import BaseStatsOutput, StatsProcessor, run_stats
from dataeval.output import set_metadata
from dataeval.utils.image import normalize_image_shape, rescale

HASH_SIZE = 8
MAX_FACTOR = 4


@dataclass(frozen=True)
class HashStatsOutput(BaseStatsOutput):
    """
    Output class for :func:`hashstats` stats metric

    Attributes
    ----------
    xxhash : List[str]
        xxHash hash of the images as a hex string
    pchash : List[str]
        :term:`Perception-based Hash` of the images as a hex string
    """

    xxhash: list[str]
    pchash: list[str]


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
        The hex string hash of the image using perceptual hashing
    """
    # Verify that the image is at least larger than an 8x8 image
    arr = as_numpy(image)
    min_dim = min(arr.shape[-2:])
    if min_dim < HASH_SIZE + 1:
        raise ValueError(f"Image must be larger than {HASH_SIZE}x{HASH_SIZE} for fuzzy hashing.")

    # Calculates the dimensions of the resized square image
    resize_dim = HASH_SIZE * min((min_dim - 1) // HASH_SIZE, MAX_FACTOR)

    # Normalizes the image to CxHxW and takes the mean over all the channels
    normalized = np.mean(normalize_image_shape(arr), axis=0).squeeze()

    # Rescales the pixel values to an 8-bit 0-255 image
    rescaled = rescale(normalized, 8).astype(np.uint8)

    # Resizes the image using the Lanczos algorithm to a square image
    im = np.array(Image.fromarray(rescaled).resize((resize_dim, resize_dim), Image.Resampling.LANCZOS))

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
    images: Iterable[ArrayLike],
    bboxes: Iterable[ArrayLike] | None = None,
) -> HashStatsOutput:
    """
    Calculates hashes for each image

    This function computes hashes from the images including exact hashes and perception-based
    hashes. These hash values can be used to determine if images are exact or near matches.

    Parameters
    ----------
    images : ArrayLike
        Images to hashing
    bboxes : Iterable[ArrayLike] or None
        Bounding boxes in `xyxy` format for each image

    Returns
    -------
    HashStatsOutput
        A dictionary-like object containing the computed hashes for each image.

    See Also
    --------
    Duplicates

    Examples
    --------
    Calculating the statistics on the images, whose shape is (C, H, W)

    >>> results = hashstats(stats_images)
    >>> print(results.xxhash)
    ['6274f837b34ed9f0', '256504fdb6e3d2a4', '7dd0c56ca8474fb0', '50956ad4592f5bbc', '5ba2354079d42aa5']
    >>> print(results.pchash)
    ['a666999999666666', 'e666999999266666', 'e666999966663299', 'e666999999266666', '96e91656e91616e9']
    """
    return run_stats(images, bboxes, False, [HashStatsProcessor])[0]
