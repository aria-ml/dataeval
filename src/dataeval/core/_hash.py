"""Hash functions for image duplicate detection."""

__all__ = []

import logging
from typing import Any

import numpy as np
import xxhash as xxh
from numpy.typing import NDArray
from scipy.fftpack import dct

from dataeval.types import Array3D
from dataeval.utils.arrays import as_numpy
from dataeval.utils.preprocessing import normalize_image_shape, resize, to_canonical_grayscale

_logger = logging.getLogger(__name__)

HASH_SIZE = 8
MAX_FACTOR = 4


def _prepare_image(image: Array3D[Any], min_size: int = HASH_SIZE + 1) -> NDArray[np.uint8] | None:
    """
    Prepare an image for perceptual hashing by normalizing and converting to grayscale.

    Parameters
    ----------
    image : Array3D
        An image in CxHxW format. Can be a 3D list, or array-like object.
    min_size : int, default 9
        Minimum dimension required for hashing.

    Returns
    -------
    NDArray[np.uint8] | None
        Grayscale image ready for hashing, or None if image is unsuitable.
    """
    image_np = as_numpy(image)

    # Perceptual hashing only works on spatial data (2D or higher)
    if image_np.ndim < 2:
        _logger.warning("Perceptual hashing requires spatial data (2D or higher dimensions)")
        return None

    # Verify that the image is at least larger than minimum size
    min_dim = min(image_np.shape[-2:])
    if min_dim < min_size:
        _logger.warning("Image too small for perceptual hashing: min_dim=%d", min_dim)
        return None

    # Normalize the image shape to CxHxW
    normalized = normalize_image_shape(image_np)

    # Convert to single-channel grayscale image
    return to_canonical_grayscale(normalized)


def phash(image: Array3D[Any]) -> str:
    """
    Compute perceptual hash using Discrete Cosine Transform (DCT).

    Resizes image to a square NxN using Lanczos algorithm where N is 32x32
    or the largest multiple of 8 smaller than input dimensions. The resampled
    image is compressed using DCT and the lowest frequency component is encoded
    as a bit array of greater or less than median value.

    Parameters
    ----------
    image : Array3D
        An image in CxHxW format. Can be a 3D list, or array-like object.

    Returns
    -------
    str
        Hex string hash of the image, or empty string if image is too small
        or not spatial data.

    Notes
    -----
    DCT-based hashing (pHash) is robust to:
    - Scaling and resizing
    - Minor color adjustments
    - Compression artifacts

    It captures frequency information, making it effective for detecting
    images that have been resized or slightly modified.
    """
    image_np = as_numpy(image)
    _logger.debug("Computing perceptual hash for image with shape: %s", image_np.shape)

    grayscale = _prepare_image(image_np)
    if grayscale is None:
        return ""

    # Calculates the dimensions of the resized square image
    min_dim = min(image_np.shape[-2:])
    resize_dim = HASH_SIZE * min((min_dim - 1) // HASH_SIZE, MAX_FACTOR)

    # Resizes the image using the Lanczos algorithm to a square image
    im = resize(grayscale, resize_dim)

    # Performs discrete cosine transforms to compress the image information
    # and takes the lowest frequency component
    transform = dct(dct(im.T).T)[:HASH_SIZE, :HASH_SIZE]

    # Encodes the transform as a bit array over the median value
    diff = transform > np.median(transform)

    # Convert the bit array to a hex string
    padded = diff.flatten()
    hash_hex = np.packbits(padded).tobytes().hex()
    result = hash_hex if hash_hex else "0"
    _logger.debug("Perceptual hash computed: %s", result[:16] + "..." if len(result) > 16 else result)
    return result


def dhash(image: Array3D[Any]) -> str:
    """
    Compute difference hash (dHash) for an image.

    Resizes then crops image to 9x8 grayscale and computes horizontal gradient
    by comparing adjacent pixels, producing a 64-bit hash. Captures relative
    brightness changes rather than absolute values.

    Parameters
    ----------
    image : Array3D
        An image in CxHxW format. Can be a 3D list, or array-like object.

    Returns
    -------
    str
        Hex string hash of the image, or empty string if image is too small
        or not spatial data.

    Notes
    -----
    Difference hash captures gradient information:
    - Captures structural information via pixel transitions
    - Complementary to DCT-based pHash (frequency vs gradient domain)

    The horizontal gradient approach makes it particularly effective for
    detecting cropped or slightly shifted versions of images.
    """
    image_np = as_numpy(image)
    _logger.debug("Computing difference hash for image with shape: %s", image_np.shape)

    grayscale = _prepare_image(image_np)
    if grayscale is None:
        return ""

    # Resize to 9x8 (9 wide to get 8 differences)
    im = resize(grayscale, HASH_SIZE + 1)
    # Crop to 9x8 if resize produced square
    im = im[:HASH_SIZE, : HASH_SIZE + 1]

    # Compute horizontal gradient: compare pixel to its right neighbor
    diff = im[:, :-1] > im[:, 1:]

    # Convert the bit array to a hex string
    hash_hex = np.packbits(diff.flatten()).tobytes().hex()
    result = hash_hex if hash_hex else "0"
    _logger.debug("Difference hash computed: %s", result)
    return result


def _get_d4_transforms(image: NDArray[np.uint8]) -> list[NDArray[np.uint8]]:
    """
    Generate all 8 dihedral group (D4) transformations of an image.

    Parameters
    ----------
    image : NDArray[np.uint8]
        2D grayscale image (H, W).

    Returns
    -------
    list[NDArray[np.uint8]]
        List of 8 transformed images: 4 rotations × 2 flip states.
    """
    transforms = []
    img = image

    # 4 rotations (0°, 90°, 180°, 270°)
    for _ in range(4):
        transforms.append(img)
        transforms.append(np.fliplr(img))  # Horizontal flip of each rotation
        img = np.rot90(img)

    return transforms


def phash_d4(image: Array3D[Any]) -> str:
    """
    Compute orientation-invariant perceptual hash using DCT.

    Computes phash for all 8 dihedral group transformations (4 rotations ×
    2 flip states) and returns the lexicographically smallest hash as the
    canonical representative.

    Parameters
    ----------
    image : Array3D
        An image in CxHxW format. Can be a 3D list, or array-like object.

    Returns
    -------
    str
        Canonical hex string hash invariant to rotation and mirroring,
        or empty string if image is too small or not spatial data.

    Notes
    -----
    This hash is invariant to:
    - 90°, 180°, 270° rotations
    - Horizontal and vertical flips
    - Any combination of rotation and flip

    The canonical hash is the lexicographically smallest hash among all
    8 orientations, ensuring that any orientation of the same image
    produces the identical hash.

    Computation cost is ~8x that of regular phash.

    See Also
    --------
    phash : Standard orientation-sensitive perceptual hash
    dhash_d4 : Orientation-invariant difference hash
    """
    from scipy.fftpack import dct

    from dataeval.utils.preprocessing import normalize_image_shape, resize, to_canonical_grayscale

    HASH_SIZE = 8
    MAX_FACTOR = 4

    image_np = as_numpy(image)

    # Validate input
    if image_np.ndim < 2:
        return ""

    min_dim = min(image_np.shape[-2:])
    if min_dim < HASH_SIZE + 1:
        return ""

    # Prepare grayscale image
    normalized = normalize_image_shape(image_np)
    grayscale = to_canonical_grayscale(normalized)

    # Compute resize dimension
    resize_dim = HASH_SIZE * min((min_dim - 1) // HASH_SIZE, MAX_FACTOR)

    # Compute hash for each D4 transformation
    hashes: list[str] = []
    for transformed in _get_d4_transforms(grayscale):
        # Resize
        im = resize(transformed, resize_dim)

        # DCT transform
        transform = dct(dct(im.T).T)[:HASH_SIZE, :HASH_SIZE]

        # Binarize against median
        diff = transform > np.median(transform)

        # Convert to hex
        hash_hex = np.packbits(diff.flatten()).tobytes().hex()
        hashes.append(hash_hex if hash_hex else "0")

    # Return canonical (lexicographically smallest) hash
    return min(hashes)


def dhash_d4(image: Array3D[Any]) -> str:
    """
    Compute orientation-invariant difference hash using gradients.

    Computes dhash for all 8 dihedral group transformations (4 rotations ×
    2 flip states) and returns the lexicographically smallest hash as the
    canonical representative.

    Parameters
    ----------
    image : Array3D
        An image in CxHxW format. Can be a 3D list, or array-like object.

    Returns
    -------
    str
        Canonical hex string hash invariant to rotation and mirroring,
        or empty string if image is too small or not spatial data.

    Notes
    -----
    This hash is invariant to:
    - 90°, 180°, 270° rotations
    - Horizontal and vertical flips
    - Any combination of rotation and flip

    The canonical hash is the lexicographically smallest hash among all
    8 orientations, ensuring that any orientation of the same image
    produces the identical hash.

    Computation cost is ~8x that of regular dhash.

    See Also
    --------
    dhash : Standard orientation-sensitive difference hash
    phash_d4 : Orientation-invariant perceptual hash
    """
    from dataeval.utils.preprocessing import normalize_image_shape, resize, to_canonical_grayscale

    HASH_SIZE = 8

    image_np = as_numpy(image)

    # Validate input
    if image_np.ndim < 2:
        return ""

    min_dim = min(image_np.shape[-2:])
    if min_dim < HASH_SIZE + 1:
        return ""

    # Prepare grayscale image
    normalized = normalize_image_shape(image_np)
    grayscale = to_canonical_grayscale(normalized)

    # Compute hash for each D4 transformation
    hashes: list[str] = []
    for transformed in _get_d4_transforms(grayscale):
        # Resize to 9x8 (9 wide to get 8 horizontal differences)
        im = resize(transformed, HASH_SIZE + 1)
        im = im[:HASH_SIZE, : HASH_SIZE + 1]

        # Compute horizontal gradient
        diff = im[:, :-1] > im[:, 1:]

        # Convert to hex
        hash_hex = np.packbits(diff.flatten()).tobytes().hex()
        hashes.append(hash_hex if hash_hex else "0")

    # Return canonical (lexicographically smallest) hash
    return min(hashes)


def xxhash(image: Array3D[Any]) -> str:
    """
    Compute fast non-cryptographic hash using xxHash algorithm.

    Hashes the raw image bytes to detect exact duplicates. Any difference
    in pixel values will produce a different hash.

    Parameters
    ----------
    image : Array3D
        An image in CxHxW format. Can be a 3D list, or array-like object.

    Returns
    -------
    str
        Hex string hash of the image using the xxHash algorithm.

    Notes
    -----
    xxHash is used for exact duplicate detection only. Unlike perceptual
    hashes, it will produce completely different values for images that
    differ by even a single pixel.
    """
    image_np = as_numpy(image)
    _logger.debug("Computing xxhash for image with shape: %s", image_np.shape)
    hash_result = xxh.xxh3_64_hexdigest(image_np.ravel().tobytes())
    _logger.debug("xxhash computed: %s", hash_result)
    return hash_result


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Compute Hamming distance between two hex-encoded hashes.

    The Hamming distance is the number of bit positions where the two hashes
    differ. Lower values indicate more similar images.

    Parameters
    ----------
    hash1 : str
        First hex-encoded hash string.
    hash2 : str
        Second hex-encoded hash string.

    Returns
    -------
    int
        Number of differing bits between the two hashes.
        Returns -1 if hashes have different lengths or are empty.

    Notes
    -----
    Typical thresholds for 64-bit hashes:
    - 0: Identical or nearly identical images
    - 1-5: Very similar images (likely duplicates)
    - 6-10: Similar images (possible duplicates)
    - >10: Different images

    Examples
    --------
    >>> hamming_distance("ff00ff00", "ff00ff00")
    0
    >>> hamming_distance("ff00ff00", "ff00ff01")
    1
    """
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return -1

    # Convert hex strings to integers and XOR
    xor_result = int(hash1, 16) ^ int(hash2, 16)

    # Count the number of 1 bits (differing positions)
    return bin(xor_result).count("1")
