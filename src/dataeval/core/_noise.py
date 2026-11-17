"""Estimates of the standard deviation of additive white Gaussian noise in an image, one per channel."""

__all__ = []

import math
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

# The fewest valid 3x3 neighborhoods a channel needs for an estimate: an 8x8 grid of responses, so a 10x10 region.
# `image_stats` reads noise with this floor fixed.
_MIN_NEIGHBORHOODS = 64


class NoiseResult(TypedDict):
    """
    Estimated noise of one image, one value per channel.

    Attributes
    ----------
    noise_std : NDArray[np.float64]
        Shape ``(C,)``. The estimated standard deviation of additive white Gaussian noise in each channel, in the
        image's own units. NaN where the channel's ``support`` is under the floor.
    support : NDArray[np.int64]
        Shape ``(C,)``. What each channel's estimate averaged over, counted whether or not it met the floor: for
        :func:`noise_immerkaer`, the valid 3x3 neighborhoods.
    """

    noise_std: NDArray[np.float64]
    support: NDArray[np.int64]


def noise_immerkaer(image: ArrayLike, *, min_neighborhoods: int = _MIN_NEIGHBORHOODS) -> NoiseResult:
    """
    Estimate the standard deviation of additive white Gaussian noise in each channel with Immerkaer's estimator.

    Each channel ``I`` is convolved with the 3x3 mask ``M = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]``, which takes
    the second difference along the rows and then along the columns, and the estimate is

    ``noise_std = sqrt(pi / 2) * mean(|I * M|) / 6``

    over the response locations whose whole 3x3 neighborhood is finite. The squared coefficients of ``M`` sum to
    36, so on white Gaussian noise of standard deviation ``sigma`` each response has a standard deviation of
    ``6 * sigma`` and its absolute value a mean of ``6 * sigma * sqrt(2 / pi)``.

    The estimate is in the image's own units: nothing is normalized, so divide the image by its full scale first
    (by 255 for ``uint8``) to read it in ``[0, 1]``.

    A pixel that is NaN or infinite is left out: a response counts only where its whole 3x3 neighborhood is finite,
    so a masked pixel creates no edge. A channel with fewer than ``min_neighborhoods`` valid neighborhoods, which
    includes any channel under 3x3, reads NaN, and ``support`` gives its count either way.

    .. versionadded:: 1.3

    Parameters
    ----------
    image : ArrayLike
        One image, ``(H, W)`` for one channel or ``(C, H, W)``, of any real dtype. Each channel is converted to
        float64, so unsigned integers do not wrap.
    min_neighborhoods : int, default 64
        The fewest valid 3x3 neighborhoods a channel needs for an estimate. At least 1.

    Returns
    -------
    NoiseResult
        ``noise_std`` and ``support``, each of shape ``(C,)``, ``(1,)`` for an ``(H, W)`` image.

    Raises
    ------
    ValueError
        If ``image`` is neither ``(H, W)`` nor ``(C, H, W)``, or ``min_neighborhoods`` is not an int of at least 1.

    Notes
    -----
    The mask removes, among other patterns, any that varies along the rows alone or along the columns alone, so an
    edge or stripes parallel to a side of the image read 0. Other structure, such as a diagonal edge or fine
    texture, adds to the reading. Pixels that carry no noise, such as letterbox bars, clipped regions or a constant
    fill, read 0 and pull the estimate down in proportion to their share. White Gaussian noise is assumed, and
    correlated, signal-dependent and re-encoded noise were not measured.

    Near the floor a single reading is widely spread: on white noise in 10x10 regions, one region in twenty reads
    under about 0.55 of the true variance and one in twenty over about 1.65 times it. Compare the distributions of
    many readings, not single ones.

    References
    ----------
    [1] Immerkær, J. (1996). Fast Noise Variance Estimation. *Computer Vision and Image Understanding* 64(2),
        300-302. https://doi.org/10.1006/cviu.1996.0060

    Examples
    --------
    White Gaussian noise of standard deviation 2 on a flat 64x64 image:

    >>> from dataeval.core import noise_immerkaer
    >>> rng = np.random.default_rng(0)
    >>> image = 100.0 + rng.normal(0.0, 2.0, size=(64, 64))
    >>> result = noise_immerkaer(image)
    >>> round(float(result["noise_std"][0]), 1)
    2.0
    >>> int(result["support"][0])
    3844
    """
    if (
        isinstance(min_neighborhoods, bool)
        or not isinstance(min_neighborhoods, int | np.integer)
        or min_neighborhoods < 1
    ):
        raise ValueError(f"min_neighborhoods must be an int of at least 1, got {min_neighborhoods!r}.")
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError(f"image must be (H, W) or (C, H, W), got shape {array.shape}.")
    channels = array[None] if array.ndim == 2 else array

    noise_std = np.full(len(channels), np.nan)
    support = np.zeros(len(channels), dtype=np.int64)
    for c, channel in enumerate(channels):
        # A copy for integers; for a float64 input it may be the caller's array, so it is never written.
        x = np.asarray(channel, dtype=np.float64)
        # ponytail: about five channel-size float64 arrays at peak; a chunked pass if one band over ~25 MP matters.
        # A response that overflows float64 is dropped as non-finite, silently; no realistic range does.
        with np.errstate(over="ignore", invalid="ignore"):
            a = x[:-2] - 2 * x[1:-1] + x[2:]
            g = a[:, :-2] - 2 * a[:, 1:-1] + a[:, 2:]
            g = np.abs(g[np.isfinite(g)])
            support[c] = g.size
            if g.size >= min_neighborhoods:
                noise_std[c] = math.sqrt(math.pi / 2) * g.mean() / 6
    return NoiseResult(noise_std=noise_std, support=support)
