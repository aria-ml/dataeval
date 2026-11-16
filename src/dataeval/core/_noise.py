"""Estimates of the standard deviation of additive white Gaussian noise in an image, one per channel."""

__all__ = []

import math
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dataeval._experimental import experimental

# The fewest valid 3x3 neighborhoods a channel needs for an estimate: an 8x8 grid of responses, so a 10x10 region.
# `image_stats` reads noise with this floor fixed.
_MIN_NEIGHBORHOODS = 64

# ponytail: the patch side of `noise_pca`, checked on simulated white noise at 7x7 only; make it an argument once a
# study at other sizes backs a floor for each.
_PATCH = 7
# ponytail: the fewest clear patches `noise_pca` needs (28x28 positions, so a 34x34 region), the smallest count at
# which the corrected estimate read within a few percent on simulated white noise at 7x7; an argument with evidence.
_MIN_PATCHES = 784


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
        :func:`noise_immerkaer`, the valid 3x3 neighborhoods; for :func:`noise_pca`, the positions whose whole 7x7
        patch is finite, of which it uses at most ``n_patches``.
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


@experimental
def noise_pca(image: ArrayLike, *, n_patches: int = 2000) -> NoiseResult:
    """
    Estimate the standard deviation of additive white Gaussian noise in each channel from its 7x7 patches.

    .. warning::
       This feature is experimental and may change or be removed in future releases.

    .. versionadded:: 1.3

    White noise of standard deviation ``sigma`` adds ``sigma**2`` of variance in every direction of the 49 values of a
    patch, while an image's structure adds variance in some directions only, so the smallest eigenvalue of the
    covariance of the patches reads the noise wherever the structure leaves a direction free of it [1].

    In each channel, the candidates are the ``N`` positions whose whole 7x7 patch is finite, so a NaN or infinite
    pixel is never read. ``n = min(n_patches, N)`` of them are taken evenly spaced over their raster order, so the
    result needs no seed. The smallest eigenvalue ``lambda_min`` of their sample covariance reads low: for ``n``
    independent samples of white noise it is close to ``sigma**2 * (1 - sqrt(49 / n))**2``, the lower edge of the
    Marchenko-Pastur law [2], so the estimate divides that factor out. Overlapping patches are not independent, so the
    correction is approximate:

    ``noise_std = sqrt(max(lambda_min, 0) / (1 - sqrt(49 / n))**2)``

    The estimate is in the image's own units: nothing is normalized, so divide the image by its full scale first
    (by 255 for ``uint8``) to read it in ``[0, 1]``. A channel with fewer than 784 candidates reads NaN, since with
    fewer patches the corrected estimate reads increasingly high on white noise, and ``support`` gives ``N`` either
    way.

    Parameters
    ----------
    image : ArrayLike
        One image, ``(H, W)`` for one channel or ``(C, H, W)``, of any real dtype. Each channel is converted to
        float64, so unsigned integers do not wrap.
    n_patches : int, default 2000
        The most patches a channel's covariance is computed from. A fixed count, so the cost beyond finding the
        candidates does not grow with the image. At least 784.

    Returns
    -------
    NoiseResult
        ``noise_std`` and ``support``, each of shape ``(C,)``, ``(1,)`` for an ``(H, W)`` image.

    Raises
    ------
    ValueError
        If ``image`` is neither ``(H, W)`` nor ``(C, H, W)``, or ``n_patches`` is not an int of at least 784.

    See Also
    --------
    :func:`noise_immerkaer` : Immerkaer's estimate of the same quantity, with the same result

    Notes
    -----
    This is not the method of Pyatykh et al. [1], which works on 5x5 blocks, checks that the smallest eigenvalues
    agree within a bound that shrinks with the number of blocks, and while they do not, drops the blocks of largest
    variance and tries again. Here one fixed set of patches is read with no such check. Structure that fills every
    direction of a patch, such as fine texture, adds to the reading: a noise-free textured image reads above 0.
    Structure whose patches span few directions, such as a straight edge or straight stripes, reads 0 on its own.

    A channel needs 784 clear patches, about a 34x34 region with no masked pixel, so a small region reads NaN where
    :func:`noise_immerkaer` reads it. The edge correction assumes noise that is independent from
    pixel to pixel; correlated, signal-dependent and re-encoded noise were not measured. The 7x7 patch, the floor of
    784 and the default of 2000 patches are provisional and were checked on simulated white noise only, and how the
    estimate compares with :func:`noise_immerkaer` on real images is not established.

    References
    ----------
    [1] Pyatykh, S., Hesser, J. and Zheng, L. (2013). Image Noise Level Estimation by Principal Component Analysis.
        *IEEE Transactions on Image Processing* 22(2), 687-699. https://doi.org/10.1109/TIP.2012.2221728

    [2] Marchenko, V. A. and Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices.
        *Mathematics of the USSR-Sbornik* 1(4), 457-483. https://doi.org/10.1070/SM1967v001n04ABEH001994

    Examples
    --------
    White Gaussian noise of standard deviation 2 on a flat 64x64 image:

    >>> from dataeval.core import noise_pca
    >>> rng = np.random.default_rng(0)
    >>> image = 100.0 + rng.normal(0.0, 2.0, size=(64, 64))
    >>> result = noise_pca(image)
    >>> round(float(result["noise_std"][0]), 1)
    2.0
    >>> int(result["support"][0])
    3364
    """
    if isinstance(n_patches, bool) or not isinstance(n_patches, int | np.integer) or n_patches < _MIN_PATCHES:
        raise ValueError(f"n_patches must be an int of at least {_MIN_PATCHES}, got {n_patches!r}.")
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError(f"image must be (H, W) or (C, H, W), got shape {array.shape}.")
    channels = array[None] if array.ndim == 2 else array

    noise_std = np.full(len(channels), np.nan)
    support = np.zeros(len(channels), dtype=np.int64)
    for c, channel in enumerate(channels):
        # A copy for integers; for a float64 input it may be the caller's array, so it is never written.
        noise_std[c], support[c] = _pca_channel(np.asarray(channel, dtype=np.float64), int(n_patches))
    return NoiseResult(noise_std=noise_std, support=support)


def _pca_channel(x: NDArray[np.float64], n_patches: int) -> tuple[float, int]:
    """Return one channel's corrected estimate, NaN under the floor, and its count of clear patches."""
    p = _PATCH
    # Positions whose whole patch is finite, from an integral image of the non-finite pixels.
    bad = np.pad((~np.isfinite(x)).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ys, xs = np.nonzero((bad[p:, p:] - bad[:-p, p:] - bad[p:, :-p] + bad[:-p, :-p]) == 0)
    if ys.size < _MIN_PATCHES:
        return math.nan, ys.size
    n = min(n_patches, ys.size)
    pick = np.linspace(0, ys.size - 1, n).round().astype(np.intp)
    patches = np.lib.stride_tricks.sliding_window_view(x, (p, p))[ys[pick], xs[pick]].reshape(n, p * p)
    with np.errstate(over="ignore", invalid="ignore"):
        covariance = np.cov(patches, rowvar=False)
    # Values near the float64 limit overflow the covariance; that channel reads NaN rather than raise.
    if not np.isfinite(covariance).all():
        return math.nan, ys.size
    smallest = float(np.linalg.eigvalsh(covariance)[0])
    return math.sqrt(max(smallest, 0.0) / (1 - math.sqrt(p * p / n)) ** 2), ys.size
