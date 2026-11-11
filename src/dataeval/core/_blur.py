"""Edge-width blur statistic of a grayscale image.

Edge points are found where the smoothed gradient peaks along an axis. The width of each is the spread of
the gradient magnitude around it, taken along the row or column that carries most of the gradient and
corrected for the gradient's direction. Two values summarize the widths: the 5th percentile (``blur``, in
pixels) and how different horizontal and vertical edges are (``directionality``).

After Liang, Wang, Chen, Ma, Zhao and Gao (2010), "No-reference perceptual image quality metric using
gradient profiles for JPEG2000", Signal Processing: Image Communication 25, 502-516, Sections 2.1.2 and 2.1.3.
The percentile summary is a modification of the paper's histogram score.

The 5th percentile follows the sharpest edges, so a blur along one axis only (motion blur) moves ``blur``
weakly and unevenly and shows in ``directionality`` instead. Noise and JPEG artifacts change ``blur`` too, by
amounts and in directions that depend on the scene, so it cannot tell them from focus.
"""

__all__ = []

from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

MIN_POINTS = 100  # fewer edge points than this and no value is reported
MIN_AXIS_POINTS = 50  # fewer on either axis and directionality is NaN
_SIGMA = 1.0  # smoothing before edge detection


class BlurValues(NamedTuple):
    """The two values of :func:`blur_stat`, both NaN when the image cannot support them."""

    blur: float
    directionality: float


def _peak_mask(g: NDArray[np.floating[Any]], gg: NDArray[np.floating[Any]], axis: int) -> NDArray[np.bool_]:
    """Pixels where the gradient `g` peaks along `axis` and is at least its mean magnitude (Eq. 1)."""
    sign = np.sign(g)
    mask = (
        (sign * np.roll(gg, 1, axis=axis) > 0)
        & (sign * np.roll(gg, -1, axis=axis) < 0)
        & (np.abs(g) >= np.abs(g).mean())
    )
    # np.roll wraps around, so the border pixels have no real neighbor
    if axis == 1:
        mask[:, [0, -1]] = False
    else:
        mask[[0, -1], :] = False
    return mask


def _edge_masks(gray: NDArray[np.float32]) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Horizontal-gradient and vertical-gradient edge point masks of the smoothed image."""
    smooth = ndimage.gaussian_filter(gray, _SIGMA, truncate=3.0, mode="mirror")
    gx = ndimage.sobel(smooth, axis=1, mode="mirror")
    gy = ndimage.sobel(smooth, axis=0, mode="mirror")
    return (
        _peak_mask(gx, ndimage.sobel(gx, axis=1, mode="mirror"), axis=1),
        _peak_mask(gy, ndimage.sobel(gy, axis=0, mode="mirror"), axis=0),
    )


def edge_points(gray: NDArray[np.float32]) -> NDArray[np.intp]:
    """Edge points of a 2D float32 image, as (N, 2) rows of (y, x), sorted and unique.

    A pixel is an edge point where the smoothed gradient peaks along its own axis (horizontal or vertical)
    and is at least the image's mean gradient magnitude. `np.argwhere` reads the mask row by row, so the rows
    come out sorted and unique.
    """
    hmask, vmask = _edge_masks(gray)
    return np.argwhere(hmask | vmask)


def _spread_axis1(m: NDArray[np.float64], ys: NDArray[np.intp], xs: NDArray[np.intp]) -> NDArray[np.float64]:
    """Spread of the profile `m[y]` around column x for each point: sqrt of the mean squared offset, weighted by m.

    Each profile runs from the nearest local minimum on the left to the nearest on the right (or the border).
    Those ends come from running minima over the rows, and the weighted sums from prefix sums, so the cost
    does not depend on how wide the profiles are.
    """
    h, n = m.shape
    mid = m[:, 1:-1]
    local_min = np.zeros(m.shape, dtype=np.bool_)
    local_min[:, 1:-1] = (mid <= m[:, :-2]) & (mid <= m[:, 2:])
    idx = np.arange(n)
    # nearest local minimum at or right of each column (n - 1 if none), and at or left of it (0 if none)
    nxt = np.minimum.accumulate(np.where(local_min, idx, n - 1)[:, ::-1], axis=1)[:, ::-1]
    prv = np.maximum.accumulate(np.where(local_min, idx, 0), axis=1)
    # right end: first local minimum strictly right of x; left end: last strictly left of x
    r = np.where(xs < n - 1, nxt[ys, np.minimum(xs + 1, n - 1)], xs)
    left = np.where(xs > 0, prv[ys, np.maximum(xs - 1, 0)], xs)
    # ponytail: the prefix sums cancel as n^3 (error 7.6e-7 at 2720x1530, 1.4e-6 to 2.7e-6 at 3840x2160, 4.4e-5 at
    # 7680x1080, against a float32-magnitude loop), which a 5th-percentile statistic does not notice; block-reset or
    # centered prefix sums if it ever matters.
    zero = np.zeros((h, 1))
    c0 = np.concatenate([zero, np.cumsum(m, axis=1)], axis=1)
    c1 = np.concatenate([zero, np.cumsum(m * idx, axis=1)], axis=1)
    c2 = np.concatenate([zero, np.cumsum(m * idx.astype(np.float64) ** 2, axis=1)], axis=1)
    s0 = c0[ys, r + 1] - c0[ys, left]
    s1 = c1[ys, r + 1] - c1[ys, left]
    s2 = c2[ys, r + 1] - c2[ys, left]
    # sum of m * (i - x)^2 over the profile, divided by the sum of m
    var = (s2 - 2 * xs * s1 + xs.astype(np.float64) ** 2 * s0) / s0
    return np.sqrt(np.maximum(var, 0.0))


def edge_widths(gray: NDArray[np.float32], points: NDArray[np.intp]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Width of the edge at each (y, x) point, in pixels, and whether its dominant gradient is horizontal.

    Measured on the unsmoothed Sobel gradient. A point whose unsmoothed gradient is exactly zero has no
    direction, so it is dropped: the result holds only the kept points, in their order.
    """
    gx = ndimage.sobel(gray, axis=1, mode="mirror")
    gy = ndimage.sobel(gray, axis=0, mode="mirror")
    m = np.sqrt(gx**2 + gy**2).astype(np.float64)
    ys, xs = points[:, 0], points[:, 1]
    keep = m[ys, xs] > 0
    ys, xs = ys[keep], xs[keep]
    horizontal = np.abs(gx[ys, xs]) > np.abs(gy[ys, xs])
    vertical = ~horizontal
    widths = np.empty(len(ys), dtype=np.float64)
    # the spread along the axis of the gradient, times the cosine between the gradient and that axis
    widths[horizontal] = (
        _spread_axis1(m, ys[horizontal], xs[horizontal])
        * np.abs(gx[ys[horizontal], xs[horizontal]])
        / m[ys[horizontal], xs[horizontal]]
    )
    widths[vertical] = (
        _spread_axis1(np.ascontiguousarray(m.T), xs[vertical], ys[vertical])
        * np.abs(gy[ys[vertical], xs[vertical]])
        / m[ys[vertical], xs[vertical]]
    )
    return widths, horizontal


def blur_stat(gray: NDArray[np.floating[Any]]) -> BlurValues:
    """Edge-width blur values of a 2D grayscale image in the 0-255 display range.

    Any float dtype is accepted and cast to float32. `blur` is the 5th percentile of the edge widths in
    pixels at the image's own resolution and `directionality` is the absolute log2 ratio of the median width of
    horizontal-gradient edges to that of vertical-gradient edges.

    Both are NaN for input that is not 2D or is empty, with fewer than `MIN_POINTS` edge points, or with a median
    width of 0. `directionality` alone is NaN with fewer than `MIN_AXIS_POINTS` edge points on either axis, or
    when the median width on either axis is 0 (`blur` is unaffected).
    """
    nan = BlurValues(np.nan, np.nan)
    gray = np.asarray(gray, dtype=np.float32)
    if gray.ndim != 2 or gray.size == 0:
        return nan
    widths, horizontal = edge_widths(gray, edge_points(gray))
    if len(widths) < MIN_POINTS:  # before any percentile, which is undefined on too few points
        return nan
    p5, p50 = np.percentile(widths, [5, 50])
    if p50 == 0:
        return nan
    across, down = widths[horizontal], widths[~horizontal]
    median_across = np.median(across) if len(across) >= MIN_AXIS_POINTS else np.nan
    median_down = np.median(down) if len(down) >= MIN_AXIS_POINTS else np.nan
    directionality = abs(np.log2(median_across / median_down)) if median_across > 0 and median_down > 0 else np.nan
    return BlurValues(float(p5), float(directionality))
