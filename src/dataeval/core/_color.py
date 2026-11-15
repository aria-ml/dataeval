"""The sRGB to CIELAB conversion behind the color statistics, and the additive sums they are read from."""

__all__ = []

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

# Linear sRGB to XYZ for the D65 white, computed from the sRGB primaries and that white, to seven digits. The four-digit
# matrix printed in IEC 61966-2-1 puts red's a* at 80.1093 instead of 80.0925 and leaves a gray ramp with |b*| up to
# 0.0104, so the digits are not a choice.
_M = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_WHITE = np.array([0.95047, 1.0, 1.08883])
_ENCODED_KNEE = 0.04045
_EPSILON = 216 / 24389
_KAPPA = 24389 / 27


def srgb_to_lab(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert sRGB to CIELAB (D65, 2-degree observer).

    Parameters
    ----------
    rgb : NDArray[np.float64]
        Encoded sRGB with red, green and blue on the last axis, each in ``[0, 1]``. A value outside the interval is
        not meaningful here, so callers keep it out.

    Returns
    -------
    NDArray[np.float64]
        ``L*``, ``a*`` and ``b*`` on the last axis, the shape of `rgb`.
    """
    linear = np.where(rgb <= _ENCODED_KNEE, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    # Three multiply-adds and not a matrix product: a BLAS matmul rounds differently for a different shape or machine,
    # and a pixel's color must not depend on how many pixels are read with it.
    xyz = (linear[..., :1] * _M[:, 0] + linear[..., 1:2] * _M[:, 1] + linear[..., 2:] * _M[:, 2]) / _WHITE
    f = np.where(xyz > _EPSILON, np.cbrt(xyz), (_KAPPA * xyz + 16) / 116)
    # Below the knee L* is exactly kappa * Y, which keeps black at 0 and not at a rounding residue of 116 * 16/116 - 16.
    lightness = np.where(xyz[..., 1] > _EPSILON, 116 * f[..., 1] - 16, _KAPPA * xyz[..., 1])
    return np.stack([lightness, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


class ColorSums(NamedTuple):
    """
    What a set of valid pixels adds up to, so that a view read in blocks equals the view read at once.

    Attributes
    ----------
    pixels : int
        Valid pixels.
    lightness : float
        Sum of ``L*``.
    chroma : float
        Sum of chroma, ``hypot(a*, b*)``.
    achromatic : int
        Pixels with chroma under the threshold.
    a_chromatic, b_chromatic : float
        Sums of ``a*`` and ``b*`` over the chromatic pixels, those at or over the threshold.
    chroma_chromatic : float
        Sum of chroma over the chromatic pixels.
    chromatic : int
        Chromatic pixels.
    """

    pixels: int
    lightness: float
    chroma: float
    achromatic: int
    a_chromatic: float
    b_chromatic: float
    chroma_chromatic: float
    chromatic: int

    def __add__(self, other: "ColorSums") -> "ColorSums":  # type: ignore[override]
        """Add two sets of pixels field by field, not as tuples."""
        return ColorSums(
            self.pixels + other.pixels,
            self.lightness + other.lightness,
            self.chroma + other.chroma,
            self.achromatic + other.achromatic,
            self.a_chromatic + other.a_chromatic,
            self.b_chromatic + other.b_chromatic,
            self.chroma_chromatic + other.chroma_chromatic,
            self.chromatic + other.chromatic,
        )


def _sums_from_lab(lab: NDArray[np.float64], chroma_threshold: float) -> ColorSums:
    """Sum ``(N, 3)`` CIELAB pixels, splitting them at `chroma_threshold`."""
    lightness, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    chroma = np.hypot(a, b)
    chromatic = chroma >= chroma_threshold
    return ColorSums(
        pixels=int(lab.shape[0]),
        lightness=float(lightness.sum()),
        chroma=float(chroma.sum()),
        achromatic=int(np.count_nonzero(~chromatic)),
        a_chromatic=float(a[chromatic].sum()),
        b_chromatic=float(b[chromatic].sum()),
        chroma_chromatic=float(chroma[chromatic].sum()),
        chromatic=int(np.count_nonzero(chromatic)),
    )


def block_sums(rgb: NDArray[np.float64], chroma_threshold: float) -> ColorSums:
    """
    Sum the color of ``(N, 3)`` valid sRGB pixels in ``[0, 1]``.

    Parameters
    ----------
    rgb : NDArray[np.float64]
        Valid pixels, one per row.
    chroma_threshold : float
        Chroma at or over which a pixel is chromatic.

    Returns
    -------
    ColorSums
        The sums, which add across blocks of one view.
    """
    return _sums_from_lab(srgb_to_lab(rgb), chroma_threshold)
