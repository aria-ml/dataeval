from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import ndimage

import dataeval.core._blur as blur_module
from dataeval.core._blur import MIN_POINTS, BlurValues, blur_stat, edge_points, edge_widths


def textured() -> NDArray[np.float32]:
    """A seeded 256x256 image of random blobs with hard edges, values 0-255.

    Sixty overlapping rectangles at random gray levels give step edges along both axes. A coarse four-level
    texture on top keeps the profile between the edges from being exactly flat: a gradient profile with an
    exactly flat top, such as a box blur of a clean step, ties at every pixel and reads as sharp. Built from
    a local generator, never the global seed.
    """
    rng = np.random.default_rng(0)
    img = np.zeros((256, 256))
    for _ in range(60):
        h, w = rng.integers(20, 120, 2)
        y, x = rng.integers(0, 256 - h), rng.integers(0, 256 - w)
        img[y : y + h, x : x + w] = rng.uniform(0, 200)
    field = ndimage.gaussian_filter(rng.standard_normal((256, 256)), 3, mode="wrap")
    img += rng.permutation(np.linspace(0, 25, 4))[np.digitize(field, np.quantile(field, [0.25, 0.5, 0.75]))]
    return (255 * (img - img.min()) / (img.max() - img.min())).astype(np.float32)


def gaussian(img: NDArray[np.float32], sigma: float) -> NDArray[np.float32]:
    return img if sigma == 0 else np.asarray(ndimage.gaussian_filter(img, sigma), dtype=np.float32)


def _sobel(gray: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    gx, gy = ndimage.sobel(gray, axis=1, mode="mirror"), ndimage.sobel(gray, axis=0, mode="mirror")
    return np.asarray(gx, dtype=np.float32), np.asarray(gy, dtype=np.float32)


# The prototype's per-point loop (dataeval-prototype, src/FMV/blur/blur_stat.py, _profile_spread and edge_widths),
# copied unchanged except for three things: `l` is named `left`, the Sobel gradients come from SciPy instead of
# OpenCV, and the profile slice is cast to float64 before the weighted sums. The gradient magnitudes stay float32 as
# in the prototype, so the walk breaks near-ties at local minima exactly as it does; only the sums are widened.
def _reference_profile_spread(line, pos):
    # walk right until a local minimum of gradient magnitude (or the border)
    n = len(line)
    r = pos + 1
    while r < n - 1 and not (line[r] <= line[r - 1] and line[r] <= line[r + 1]):
        r += 1
    r = min(r, n - 1) if pos < n - 1 else pos
    # walk left the same way
    left = pos - 1
    while left > 0 and not (line[left] <= line[left - 1] and line[left] <= line[left + 1]):
        left -= 1
    left = max(left, 0) if pos > 0 else pos
    # Eq. 2: treat magnitudes as a distribution, take root of mean squared offset
    prof = line[left : r + 1].astype(np.float64)
    offs = np.arange(left, r + 1) - pos
    w = prof / prof.sum()
    return np.sqrt(np.sum(w * offs**2))


def _reference_widths(gray, pts):
    gx, gy = _sobel(gray)
    m = np.sqrt(gx**2 + gy**2)

    sigmas = []
    is_horizontal = []
    for y, x in pts:
        if abs(gx[y, x]) > abs(gy[y, x]):
            s = _reference_profile_spread(m[y, :], x)
            g = abs(gx[y, x]) / m[y, x]
            is_horizontal.append(True)
        else:
            s = _reference_profile_spread(m[:, x], y)
            g = abs(gy[y, x]) / m[y, x]
            is_horizontal.append(False)
        sigmas.append(g * s)
    return np.array(sigmas), np.array(is_horizontal)


def _nonzero_gradient(gray, pts):
    gx, gy = _sobel(gray)
    return pts[(gx[pts[:, 0], pts[:, 1]] != 0) | (gy[pts[:, 0], pts[:, 1]] != 0)]


@pytest.mark.required
class TestBlurStat:
    def test_zero_gradient_points_do_not_poison(self):
        """8-bit rounding leaves edge points whose unsmoothed gradient is exactly zero (0/0 is NaN)."""
        img = np.clip(np.round(textured() * 1.6 - 60), 0, 255).astype(np.float32)
        assert (img == 0).any()
        assert (img == 255).any()
        pts = edge_points(img)
        assert len(_nonzero_gradient(img, pts)) < len(pts), "fixture has no exactly-zero-gradient edge point"
        assert np.isfinite(blur_stat(img)).all()

    def test_blur_rises_with_sigma(self):
        img = textured()
        blurs = [blur_stat(gaussian(img, s)).blur for s in (0, 1, 2, 3)]
        assert np.isfinite(blurs).all()
        assert all(later > earlier for earlier, later in zip(blurs, blurs[1:], strict=False))

    def test_contrast_invariant(self):
        """Halving the contrast of a sharp 8-bit image leaves `blur` within 2%.

        This covers a sharp image only. A soft, low-contrast 8-bit image reads sharper than it is (see
        docs/source/concepts/BlurStatistics.md), and no test pins that as correct.
        """
        img = np.round(textured()).astype(np.uint8)
        half = np.round(128 + 0.5 * (img.astype(np.float64) - 128)).astype(np.uint8)
        assert abs(blur_stat(half.astype(np.float32)).blur / blur_stat(img.astype(np.float32)).blur - 1) < 0.02

    def test_directionality_sees_horizontal_blur(self):
        """A horizontal box blur widens only horizontal gradients: directionality rises, P5 blur does not."""
        img = textured()
        smeared = ndimage.uniform_filter1d(img, 15, axis=1)
        before, after = blur_stat(img), blur_stat(smeared)
        assert after.directionality >= before.directionality + 0.5
        assert abs(after.blur / before.blur - 1) < 0.15

    def test_widths_match_reference_loop(self):
        """The vectorized walk equals the prototype's loop to 1e-6 on the same points.

        The loop's sums are float64 because its float32 `prof / prof.sum()` rounds at about 1e-6 by itself, which
        would hide the port's error; its gradient magnitudes stay float32, so both pick the same profile ends.
        """
        img = gaussian(textured(), 1.0)
        pts = _nonzero_gradient(img, edge_points(img))
        assert len(pts) > 1000
        fast, fast_horizontal = edge_widths(img, pts)
        ref, ref_horizontal = _reference_widths(img, pts)
        assert fast.shape == ref.shape
        assert np.abs(fast - ref).max() < 1e-6
        assert (fast_horizontal == ref_horizontal).all()

    def test_edge_points_sorted_unique(self):
        pts = edge_points(textured())
        assert pts.dtype == np.intp
        assert pts.ndim == 2
        assert pts.shape[1] == 2
        assert np.array_equal(np.unique(pts, axis=0), pts)

    def test_nan_rules(self):
        step = np.zeros((30, 30), np.float32)
        step[:, 15:] = 200
        assert len(edge_points(step)) < MIN_POINTS
        ramp = np.tile(np.arange(64, dtype=np.float32) * 4, (64, 1))
        for img in (np.full((64, 64), 128.0, np.float32), step, ramp, np.stack([textured()] * 3), np.zeros((0, 5))):
            result = blur_stat(img)
            assert isinstance(result, BlurValues)
            assert np.isnan(result).all()

    def test_directionality_needs_both_axes(self):
        """Vertical stripes have only horizontal gradients: blur is finite, directionality is not."""
        stripes = np.tile(np.where(np.arange(256) // 8 % 2 == 0, 40.0, 210.0).astype(np.float32), (256, 1))
        result = blur_stat(stripes)
        assert np.isfinite(result.blur)
        assert np.isnan(result.directionality)

    def test_zero_axis_median_only_hides_directionality(self):
        """One-pixel vertical lines have zero-width flanks: the horizontal median is 0, the overall median is not."""
        img = np.where(np.arange(256)[:, None] // 8 % 2 == 0, 60.0, 160.0) * np.ones((256, 256))
        img[:, 16::32] = 255
        img = img.astype(np.float32)
        widths, horizontal = edge_widths(img, edge_points(img))
        assert np.median(widths[horizontal]) == 0
        assert np.median(widths) > 0
        result = blur_stat(img)
        assert np.isfinite(result.blur)
        assert np.isnan(result.directionality)

    def test_zero_median_width_is_nan(self, monkeypatch):
        monkeypatch.setattr(blur_module, "edge_widths", lambda gray, pts: (np.zeros(200), np.ones(200, bool)))
        assert np.isnan(blur_stat(textured())).all()

    def test_deterministic_and_pure(self):
        img = textured()
        original = img.copy()
        state = cast("tuple[Any, ...]", np.random.get_state())
        first, second = blur_stat(img), blur_stat(img)
        assert first == second
        assert (img == original).all()
        assert blur_stat(img.astype(np.float64)) == first
        after = cast("tuple[Any, ...]", np.random.get_state())
        assert state[0] == after[0]
        assert np.array_equal(state[1], after[1])
        assert state[2:] == after[2:]
