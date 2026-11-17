"""noise_immerkaer: Immerkaer's noise estimate per channel, with non-finite pixels left out.

Every fixture is synthetic: white Gaussian noise from a fixed seed, planes, edges and impulses whose response to the
3x3 mask is known in closed form.
"""

import math
import warnings

import numpy as np
import pytest

from dataeval.core import NoiseResult, noise_immerkaer

pytestmark = pytest.mark.required

SCALE = math.sqrt(math.pi / 2) / 6  # noise_std = SCALE * mean(|g|)


def _noise(shape: tuple[int, ...], sigma: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, sigma, size=shape)


def _std(image, **kwargs) -> float:
    return float(noise_immerkaer(image, **kwargs)["noise_std"][0])


class TestClosedForm:
    def test_impulse_reads_the_mean_absolute_mask_coefficient(self):
        image = np.zeros((5, 5))
        image[2, 2] = 1.0
        result = noise_immerkaer(image, min_neighborhoods=9)
        # each of the 9 windows holds the impulse once, so |g| runs over the 9 mask coefficients, which sum to 16
        assert result["support"].tolist() == [9]
        assert result["noise_std"][0] == pytest.approx(SCALE * 16 / 9, rel=1e-12)
        assert round(float(result["noise_std"][0]), 4) == 0.3714

    def test_result_shapes_and_dtypes(self):
        result: NoiseResult = noise_immerkaer(np.zeros((3, 12, 12)))
        assert set(result) == {"noise_std", "support"}
        assert result["noise_std"].shape == (3,)
        assert result["noise_std"].dtype == np.float64
        assert result["support"].shape == (3,)
        assert result["support"].dtype == np.int64


class TestNullSpace:
    """The mask removes any function of the row plus a function of the column; other structure adds to the reading."""

    rows, cols = np.mgrid[0:12, 0:12]

    def test_constant_reads_exactly_zero(self):
        assert _std(np.full((12, 12), 37.5)) == 0.0

    def test_integer_plane_reads_exactly_zero(self):
        assert _std(3 * self.rows + 5 * self.cols + 7) == 0.0

    def test_float_plane_reads_zero(self):
        assert _std(0.37 * self.rows + 1.13 * self.cols + 0.5) < 1e-12

    @pytest.mark.parametrize(
        "image",
        [
            pytest.param((cols >= 6).astype(np.float64), id="vertical-edge"),
            pytest.param((rows >= 6).astype(np.float64), id="horizontal-edge"),
            pytest.param(np.sin(rows) + np.cos(cols), id="sin-row-plus-cos-col"),
        ],
    )
    def test_separable_patterns_read_zero(self, image):
        assert _std(image) < 1e-12

    def test_diagonal_edge_reads_above_zero(self):
        assert _std((self.rows > self.cols).astype(np.float64)) > 0.1


class TestWhiteNoise:
    @pytest.mark.parametrize(("sigma", "seed"), [(0.01, 1), (0.05, 2), (0.2, 3)])
    def test_large_image_reads_sigma(self, sigma, seed):
        std = _std(_noise((256, 256), sigma, seed))
        assert std**2 == pytest.approx(sigma**2, rel=0.10)
        assert std == pytest.approx(sigma, rel=0.05)

    def test_spread_at_the_default_floor(self):
        # 10x10 gives 64 neighborhoods, exactly the default floor; sigma 1, so the reading's square is the ratio
        draws = _noise((200, 10, 10), 1.0, 20261116)
        ratios = np.array([_std(draw) ** 2 for draw in draws])
        assert np.isfinite(ratios).all()
        assert np.median(ratios) == pytest.approx(1.0, rel=0.10)
        # measured 0.529 and 1.647 on these draws (0.557 and 1.659 over 20,000); the Notes quote about 0.55 and 1.65
        assert 0.50 < np.percentile(ratios, 5) < 0.60
        assert 1.60 < np.percentile(ratios, 95) < 1.72

    def test_affine_change_scales_by_the_gain(self):
        image = _noise((64, 64), 1.0, 4)
        base = _std(image)
        assert _std(-3.0 * image + 7.5) == pytest.approx(3.0 * base, rel=1e-12)
        assert _std(0.25 * image - 2.0) == pytest.approx(0.25 * base, rel=1e-12)

    def test_flat_bars_pull_the_reading_down_by_their_share(self):
        image = _noise((64, 128), 1.0, 5)
        bar = 34
        padded = np.pad(image, ((0, 0), (bar, bar)))
        # the share of response columns whose 3x3 window lies wholly in a bar
        s = 2 * (bar - 2) / (padded.shape[1] - 2)
        unpadded, read = _std(image), _std(padded)
        assert 0 < read < unpadded
        assert read / unpadded == pytest.approx(1 - s, rel=0.10)
        assert (read / unpadded) ** 2 == pytest.approx((1 - s) ** 2, rel=0.10)


class TestMaskedPixels:
    @pytest.mark.parametrize("hole", [np.nan, np.inf, -np.inf])
    def test_a_non_finite_pixel_removes_every_window_that_holds_it(self, hole):
        image = np.full((7, 7), 4.0)
        image[3, 3] = hole
        result = noise_immerkaer(image, min_neighborhoods=16)
        assert result["support"].tolist() == [16]  # 25 windows, 9 of them hold the center
        assert result["noise_std"][0] == 0.0  # no edge is made at the hole

    def test_the_same_image_without_the_hole_counts_every_window(self):
        result = noise_immerkaer(np.full((7, 7), 4.0), min_neighborhoods=16)
        assert result["support"].tolist() == [25]
        assert result["noise_std"][0] == 0.0


class TestFloor:
    def test_ten_by_ten_meets_the_default(self):
        result = noise_immerkaer(_noise((10, 10), 1.0, 6))
        assert result["support"].tolist() == [64]
        assert np.isfinite(result["noise_std"][0])

    def test_nine_by_nine_is_nan_and_reports_its_support(self):
        image = _noise((9, 9), 1.0, 7)
        result = noise_immerkaer(image)
        assert result["support"].tolist() == [49]
        assert np.isnan(result["noise_std"][0])
        assert np.isfinite(_std(image, min_neighborhoods=49))
        assert np.isfinite(_std(image, min_neighborhoods=np.int64(49)))

    @pytest.mark.parametrize("shape", [(2, 2), (1, 50), (50, 2), (3, 2, 9), (0, 0)])
    def test_under_three_by_three_has_no_support(self, shape):
        result = noise_immerkaer(np.ones(shape), min_neighborhoods=1)
        assert (result["support"] == 0).all()
        assert np.isnan(result["noise_std"]).all()

    @pytest.mark.parametrize("value", [0, -1, 2.5, True, np.nan, "8"])
    def test_invalid_floor_raises_naming_it(self, value):
        with pytest.raises(ValueError, match="min_neighborhoods"):
            noise_immerkaer(np.zeros((10, 10)), min_neighborhoods=value)


class TestInput:
    def test_two_dimensional_is_one_channel(self):
        image = _noise((16, 16), 1.0, 8)
        flat, stacked = noise_immerkaer(image), noise_immerkaer(image[None])
        assert flat["noise_std"].shape == (1,)
        np.testing.assert_array_equal(flat["noise_std"], stacked["noise_std"])
        np.testing.assert_array_equal(flat["support"], stacked["support"])

    @pytest.mark.parametrize("shape", [(16,), (2, 3, 16, 16)])
    def test_other_dimensions_raise(self, shape):
        with pytest.raises(ValueError, match=r"\(H, W\) or \(C, H, W\)"):
            noise_immerkaer(np.zeros(shape))

    def test_unsigned_integers_do_not_wrap(self):
        checkerboard = (np.indices((16, 16)).sum(axis=0) % 2 * 255).astype(np.uint8)
        as_uint8, as_float = _std(checkerboard), _std(checkerboard.astype(np.float64))
        assert as_float > 0
        assert as_uint8 == as_float

    def test_channel_order_permutes_the_output(self):
        image = np.stack([_noise((16, 16), sigma, 9 + k) for k, sigma in enumerate([0.5, 1.0, 2.0])])
        image[1, 4:7, 4:7] = np.nan
        forward, backward = noise_immerkaer(image), noise_immerkaer(image[::-1])
        np.testing.assert_array_equal(backward["noise_std"], forward["noise_std"][::-1])
        np.testing.assert_array_equal(backward["support"], forward["support"][::-1])

    def test_input_is_not_changed(self):
        image = _noise((3, 16, 16), 1.0, 12)
        image[0, 2, 2] = np.nan
        image[2, 5, 5] = np.inf
        before = image.copy()
        noise_immerkaer(image)
        np.testing.assert_array_equal(image, before)


def test_raises_no_warning():
    images = [
        _noise((3, 16, 16), 1.0, 13),
        np.full((10, 10), np.nan),
        np.full((10, 10), np.inf),
        np.where(np.eye(12, dtype=bool), np.nan, 1.0),
        np.array([[np.inf, -np.inf, 1.0]] * 3),
        np.full((10, 10), 1e308),
        np.arange(144, dtype=np.int16).reshape(12, 12) - 70,
        np.zeros((2, 2)),
        np.zeros((0, 0)),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for image in images:
            noise_immerkaer(image, min_neighborhoods=1)
            noise_immerkaer(image)
