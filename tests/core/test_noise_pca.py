"""noise_pca: the smallest eigenvalue of the covariance of 7x7 patches, corrected for the Marchenko-Pastur edge.

Every fixture is synthetic: white Gaussian noise from a fixed seed, constants, NaN blocks whose effect on the count of
clear patches is known in closed form, and a noise-free zone plate.
"""

import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest

from dataeval.config import use_seed
from dataeval.core import NoiseResult, noise_pca
from dataeval.exceptions import ExperimentalWarning

pytestmark = [pytest.mark.required, pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")]

PATCH = 7
FLOOR = 784  # clear patches a channel needs: 28x28 positions, so a 34x34 region with nothing masked


def _noise(shape: tuple[int, ...], sigma: float, seed: int | list[int]) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, sigma, size=shape)


def _std(image, **kwargs) -> float:
    return float(noise_pca(image, **kwargs)["noise_std"][0])


def _positions(height: int, width: int) -> int:
    return max(height - PATCH + 1, 0) * max(width - PATCH + 1, 0)


class TestWhiteNoise:
    @pytest.mark.parametrize(("side", "tolerance"), [(34, 0.10), (130, 0.05)])
    def test_median_reads_the_variance(self, side, tolerance):
        # 34x34 holds exactly the floor of 784 patches and reads about 4% high there (a median of 1.042 over 4000
        # draws), so its bound is 10%; 130x130 holds more than the default 2000, so 2000 are used (1.019 here)
        ratios = [_std(_noise((side, side), 10.0, [side, d])) ** 2 / 100 for d in range(200)]
        assert np.isfinite(ratios).all()
        assert np.median(ratios) == pytest.approx(1.0, rel=tolerance)

    def test_affine_change_scales_by_the_gain(self):
        image = _noise((64, 64), 1.0, 1)
        base = _std(image)
        assert _std(-3.0 * image + 7.5) == pytest.approx(3.0 * base, rel=1e-9)
        assert _std(0.25 * image - 2.0) == pytest.approx(0.25 * base, rel=1e-9)


class TestStructure:
    """Structure whose patches span few of the 49 directions reads 0; structure that fills them reads as noise."""

    rows, cols = np.mgrid[0:64, 0:64]

    def test_constant_reads_exactly_zero(self):
        assert _std(np.full((40, 40), 37.5)) == 0.0
        assert _std(np.full((40, 40), 200, dtype=np.uint8)) == 0.0

    @pytest.mark.parametrize(
        "image",
        [
            pytest.param((cols >= 30).astype(np.float64), id="vertical-edge"),
            pytest.param((rows > cols).astype(np.float64), id="diagonal-edge"),
            pytest.param(((rows + cols) % 5 < 2).astype(np.float64), id="diagonal-stripes"),
            pytest.param(np.sin((rows + cols) ** 2 / 50.0), id="diagonal-chirp-stripes"),
        ],
    )
    def test_straight_edges_and_stripes_read_zero(self, image):
        # a function of one direction gives patches that hold at most 13 distinct values, far fewer than 49
        assert _std(image) < 1e-6

    def test_noise_free_texture_reads_as_noise(self):
        # a zone plate of amplitude 1 holds no noise, but its patches fill every direction (measured: 0.43)
        assert _std(np.cos((self.rows**2 + self.cols**2) / 10.0)) > 0.1


class TestMaskedPixels:
    def test_a_nan_block_removes_exactly_the_patches_that_touch_it(self):
        image = _noise((40, 40), 1.0, 2)
        image[10:13, 20:22] = np.nan
        result = noise_pca(image)
        # a patch at (i, j) covers rows i..i+6 and columns j..j+6, so 9 rows by 8 columns of positions touch the block
        assert result["support"].tolist() == [_positions(40, 40) - 9 * 8]
        assert np.isfinite(result["noise_std"][0])

    @pytest.mark.parametrize("hole", [np.nan, np.inf, -np.inf])
    def test_any_non_finite_value_masks(self, hole):
        image = _noise((40, 40), 1.0, 3)
        image[20, 20] = hole
        assert noise_pca(image)["support"].tolist() == [_positions(40, 40) - PATCH * PATCH]

    def test_when_all_clear_patches_fit_every_one_is_used(self):
        image = _noise((40, 40), 1.0, 4)
        image[0, 0] = np.nan
        clear = _positions(40, 40) - 1
        assert _std(image, n_patches=clear) == _std(image, n_patches=5000)


class TestFloor:
    def test_the_floor_is_met_at_34x34(self):
        result = noise_pca(_noise((34, 34), 1.0, 5))
        assert result["support"].tolist() == [FLOOR]
        assert np.isfinite(result["noise_std"][0])

    def test_under_the_floor_is_nan_and_reports_its_support(self):
        result = noise_pca(_noise((33, 33), 1.0, 6))
        assert result["support"].tolist() == [_positions(33, 33)]
        assert np.isnan(result["noise_std"][0])

    @pytest.mark.parametrize("shape", [(6, 100), (100, 6), (2, 2), (0, 0), (3, 6, 6)])
    def test_under_one_patch_has_no_support(self, shape):
        result = noise_pca(np.ones(shape))
        assert (result["support"] == 0).all()
        assert np.isnan(result["noise_std"]).all()

    def test_masked_and_overflowing_channels_are_nan_without_raising(self):
        image = np.stack([np.full((40, 40), np.nan), np.full((40, 40), np.inf), _noise((40, 40), 1.0, 7)])
        image = np.concatenate([image, np.where(np.indices((1, 40, 40)).sum(axis=0) % 2, 1e308, -1e308)])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.simplefilter("ignore", ExperimentalWarning)
            result = noise_pca(image)
        assert result["support"].tolist() == [0, 0, _positions(40, 40), _positions(40, 40)]
        assert np.isnan(result["noise_std"][[0, 1, 3]]).all()
        assert np.isfinite(result["noise_std"][2])

    @pytest.mark.parametrize("value", [FLOOR - 1, 0, -1, 2000.5, True, np.nan, "2000", None])
    def test_invalid_n_patches_raises_naming_it(self, value):
        with pytest.raises(ValueError, match="n_patches"):
            noise_pca(np.zeros((40, 40)), n_patches=value)

    def test_numpy_integers_are_accepted(self):
        image = _noise((40, 40), 1.0, 8)
        assert _std(image, n_patches=np.int64(FLOOR)) == _std(image, n_patches=FLOOR)


class TestInput:
    def test_result_shapes_and_dtypes(self):
        result: NoiseResult = noise_pca(np.zeros((3, 40, 40)))
        assert set(result) == {"noise_std", "support"}
        assert result["noise_std"].shape == (3,)
        assert result["noise_std"].dtype == np.float64
        assert result["support"].shape == (3,)
        assert result["support"].dtype == np.int64

    def test_two_dimensional_is_one_channel(self):
        image = _noise((40, 40), 1.0, 9)
        flat, stacked = noise_pca(image), noise_pca(image[None])
        assert flat["noise_std"].shape == (1,)
        np.testing.assert_array_equal(flat["noise_std"], stacked["noise_std"])
        np.testing.assert_array_equal(flat["support"], stacked["support"])

    @pytest.mark.parametrize("shape", [(40,), (2, 3, 40, 40)])
    def test_other_dimensions_raise(self, shape):
        with pytest.raises(ValueError, match=r"\(H, W\) or \(C, H, W\)"):
            noise_pca(np.zeros(shape))

    def test_unsigned_integers_do_not_wrap(self):
        image = np.random.default_rng(10).integers(0, 256, size=(40, 40)).astype(np.uint8)
        as_uint8, as_float = _std(image), _std(image.astype(np.float64))
        assert as_float > 0
        assert as_uint8 == as_float

    def test_channel_order_permutes_the_output(self):
        image = np.stack([_noise((40, 40), sigma, 11 + k) for k, sigma in enumerate([0.5, 1.0, 2.0])])
        image[1, 4:7, 4:7] = np.nan
        forward, backward = noise_pca(image), noise_pca(image[::-1])
        np.testing.assert_array_equal(backward["noise_std"], forward["noise_std"][::-1])
        np.testing.assert_array_equal(backward["support"], forward["support"][::-1])

    def test_input_is_not_changed(self):
        image = _noise((3, 40, 40), 1.0, 14)
        image[0, 2, 2] = np.nan
        image[2, 5, 5] = np.inf
        before = image.copy()
        noise_pca(image)
        np.testing.assert_array_equal(image, before)


class TestDeterminism:
    def test_bit_identical_whatever_the_seed(self):
        image = _noise((3, 80, 80), 1.0, 15)
        state = np.random.get_state()
        try:
            with use_seed(None):
                first = noise_pca(image)
            np.random.seed(12345)
            with use_seed(12345):
                second = noise_pca(image)
        finally:
            np.random.set_state(state)
        np.testing.assert_array_equal(first["noise_std"], second["noise_std"])
        np.testing.assert_array_equal(first["support"], second["support"])


class TestExperimental:
    def test_the_first_call_warns_once(self):
        code = textwrap.dedent(
            """
            import warnings
            import numpy as np
            from dataeval.core import noise_pca
            from dataeval.exceptions import ExperimentalWarning

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                noise_pca(np.zeros((40, 40)))
                noise_pca(np.zeros((40, 40)))
            print(sum(issubclass(w.category, ExperimentalWarning) for w in caught), len(caught))
            """,
        )
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert done.stdout.strip() == "1 1"
