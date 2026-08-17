import logging
import warnings
from unittest.mock import patch

import numpy as np
import pytest
from scipy.stats import entropy as scipy_entropy
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew

from dataeval.core import combine_stats_results, compute_stats
from dataeval.core._calculators._base import Calculator, Handler, ViewKind
from dataeval.core._calculators._cache import CalculatorCache
from dataeval.core._calculators._dimensionstats import DimensionStatCalculator
from dataeval.core._calculators._hashstats import HashStatCalculator
from dataeval.core._calculators._pixelstats import PixelStatCalculator
from dataeval.core._calculators._visualstats import QUARTILES, VisualStatCalculator
from dataeval.flags import ImageStats
from dataeval.types import SourceIndex
from dataeval.utils.preprocessing import BoundingBox, get_value_range, rescale


class TestPixelStats:
    @pytest.mark.parametrize("n_channels", [1, 3])
    def test_process_basic_stats(self, n_channels):
        """Test pixel statistics calculation."""
        # Create deterministic image
        images = [np.random.random((n_channels, 10, 10))]

        result = compute_stats(images, stats=ImageStats.PIXEL, per_channel=False)

        assert "mean" in result["stats"]
        assert "std" in result["stats"]
        assert "var" in result["stats"]
        assert "skew" in result["stats"]
        assert "kurtosis" in result["stats"]
        assert "entropy" in result["stats"]
        assert "missing" in result["stats"]
        assert "zeros" in result["stats"]
        assert "histogram" in result["stats"]

        assert len(result["stats"]["mean"]) == 1
        assert result["stats"]["mean"].dtype == np.float32
        assert len(result["stats"]["histogram"][0]) == 256

    def test_process_with_nans(self):
        """Test pixel statistics with NaN values."""
        images = [np.array([[[np.nan, 0.5], [0.5, 0.5]]])]

        result = compute_stats(images, stats=ImageStats.PIXEL, per_channel=False)
        assert result["stats"]["missing"][0] > 0

    def test_missing_global_mode_counts_all_channels(self):
        """Test that global missing calculation counts across all channels correctly.

        Regression test for bug where denominator only counted H×W instead of C×H×W.
        """
        # Create a 3-channel image (3, 2, 2) with specific NaN pattern
        # Channel 0: 1 NaN out of 4 pixels
        # Channel 1: 2 NaNs out of 4 pixels
        # Channel 2: 0 NaNs out of 4 pixels
        # Total: 3 NaNs out of 12 pixel values
        images = [
            np.array(
                [
                    [[np.nan, 1.0], [1.0, 1.0]],  # Channel 0: 1 NaN
                    [[np.nan, np.nan], [1.0, 1.0]],  # Channel 1: 2 NaNs
                    [[1.0, 1.0], [1.0, 1.0]],  # Channel 2: 0 NaNs
                ],
            ),
        ]

        result = compute_stats(images, stats=ImageStats.PIXEL_MISSING, per_channel=False)

        # Global mode should count: 3 NaN values / 12 total values = 0.25
        expected_missing = 3 / 12
        assert result["stats"]["missing"][0] == pytest.approx(expected_missing, abs=1e-4)

    def test_missing_per_channel_mode(self):
        """Test that per-channel missing calculation is correct."""
        # Same image as above
        images = [
            np.array(
                [
                    [[np.nan, 1.0], [1.0, 1.0]],  # Channel 0: 1 NaN / 4 = 0.25
                    [[np.nan, np.nan], [1.0, 1.0]],  # Channel 1: 2 NaN / 4 = 0.5
                    [[1.0, 1.0], [1.0, 1.0]],  # Channel 2: 0 NaN / 4 = 0.0
                ],
            ),
        ]

        result = compute_stats(images, stats=ImageStats.PIXEL_MISSING, per_channel=True)

        # Per-channel mode should return list with one value per channel
        assert len(result["stats"]["missing"]) == 3
        assert result["stats"]["missing"][0] == pytest.approx(0.25, abs=1e-4)
        assert result["stats"]["missing"][1] == pytest.approx(0.5, abs=1e-4)
        assert result["stats"]["missing"][2] == pytest.approx(0.0, abs=1e-4)

    def test_missing_single_channel_image(self):
        """Test missing calculation for single-channel image."""
        # Single channel (1, 3, 3) with 2 NaNs out of 9 pixels
        images = [np.array([[[np.nan, 1.0, 1.0], [1.0, np.nan, 1.0], [1.0, 1.0, 1.0]]])]

        result = compute_stats(images, stats=ImageStats.PIXEL_MISSING, per_channel=False)

        # 2 NaNs / 9 total = 0.222...
        expected_missing = 2 / 9
        assert result["stats"]["missing"][0] == pytest.approx(expected_missing, abs=1e-4)

    def test_zeros_global_mode_counts_all_channels(self):
        """Test that global zeros calculation counts across all channels correctly.

        Regression test for bug where global mode counted spatial positions where
        all channels were zero, instead of counting individual zero pixel values.
        """
        # Create a 3-channel image (3, 2, 2) with specific zero pattern
        # Channel 0: 1 zero out of 4 pixels
        # Channel 1: 2 zeros out of 4 pixels
        # Channel 2: 0 zeros out of 4 pixels
        # Total: 3 zeros out of 12 pixel values
        images = [
            np.array(
                [
                    [[0.0, 1.0], [1.0, 1.0]],  # Channel 0: 1 zero
                    [[0.0, 0.0], [1.0, 1.0]],  # Channel 1: 2 zeros
                    [[1.0, 1.0], [1.0, 1.0]],  # Channel 2: 0 zeros
                ],
            ),
        ]

        result = compute_stats(images, stats=ImageStats.PIXEL_ZEROS, per_channel=False)

        # Global mode should count: 3 zero values / 12 total values = 0.25
        expected_zeros = 3 / 12
        assert result["stats"]["zeros"][0] == pytest.approx(expected_zeros, abs=1e-4)

    def test_zeros_per_channel_mode(self):
        """Test that per-channel zeros calculation is correct."""
        # Same image as above
        images = [
            np.array(
                [
                    [[0.0, 1.0], [1.0, 1.0]],  # Channel 0: 1 zero / 4 = 0.25
                    [[0.0, 0.0], [1.0, 1.0]],  # Channel 1: 2 zeros / 4 = 0.5
                    [[1.0, 1.0], [1.0, 1.0]],  # Channel 2: 0 zeros / 4 = 0.0
                ],
            ),
        ]

        result = compute_stats(images, stats=ImageStats.PIXEL_ZEROS, per_channel=True)

        # Per-channel mode should return list with one value per channel
        assert len(result["stats"]["zeros"]) == 3
        assert result["stats"]["zeros"][0] == pytest.approx(0.25, abs=1e-4)
        assert result["stats"]["zeros"][1] == pytest.approx(0.5, abs=1e-4)
        assert result["stats"]["zeros"][2] == pytest.approx(0.0, abs=1e-4)

    def test_zeros_single_channel_image(self):
        """Test zeros calculation for single-channel image."""
        # Single channel (1, 3, 3) with 2 zeros out of 9 pixels
        images = [np.array([[[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]])]

        result = compute_stats(images, stats=ImageStats.PIXEL_ZEROS, per_channel=False)

        # 2 zeros / 9 total = 0.222...
        expected_zeros = 2 / 9
        assert result["stats"]["zeros"][0] == pytest.approx(expected_zeros, abs=1e-4)

    def test_zeros_all_zeros_image(self):
        """Test zeros calculation when entire image is zeros."""
        images = [np.zeros((3, 10, 10))]

        result = compute_stats(images, stats=ImageStats.PIXEL_ZEROS, per_channel=False)

        # All pixels are zero, so should be 1.0
        assert result["stats"]["zeros"][0] == pytest.approx(1.0, abs=1e-4)

    def test_missing_all_nans_image(self):
        """Test missing calculation when entire image is NaN."""
        images = [np.full((3, 10, 10), np.nan)]

        result = compute_stats(images, stats=ImageStats.PIXEL_MISSING, per_channel=False)

        # All pixels are NaN, so should be 1.0
        assert result["stats"]["missing"][0] == pytest.approx(1.0, abs=1e-4)


class TestSkewKurtosisEntropySciPyEquivalence:
    """Verify our NumPy implementations match scipy.stats for skew, kurtosis, and entropy."""

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def _make_calculator(self, image, per_channel=False):
        cache = CalculatorCache(image, box=None, per_channel=per_channel)
        return PixelStatCalculator(image, cache, per_channel=per_channel)

    # --- Skew ---

    def test_skew_matches_scipy_global(self, rng):
        image = rng.random((3, 32, 32), dtype=np.float64)
        calc = self._make_calculator(image)
        result = calc._skew()
        expected = float(scipy_skew(calc.cache.scaled.ravel(), nan_policy="omit"))
        assert result[0] == pytest.approx(expected, rel=1e-5)

    def test_skew_matches_scipy_per_channel(self, rng):
        image = rng.random((3, 32, 32), dtype=np.float64)
        calc = self._make_calculator(image, per_channel=True)
        result = calc._skew()
        for ch in range(3):
            expected = float(scipy_skew(calc.cache.per_channel[ch], nan_policy="omit"))
            assert result[ch] == pytest.approx(expected, rel=1e-5)

    def test_skew_with_nans(self, rng):
        image = rng.random((1, 20, 20), dtype=np.float64)
        image[0, :5, :5] = np.nan
        calc = self._make_calculator(image)
        result = calc._skew()
        expected = float(scipy_skew(calc.cache.scaled.ravel(), nan_policy="omit"))
        assert result[0] == pytest.approx(expected, rel=1e-5)

    def test_skew_uniform_image_is_zero(self):
        image = np.full((1, 10, 10), 0.5)
        calc = self._make_calculator(image)
        result = calc._skew()
        assert result[0] == 0.0

    def test_skew_asymmetric_distribution(self):
        """Right-skewed data should produce positive skew."""
        rng = np.random.default_rng(99)
        vals = rng.exponential(scale=0.2, size=(1, 50, 50)).clip(0, 1)
        calc = self._make_calculator(vals)
        result = calc._skew()
        expected = float(scipy_skew(calc.cache.scaled.ravel(), nan_policy="omit"))
        assert result[0] == pytest.approx(expected, rel=1e-4)
        assert result[0] > 0  # exponential is right-skewed

    # --- Kurtosis ---

    def test_kurtosis_matches_scipy_global(self, rng):
        image = rng.random((3, 32, 32), dtype=np.float64)
        calc = self._make_calculator(image)
        result = calc._kurtosis()
        expected = float(scipy_kurtosis(calc.cache.scaled.ravel(), nan_policy="omit"))
        assert result[0] == pytest.approx(expected, rel=1e-5)

    def test_kurtosis_matches_scipy_per_channel(self, rng):
        image = rng.random((3, 32, 32), dtype=np.float64)
        calc = self._make_calculator(image, per_channel=True)
        result = calc._kurtosis()
        for ch in range(3):
            expected = float(scipy_kurtosis(calc.cache.per_channel[ch], nan_policy="omit"))
            assert result[ch] == pytest.approx(expected, rel=1e-5)

    def test_kurtosis_with_nans(self, rng):
        image = rng.random((1, 20, 20), dtype=np.float64)
        image[0, :5, :5] = np.nan
        calc = self._make_calculator(image)
        result = calc._kurtosis()
        expected = float(scipy_kurtosis(calc.cache.scaled.ravel(), nan_policy="omit"))
        assert result[0] == pytest.approx(expected, rel=1e-5)

    def test_kurtosis_uniform_image_is_zero(self):
        image = np.full((1, 10, 10), 0.5)
        calc = self._make_calculator(image)
        result = calc._kurtosis()
        assert result[0] == 0.0

    def test_kurtosis_heavy_tailed_is_positive(self):
        """Data from a heavy-tailed distribution should have positive excess kurtosis."""
        rng = np.random.default_rng(99)
        vals = rng.standard_t(df=3, size=(1, 100, 100)).clip(-5, 5)
        vals = (vals - vals.min()) / (vals.max() - vals.min())  # scale to [0,1]
        calc = self._make_calculator(vals)
        result = calc._kurtosis()
        expected = float(scipy_kurtosis(calc.cache.scaled.ravel(), nan_policy="omit"))
        assert result[0] == pytest.approx(expected, rel=1e-4)
        assert result[0] > 0  # t(3) has heavy tails

    # --- Entropy ---

    def test_entropy_matches_scipy_global(self, rng):
        image = rng.random((3, 32, 32), dtype=np.float64)
        calc = self._make_calculator(image)
        result = calc._entropy()
        h = calc.histogram.astype(np.float64)
        h = h / h.sum()
        expected = float(scipy_entropy(h))
        assert result[0] == pytest.approx(expected, rel=1e-10)

    def test_entropy_matches_scipy_per_channel(self, rng):
        image = rng.random((3, 32, 32), dtype=np.float64)
        calc = self._make_calculator(image, per_channel=True)
        result = calc._entropy()
        for ch in range(3):
            h = calc.histogram[ch].astype(np.float64)
            h = h / h.sum()
            expected = float(scipy_entropy(h))
            assert result[ch] == pytest.approx(expected, rel=1e-10)

    def test_entropy_uniform_histogram_is_max(self):
        """A perfectly uniform distribution should have maximum entropy = log(256)."""
        # Create image with values spread across all 256 bins
        vals = np.linspace(0, 1, 256 * 4, endpoint=False).reshape(1, 32, 32)
        calc = self._make_calculator(vals)
        result = calc._entropy()
        # Not perfectly uniform due to binning, but should be close to log(256)
        assert result[0] > 5.0  # log(256) ≈ 5.545

    def test_entropy_single_value_is_zero(self):
        """A constant image has zero entropy."""
        image = np.full((1, 10, 10), 0.5)
        calc = self._make_calculator(image)
        result = calc._entropy()
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_entropy_all_nan_is_nan(self):
        """An all-NaN image was not measured, so its entropy is NaN, not zero.

        Zero is a legitimate-looking extreme — a perfectly flat image has zero entropy —
        so returning it for unmeasured data makes an out-of-bounds box, or an image whose
        boxes cover it entirely, indistinguishable from a real low-entropy sample. Every
        other statistic here already answers NaN for all-NaN input.
        """
        image = np.full((1, 10, 10), np.nan)
        calc = self._make_calculator(image)
        assert np.isnan(calc._entropy()[0])
        assert np.isnan(calc._histogram()[0]).all()

    # --- Edge cases applied across all three ---

    def test_all_nan_image_returns_nan(self):
        """Skew and kurtosis should return NaN for all-NaN images."""
        image = np.full((3, 10, 10), np.nan)
        calc = self._make_calculator(image)
        assert all(np.isnan(v) for v in calc._skew())
        assert all(np.isnan(v) for v in calc._kurtosis())

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
    def test_different_dtypes(self, rng, dtype):
        """Results should be consistent across input dtypes."""
        if np.issubdtype(dtype, np.integer):
            image = rng.integers(0, 256, (3, 32, 32), dtype=dtype)
        else:
            image = rng.random((3, 32, 32)).astype(dtype)
        calc = self._make_calculator(image)
        skew_val = calc._skew()
        kurt_val = calc._kurtosis()
        ent_val = calc._entropy()
        assert all(np.isfinite(v) for v in skew_val)
        assert all(np.isfinite(v) for v in kurt_val)
        assert all(np.isfinite(v) for v in ent_val)

    @pytest.mark.parametrize("shape", [(1, 8, 8), (3, 64, 64), (1, 256, 256)])
    def test_various_image_sizes(self, rng, shape):
        """Equivalence should hold across different image sizes."""
        image = rng.random(shape, dtype=np.float64)
        calc = self._make_calculator(image)
        expected_skew = float(scipy_skew(calc.cache.scaled.ravel(), nan_policy="omit"))
        expected_kurt = float(scipy_kurtosis(calc.cache.scaled.ravel(), nan_policy="omit"))
        assert calc._skew()[0] == pytest.approx(expected_skew, rel=1e-5)
        assert calc._kurtosis()[0] == pytest.approx(expected_kurt, rel=1e-5)


class TestVisualStats:
    @pytest.mark.parametrize("n_channels", [1, 3])
    def test_process_visual_stats(self, n_channels):
        """Test visual statistics calculation."""
        images = [np.random.random((n_channels, 10, 10))]

        result = compute_stats(images, stats=ImageStats.VISUAL, per_channel=False)

        assert "brightness" in result["stats"]
        assert "contrast" in result["stats"]
        assert "darkness" in result["stats"]
        assert "sharpness" in result["stats"]
        assert "percentiles" in result["stats"]

        assert len(result["stats"]["brightness"]) == 1
        assert result["stats"]["brightness"].dtype == np.float32
        assert len(result["stats"]["percentiles"][0]) == 5  # QUARTILES length


class TestPixelStatsPerChannel:
    @pytest.mark.parametrize("n_channels", [1, 3])
    def test_process_per_channel(self, n_channels):
        """Test per-channel pixel statistics."""
        images = [np.random.random((n_channels, 10, 10))]

        result = compute_stats(images, stats=ImageStats.PIXEL, per_channel=True)

        assert len(result["stats"]["mean"]) == n_channels
        assert len(result["stats"]["std"]) == n_channels
        assert len(result["stats"]["histogram"]) == n_channels
        assert len(result["stats"]["histogram"][0]) == 256


class TestVisualStatsPerChannel:
    @pytest.mark.parametrize("n_channels", [1, 3])
    def test_process_visual_per_channel(self, n_channels):
        """Test per-channel visual statistics."""
        images = [np.random.random((n_channels, 10, 10))]

        result = compute_stats(images, stats=ImageStats.VISUAL, per_channel=True)

        assert len(result["stats"]["brightness"]) == n_channels
        assert len(result["stats"]["contrast"]) == n_channels
        assert len(result["stats"]["percentiles"]) == n_channels
        assert len(result["stats"]["percentiles"][0]) == 5


class TestDimensionStatsCalculator:
    def test_process_dimensions(self):
        """Test dimension statistics calculation."""
        image = np.random.rand(3, 100, 150)
        box = BoundingBox(10, 20, 60, 80, image_shape=image.shape)
        datum_calculator = CalculatorCache(image, box)
        calculator = DimensionStatCalculator(image, datum_calculator)

        stats = calculator.compute(ImageStats.DIMENSION)

        assert stats["offset_x"][0] == 10
        assert stats["offset_y"][0] == 20
        assert stats["width"][0] == 50
        assert stats["height"][0] == 60
        assert stats["channels"][0] == 3
        assert stats["size"][0] == 3000
        assert stats["aspect_ratio"][0] == pytest.approx(-1 + (50 / 60))
        assert len(stats["center"][0]) == 2

    def test_process_invalid_box(self):
        """Test dimension stats with invalid bounding box."""
        image = np.random.rand(3, 100, 100)
        box = BoundingBox(-1, -1, 0, 0, image_shape=image.shape)
        datum_calculator = CalculatorCache(image, box)
        calculator = DimensionStatCalculator(image, datum_calculator)
        stats = calculator.compute(ImageStats.DIMENSION)

        assert stats["invalid_box"][0] is True

    def test_aspect_ratio_square(self):
        """Test normalized aspect ratio for square images (should be 0)."""
        image = np.random.rand(3, 100, 100)
        box = BoundingBox(10, 10, 60, 60, image_shape=image.shape)  # 50x50 square
        datum_calculator = CalculatorCache(image, box)
        calculator = DimensionStatCalculator(image, datum_calculator)

        result = calculator._aspect_ratio()

        # Square should have normalized aspect ratio of 0
        assert result[0] == pytest.approx(0.0)

    def test_aspect_ratio_wide(self):
        """Test normalized aspect ratio for wide images (width > height)."""
        image = np.random.rand(3, 100, 200)
        box = BoundingBox(10, 20, 110, 70, image_shape=image.shape)  # width=100, height=50
        datum_calculator = CalculatorCache(image, box)
        calculator = DimensionStatCalculator(image, datum_calculator)

        result = calculator._aspect_ratio()

        # Wide image should be positive: 1 - (50/100) = 0.5
        assert result[0] == pytest.approx(0.5)
        assert result[0] > 0

    def test_aspect_ratio_tall(self):
        """Test normalized aspect ratio for tall images (height > width)."""
        image = np.random.rand(3, 200, 100)
        box = BoundingBox(10, 20, 60, 120, image_shape=image.shape)  # width=50, height=100
        datum_calculator = CalculatorCache(image, box)
        calculator = DimensionStatCalculator(image, datum_calculator)

        result = calculator._aspect_ratio()

        # Tall image should be negative: -1 * (1 - (50/100)) = -0.5
        assert result[0] == pytest.approx(-0.5)
        assert result[0] < 0

    def test_aspect_ratio_very_wide(self):
        """Test normalized aspect ratio for very wide images."""
        image = np.random.rand(3, 50, 400)
        box = BoundingBox(0, 0, 400, 50, image_shape=image.shape)  # width=400, height=50
        datum_calculator = CalculatorCache(image, box)
        calculator = DimensionStatCalculator(image, datum_calculator)

        result = calculator._aspect_ratio()

        # Very wide: 1 - (50/400) = 0.875
        assert result[0] == pytest.approx(0.875)
        assert result[0] > 0

    def test_aspect_ratio_very_tall(self):
        """Test normalized aspect ratio for very tall images."""
        image = np.random.rand(3, 400, 50)
        box = BoundingBox(0, 0, 50, 400, image_shape=image.shape)  # width=50, height=400
        datum_calculator = CalculatorCache(image, box)
        calculator = DimensionStatCalculator(image, datum_calculator)

        result = calculator._aspect_ratio()

        # Very tall: -1 * (1 - (50/400)) = -0.875
        assert result[0] == pytest.approx(-0.875)
        assert result[0] < 0

    def test_aspect_ratio_zero_width(self):
        """Test normalized aspect ratio with zero width (edge case)."""
        image = np.random.rand(3, 100, 100)
        box = BoundingBox(50, 10, 50, 60, image_shape=image.shape)  # width=0, height=50
        datum_calculator = CalculatorCache(image, box)
        calculator = DimensionStatCalculator(image, datum_calculator)

        result = calculator._aspect_ratio()

        # Zero width means infinitely tall: -1 * (1 - 0/50) = -1.0
        assert result[0] == pytest.approx(-1.0)
        assert result[0] < 0

    def test_aspect_ratio_zero_height(self):
        """Test normalized aspect ratio with zero height (edge case)."""
        image = np.random.rand(3, 100, 100)
        box = BoundingBox(10, 50, 60, 50, image_shape=image.shape)  # width=50, height=0
        datum_calculator = CalculatorCache(image, box)
        calculator = DimensionStatCalculator(image, datum_calculator)

        result = calculator._aspect_ratio()

        # Zero height means infinitely wide: 1 * (1 - 0/50) = 1.0
        assert result[0] == pytest.approx(1.0)
        assert result[0] > 0

    def test_aspect_ratio_both_zero(self):
        """Test normalized aspect ratio with both dimensions zero (edge case)."""
        image = np.random.rand(3, 100, 100)
        box = BoundingBox(50, 50, 50, 50, image_shape=image.shape)  # width=0, height=0
        datum_calculator = CalculatorCache(image, box)
        calculator = DimensionStatCalculator(image, datum_calculator)

        result = calculator._aspect_ratio()

        # Both zero should return NaN (division by zero)
        assert np.isnan(result[0])

    def test_aspect_ratio_non_spatial(self):
        """Test normalized aspect ratio for non-spatial (1D) data."""
        data = np.random.rand(100)
        datum_calculator = CalculatorCache(data, None)
        calculator = DimensionStatCalculator(data, datum_calculator)

        result = calculator._aspect_ratio()

        # Non-spatial data should return NaN
        assert np.isnan(result[0])


class TestHashStatsCalculator:
    @patch("dataeval.core._hash._xxhash")
    @patch("dataeval.core._hash._phash")
    def test_process_hashes(self, mock_phash, mock_xxhash):
        """Test hash statistics calculation."""
        mock_xxhash.return_value = ("xxhash_result", None)
        mock_phash.return_value = ("phash_result", None)

        image = np.random.rand(3, 50, 50)
        datum_calculator = CalculatorCache(image, None)
        calculator = HashStatCalculator(image, datum_calculator)

        stats = calculator.compute(ImageStats.HASH)

        assert stats["xxhash"][0] == "xxhash_result"
        assert stats["phash"][0] == "phash_result"


class TestPerImagePerBox:
    """Test per_image and per_target parameter combinations."""

    def test_per_image_only_no_boxes(self):
        """Test per_image=True with no boxes provided."""
        images = [np.random.random((3, 10, 10))]

        result = compute_stats(
            images,
            stats=ImageStats.PIXEL_MEAN,
            per_image=True,
            per_target=True,
            per_channel=False,
        )

        # Should have 1 result (full image)
        assert len(result["stats"]["mean"]) == 1
        assert len(result["source_index"]) == 1
        assert result["source_index"][0].item == 0
        assert result["source_index"][0].target is None
        assert result["source_index"][0].channel is None

    def test_per_image_and_per_target_with_boxes(self):
        """Test per_image=True and per_target=True with boxes provided."""
        images = [np.random.random((3, 100, 100))]
        boxes = [
            [
                BoundingBox(0, 0, 50, 50, image_shape=(3, 100, 100)),
                BoundingBox(50, 50, 100, 100, image_shape=(3, 100, 100)),
            ],
        ]

        result = compute_stats(
            images,
            boxes=boxes,
            stats=ImageStats.PIXEL_MEAN,
            per_image=True,
            per_target=True,
            per_channel=False,
        )

        # Should have 3 results: full image + 2 boxes
        assert len(result["stats"]["mean"]) == 3
        assert len(result["source_index"]) == 3

        # First should be full image
        assert result["source_index"][0].item == 0
        assert result["source_index"][0].target is None

        # Next two should be boxes
        assert result["source_index"][1].item == 0
        assert result["source_index"][1].target == 0
        assert result["source_index"][2].item == 0
        assert result["source_index"][2].target == 1

    def test_per_target_only_with_boxes(self):
        """Test per_image=False and per_target=True with boxes provided."""
        images = [np.random.random((3, 100, 100))]
        boxes = [
            [
                BoundingBox(0, 0, 50, 50, image_shape=(3, 100, 100)),
                BoundingBox(50, 50, 100, 100, image_shape=(3, 100, 100)),
            ],
        ]

        result = compute_stats(
            images,
            boxes=boxes,
            stats=ImageStats.PIXEL_MEAN,
            per_image=False,
            per_target=True,
            per_channel=False,
        )

        # Should have 2 results: only boxes
        assert len(result["stats"]["mean"]) == 2
        assert len(result["source_index"]) == 2

        # Both should be boxes (no full image)
        assert result["source_index"][0].item == 0
        assert result["source_index"][0].target == 0
        assert result["source_index"][1].item == 0
        assert result["source_index"][1].target == 1

    def test_per_image_only_with_boxes_ignored(self):
        """Test per_image=True and per_target=False with boxes provided (boxes ignored)."""
        images = [np.random.random((3, 100, 100))]
        boxes = [
            [
                BoundingBox(0, 0, 50, 50, image_shape=(3, 100, 100)),
                BoundingBox(50, 50, 100, 100, image_shape=(3, 100, 100)),
            ],
        ]

        result = compute_stats(
            images,
            boxes=boxes,
            stats=ImageStats.PIXEL_MEAN,
            per_image=True,
            per_target=False,
            per_channel=False,
        )

        # Should have 1 result: only full image (boxes ignored)
        assert len(result["stats"]["mean"]) == 1
        assert len(result["source_index"]) == 1

        # Should be full image
        assert result["source_index"][0].item == 0
        assert result["source_index"][0].target is None

    def test_per_image_and_per_target_with_per_channel(self):
        """Test per_image=True, per_target=True, and per_channel=True."""
        images = [np.random.random((3, 100, 100))]
        boxes = [[BoundingBox(0, 0, 50, 50, image_shape=(3, 100, 100))]]

        result = compute_stats(
            images,
            boxes=boxes,
            stats=ImageStats.PIXEL_MEAN,
            per_image=True,
            per_target=True,
            per_channel=True,
        )

        # Should have 6 results: (full image + 1 box) × 3 channels = 6
        assert len(result["stats"]["mean"]) == 6
        assert len(result["source_index"]) == 6

        # Check structure: full image channels first, then box channels
        # Full image - channel 0, 1, 2
        assert result["source_index"][0].item == 0
        assert result["source_index"][0].target is None
        assert result["source_index"][0].channel == 0

        assert result["source_index"][1].item == 0
        assert result["source_index"][1].target is None
        assert result["source_index"][1].channel == 1

        assert result["source_index"][2].item == 0
        assert result["source_index"][2].target is None
        assert result["source_index"][2].channel == 2

        # Box - channel 0, 1, 2
        assert result["source_index"][3].item == 0
        assert result["source_index"][3].target == 0
        assert result["source_index"][3].channel == 0

        assert result["source_index"][4].item == 0
        assert result["source_index"][4].target == 0
        assert result["source_index"][4].channel == 1

        assert result["source_index"][5].item == 0
        assert result["source_index"][5].target == 0
        assert result["source_index"][5].channel == 2

    def test_multiple_images_per_image_and_per_target(self):
        """Test multiple images with per_image=True and per_target=True."""
        images = [np.random.random((3, 100, 100)), np.random.random((3, 100, 100))]
        boxes = [
            [BoundingBox(0, 0, 50, 50, image_shape=(3, 100, 100))],
            [
                BoundingBox(25, 25, 75, 75, image_shape=(3, 100, 100)),
                BoundingBox(50, 50, 100, 100, image_shape=(3, 100, 100)),
            ],
        ]

        result = compute_stats(
            images,
            boxes=boxes,
            stats=ImageStats.PIXEL_MEAN,
            per_image=True,
            per_target=True,
            per_channel=False,
        )

        # Should have 5 results: image0 (1 full + 1 box) + image1 (1 full + 2 boxes)
        assert len(result["stats"]["mean"]) == 5
        assert len(result["source_index"]) == 5

        # Image 0: full image
        assert result["source_index"][0].item == 0
        assert result["source_index"][0].target is None

        # Image 0: box 0
        assert result["source_index"][1].item == 0
        assert result["source_index"][1].target == 0

        # Image 1: full image
        assert result["source_index"][2].item == 1
        assert result["source_index"][2].target is None

        # Image 1: box 0
        assert result["source_index"][3].item == 1
        assert result["source_index"][3].target == 0

        # Image 1: box 1
        assert result["source_index"][4].item == 1
        assert result["source_index"][4].target == 1

    def test_invalid_both_false_raises_error(self):
        """Test that per_image=False and per_target=False raises ValueError."""
        images = [np.random.random((3, 10, 10))]

        with pytest.raises(ValueError, match="At least one of 'per_image', 'per_target' or 'per_background'"):
            compute_stats(images, stats=ImageStats.PIXEL_MEAN, per_image=False, per_target=False, per_channel=False)

    def test_object_count_tracking(self):
        """Test that object_count is correctly tracked with per_image and per_target."""
        images = [np.random.random((3, 100, 100))]
        boxes = [
            [
                BoundingBox(0, 0, 50, 50, image_shape=(3, 100, 100)),
                BoundingBox(50, 50, 100, 100, image_shape=(3, 100, 100)),
            ],
        ]

        result = compute_stats(
            images,
            boxes=boxes,
            stats=ImageStats.PIXEL_MEAN,
            per_image=True,
            per_target=True,
            per_channel=False,
        )

        # Object count should be 2 (number of boxes)
        assert result["object_count"][0] == 2
        assert result["image_count"] == 1


class TestLowerDimensionalPixelStats:
    """Test pixel statistics with lower dimensional data (1D and 2D)."""

    def test_1d_data_pixel_stats(self):
        """Test pixel statistics calculation with 1D data."""
        # Create 1D data (shape: (length,))
        data = [np.random.random(100)]

        result = compute_stats(data, stats=ImageStats.PIXEL, per_channel=False)

        assert "mean" in result["stats"]
        assert "std" in result["stats"]
        assert "var" in result["stats"]
        assert "skew" in result["stats"]
        assert "kurtosis" in result["stats"]
        assert "entropy" in result["stats"]
        assert "missing" in result["stats"]
        assert "zeros" in result["stats"]
        assert "histogram" in result["stats"]

        assert len(result["stats"]["mean"]) == 1
        assert result["stats"]["mean"].dtype == np.float32
        assert len(result["stats"]["histogram"][0]) == 256

    def test_2d_data_pixel_stats(self):
        """Test pixel statistics calculation with 2D data (single channel image)."""
        # Create 2D data (shape: (height, width))
        data = [np.random.random((10, 10))]

        result = compute_stats(data, stats=ImageStats.PIXEL, per_channel=False)

        assert "mean" in result["stats"]
        assert "std" in result["stats"]
        assert "var" in result["stats"]
        assert "skew" in result["stats"]
        assert "kurtosis" in result["stats"]
        assert "entropy" in result["stats"]
        assert "missing" in result["stats"]
        assert "zeros" in result["stats"]
        assert "histogram" in result["stats"]

        assert len(result["stats"]["mean"]) == 1
        assert result["stats"]["mean"].dtype == np.float32
        assert len(result["stats"]["histogram"][0]) == 256

    def test_1d_data_with_nans(self):
        """Test pixel statistics with 1D data containing NaN values."""
        data = [np.array([np.nan, 0.5, 0.5, 0.5, np.nan])]

        result = compute_stats(data, stats=ImageStats.PIXEL, per_channel=False)
        assert result["stats"]["missing"][0] > 0

    def test_1d_data_per_channel(self):
        """Test that 1D data is treated as single channel when per_channel=True."""
        data = [np.random.random(100)]

        result = compute_stats(data, stats=ImageStats.PIXEL, per_channel=True)

        # Should be treated as 1 channel
        assert len(result["stats"]["mean"]) == 1
        assert len(result["stats"]["std"]) == 1
        assert len(result["stats"]["histogram"]) == 1

    def test_2d_data_per_channel(self):
        """Test that 2D data is treated as single channel when per_channel=True."""
        data = [np.random.random((10, 10))]

        result = compute_stats(data, stats=ImageStats.PIXEL, per_channel=True)

        # Should be treated as 1 channel
        assert len(result["stats"]["mean"]) == 1
        assert len(result["stats"]["std"]) == 1
        assert len(result["stats"]["histogram"]) == 1


class TestLowerDimensionalVisualStats:
    """Test visual statistics with lower dimensional data (1D and 2D)."""

    def test_1d_data_visual_stats(self):
        """Test visual statistics calculation with 1D data."""
        data = [np.random.random(100)]

        result = compute_stats(data, stats=ImageStats.VISUAL, per_channel=False)

        assert "brightness" in result["stats"]
        assert "contrast" in result["stats"]
        assert "darkness" in result["stats"]
        assert "sharpness" in result["stats"]
        assert "percentiles" in result["stats"]

        assert len(result["stats"]["brightness"]) == 1
        assert result["stats"]["brightness"].dtype == np.float32
        assert len(result["stats"]["percentiles"][0]) == 5  # QUARTILES length
        # Sharpness should be NaN for 1D data
        assert np.isnan(result["stats"]["sharpness"][0])

    def test_2d_data_visual_stats(self):
        """Test visual statistics calculation with 2D data."""
        data = [np.random.random((10, 10))]

        result = compute_stats(data, stats=ImageStats.VISUAL, per_channel=False)

        assert "brightness" in result["stats"]
        assert "contrast" in result["stats"]
        assert "darkness" in result["stats"]
        assert "sharpness" in result["stats"]
        assert "percentiles" in result["stats"]

        assert len(result["stats"]["brightness"]) == 1
        assert result["stats"]["brightness"].dtype == np.float32
        assert len(result["stats"]["percentiles"][0]) == 5
        # Sharpness should be computed for 2D data
        assert result["stats"]["sharpness"].dtype == np.float32
        assert not np.isnan(result["stats"]["sharpness"][0])

    def test_1d_data_visual_stats_per_channel(self):
        """Test visual statistics with 1D data and per_channel=True."""
        data = [np.random.random(100)]

        result = compute_stats(data, stats=ImageStats.VISUAL, per_channel=True)

        # Should be treated as 1 channel
        assert len(result["stats"]["brightness"]) == 1
        assert len(result["stats"]["contrast"]) == 1
        assert len(result["stats"]["sharpness"]) == 1
        # Sharpness should be NaN for 1D data
        assert np.isnan(result["stats"]["sharpness"][0])

    def test_2d_data_visual_stats_per_channel(self):
        """Test visual statistics with 2D data and per_channel=True."""
        data = [np.random.random((10, 10))]

        result = compute_stats(data, stats=ImageStats.VISUAL, per_channel=True)

        # Should be treated as 1 channel
        assert len(result["stats"]["brightness"]) == 1
        assert len(result["stats"]["contrast"]) == 1
        assert len(result["stats"]["sharpness"]) == 1
        assert len(result["stats"]["percentiles"]) == 1


class TestLowerDimensionalDimensionStats:
    """Test dimension statistics with lower dimensional data (1D and 2D)."""

    def test_1d_data_dimension_stats(self):
        """Test dimension statistics calculation with 1D data."""
        data = np.random.rand(100)
        datum_calculator = CalculatorCache(data, None)
        calculator = DimensionStatCalculator(data, datum_calculator)

        stats = calculator.compute(ImageStats.DIMENSION)

        # For 1D data: width is length, other spatial metrics are NaN
        assert stats["width"][0] == 100
        assert np.isnan(stats["height"][0])
        assert np.isnan(stats["offset_x"][0])
        assert np.isnan(stats["offset_y"][0])
        assert np.isnan(stats["aspect_ratio"][0])
        assert np.isnan(stats["center"][0][0])
        assert np.isnan(stats["center"][0][1])
        assert np.isnan(stats["distance_center"][0])
        assert np.isnan(stats["distance_edge"][0])
        assert stats["channels"][0] == 1
        assert stats["size"][0] == 100

    def test_2d_data_dimension_stats(self):
        """Test dimension statistics calculation with 2D data (single channel)."""
        data = np.random.rand(50, 100)
        box = BoundingBox(10, 20, 60, 80, image_shape=data.shape)
        datum_calculator = CalculatorCache(data, box)
        calculator = DimensionStatCalculator(data, datum_calculator)

        stats = calculator.compute(ImageStats.DIMENSION)

        # For 2D data: spatial metrics should work
        assert stats["offset_x"][0] == 10
        assert stats["offset_y"][0] == 20
        assert stats["width"][0] == 50
        assert stats["height"][0] == 60
        assert stats["channels"][0] == 1  # Single channel for 2D data
        assert stats["size"][0] == 3000
        assert stats["aspect_ratio"][0] == pytest.approx(-1 + (50 / 60))
        assert len(stats["center"][0]) == 2

    def test_1d_data_without_box(self):
        """Test dimension statistics with 1D data and no bounding box."""
        data = np.random.rand(50)
        datum_calculator = CalculatorCache(data, None)
        calculator = DimensionStatCalculator(data, datum_calculator)

        stats = calculator.compute(ImageStats.DIMENSION)

        assert stats["width"][0] == 50
        assert stats["channels"][0] == 1
        assert stats["size"][0] == 50
        # Spatial metrics should be NaN
        assert np.isnan(stats["height"][0])
        assert np.isnan(stats["aspect_ratio"][0])

    def test_2d_data_without_box(self):
        """Test dimension statistics with 2D data and no bounding box."""
        data = np.random.rand(30, 40)
        datum_calculator = CalculatorCache(data, None)
        calculator = DimensionStatCalculator(data, datum_calculator)

        stats = calculator.compute(ImageStats.DIMENSION)

        # Should use full image dimensions
        assert stats["width"][0] == 40
        assert stats["height"][0] == 30
        assert stats["channels"][0] == 1
        assert stats["size"][0] == 1200  # 30 * 40

    def test_calculate_1d_dimension_stats(self):
        """Test dimension statistics via compute_stats() with 1D data."""
        data = [np.random.random(100)]

        result = compute_stats(data, stats=ImageStats.DIMENSION, per_channel=False)

        assert "width" in result["stats"]
        assert "height" in result["stats"]
        assert "channels" in result["stats"]
        assert "size" in result["stats"]

        assert result["stats"]["width"][0] == 100
        assert np.isnan(result["stats"]["height"][0])
        assert result["stats"]["channels"][0] == 1
        assert result["stats"]["size"][0] == 100

    def test_calculate_2d_dimension_stats(self):
        """Test dimension statistics via compute_stats() with 2D data."""
        data = [np.random.random((10, 20))]

        result = compute_stats(data, stats=ImageStats.DIMENSION, per_channel=False)

        assert "width" in result["stats"]
        assert "height" in result["stats"]
        assert "channels" in result["stats"]
        assert "size" in result["stats"]

        assert result["stats"]["width"][0] == 20
        assert result["stats"]["height"][0] == 10
        assert result["stats"]["channels"][0] == 1
        assert result["stats"]["size"][0] == 200


class TestLowerDimensionalHashStats:
    """Test hash statistics with lower dimensional data (1D and 2D)."""

    @patch("dataeval.core._hash._xxhash")
    @patch("dataeval.core._hash._phash")
    def test_1d_data_hashes(self, mock_phash, mock_xxhash):
        """Test hash statistics calculation with 1D data."""
        mock_xxhash.return_value = ("xxhash_1d_result", None)
        mock_phash.return_value = ("phash_1d_result", None)

        data = np.random.rand(50)
        datum_calculator = CalculatorCache(data, None)
        calculator = HashStatCalculator(data, datum_calculator)

        stats = calculator.compute(ImageStats.HASH)

        assert stats["xxhash"][0] == "xxhash_1d_result"
        assert stats["phash"][0] == "phash_1d_result"

    @patch("dataeval.core._hash._xxhash")
    @patch("dataeval.core._hash._phash")
    def test_2d_data_hashes(self, mock_phash, mock_xxhash):
        """Test hash statistics calculation with 2D data."""
        mock_xxhash.return_value = ("xxhash_2d_result", None)
        mock_phash.return_value = ("phash_2d_result", None)

        data = np.random.rand(10, 10)
        datum_calculator = CalculatorCache(data, None)
        calculator = HashStatCalculator(data, datum_calculator)

        stats = calculator.compute(ImageStats.HASH)

        assert stats["xxhash"][0] == "xxhash_2d_result"
        assert stats["phash"][0] == "phash_2d_result"

    @patch("dataeval.core._hash._xxhash")
    @patch("dataeval.core._hash._phash")
    def test_calculate_1d_hash_stats(self, mock_phash, mock_xxhash):
        """Test hash statistics via compute_stats() with 1D data."""
        mock_xxhash.return_value = ("xxhash_calc_result", None)
        mock_phash.return_value = ("phash_calc_result", None)

        data = [np.random.random(100)]

        result = compute_stats(data, stats=ImageStats.HASH, per_channel=False)

        assert "xxhash" in result["stats"]
        assert "phash" in result["stats"]
        assert result["stats"]["xxhash"][0] == "xxhash_calc_result"
        assert result["stats"]["phash"][0] == "phash_calc_result"

    def test_1d_data_phash_warning(self):
        """Test that phash collects a warning for 1D data."""
        data = np.random.rand(50)
        datum_calculator = CalculatorCache(data, None)
        calculator = HashStatCalculator(data, datum_calculator)

        stats = calculator.compute(ImageStats.HASH)

        assert any("Perceptual hashing requires spatial data" in w for w in calculator.warnings)

        # phash should return empty string for 1D data
        assert stats["phash"][0] == ""
        # xxhash should still work
        assert stats["xxhash"][0] != ""

    def test_1d_data_phash_returns_empty_via_calculate(self):
        """Test that phash returns empty string for 1D data via compute_stats().

        Note: The warning is emitted but not captured due to multiprocessing.
        We test the behavior (empty string return) instead.
        """
        data = [np.random.random(100)]

        result = compute_stats(data, stats=ImageStats.HASH, per_channel=False)

        # phash should return empty string for 1D data
        assert result["stats"]["phash"][0] == ""
        # xxhash should still work
        assert result["stats"]["xxhash"][0] != ""

    def test_2d_small_image_phash_warning(self):
        """Test that phash collects a warning for images smaller than 9x9."""
        # Create a 5x5 image (smaller than required 9x9)
        data = np.random.rand(5, 5)
        datum_calculator = CalculatorCache(data, None)
        calculator = HashStatCalculator(data, datum_calculator)

        stats = calculator.compute(ImageStats.HASH)

        assert any("Image too small for perceptual hashing" in w for w in calculator.warnings)

        # phash should return empty string for small images
        assert stats["phash"][0] == ""
        # xxhash should still work
        assert stats["xxhash"][0] != ""


class TestImageClassificationDataset:
    """Test compute_stats() with ImageClassificationDataset input."""

    def test_ic_dataset_without_boxes(self, get_mock_ic_dataset):
        """Test ImageClassificationDataset processing without boxes."""
        images = [np.random.random((3, 100, 100)) for _ in range(3)]
        labels = [0, 1, 0]

        dataset = get_mock_ic_dataset(images, labels)

        result = compute_stats(dataset, stats=ImageStats.PIXEL_MEAN, per_image=True, per_target=True, per_channel=False)

        # Should process 3 images without boxes
        assert len(result["stats"]["mean"]) == 3
        assert len(result["source_index"]) == 3
        assert result["image_count"] == 3

        # All should be full images (box=None)
        for i in range(3):
            assert result["source_index"][i].item == i
            assert result["source_index"][i].target is None
            assert result["source_index"][i].channel is None

    def test_ic_dataset_with_explicit_boxes_param(self, get_mock_ic_dataset):
        """Test ImageClassificationDataset with explicit boxes parameter (should be ignored)."""
        images = [np.random.random((3, 100, 100)) for _ in range(2)]
        labels = [0, 1]

        dataset = get_mock_ic_dataset(images, labels)

        boxes = [
            [BoundingBox(0, 0, 50, 50, image_shape=(3, 100, 100))],
            [BoundingBox(25, 25, 75, 75, image_shape=(3, 100, 100))],
        ]

        result = compute_stats(
            dataset,
            boxes=boxes,
            stats=ImageStats.PIXEL_MEAN,
            per_image=True,
            per_target=True,
            per_channel=False,
        )

        # Should process boxes since they are explicitly provided
        assert len(result["stats"]["mean"]) == 4  # 2 images + 2 boxes
        assert len(result["source_index"]) == 4
        assert result["image_count"] == 2

    def test_ic_dataset_per_channel(self, get_mock_ic_dataset):
        """Test ImageClassificationDataset with per_channel=True."""
        images = [np.random.random((3, 50, 50)) for _ in range(2)]
        labels = [0, 1]

        dataset = get_mock_ic_dataset(images, labels)

        result = compute_stats(dataset, stats=ImageStats.PIXEL_MEAN, per_image=True, per_target=True, per_channel=True)

        # Should have 6 results: 2 images × 3 channels
        assert len(result["stats"]["mean"]) == 6
        assert len(result["source_index"]) == 6

        # Check channel ordering for first image
        assert result["source_index"][0].item == 0
        assert result["source_index"][0].target is None
        assert result["source_index"][0].channel == 0

        assert result["source_index"][1].item == 0
        assert result["source_index"][1].target is None
        assert result["source_index"][1].channel == 1

        assert result["source_index"][2].item == 0
        assert result["source_index"][2].target is None
        assert result["source_index"][2].channel == 2

    def test_ic_dataset_multiple_stats(self, get_mock_ic_dataset):
        """Test ImageClassificationDataset with multiple statistics."""
        images = [np.random.random((3, 100, 100)) for _ in range(2)]
        labels = [0, 1]

        dataset = get_mock_ic_dataset(images, labels)

        result = compute_stats(dataset, stats=ImageStats.PIXEL | ImageStats.VISUAL, per_image=True, per_channel=False)
        stats = result["stats"]

        # Check pixel stats
        assert "mean" in stats
        assert "std" in stats
        assert "var" in stats

        # Check visual stats
        assert "brightness" in stats
        assert "contrast" in stats
        assert "sharpness" in stats

        assert len(stats["mean"]) == 2
        assert result["image_count"] == 2


class TestObjectDetectionDataset:
    """Test compute_stats() with ObjectDetectionDataset input."""

    def test_od_dataset_with_boxes(self, get_mock_od_dataset):
        """Test ObjectDetectionDataset automatically processes boxes."""
        images = [np.random.random((3, 100, 100)) for _ in range(2)]
        labels = [[0, 1], [1]]
        bboxes = [
            [[10, 10, 50, 50], [60, 60, 90, 90]],
            [[20, 20, 70, 70]],
        ]

        dataset = get_mock_od_dataset(images, labels, bboxes)

        result = compute_stats(dataset, stats=ImageStats.PIXEL_MEAN, per_image=True, per_target=True, per_channel=False)

        # Should have: image0 (1 full + 2 boxes) + image1 (1 full + 1 box) = 5 results
        assert len(result["stats"]["mean"]) == 5
        assert len(result["source_index"]) == 5
        assert result["image_count"] == 2

        # Check object counts
        assert result["object_count"][0] == 2
        assert result["object_count"][1] == 1

        # Image 0: full image
        assert result["source_index"][0].item == 0
        assert result["source_index"][0].target is None

        # Image 0: box 0
        assert result["source_index"][1].item == 0
        assert result["source_index"][1].target == 0

        # Image 0: box 1
        assert result["source_index"][2].item == 0
        assert result["source_index"][2].target == 1

        # Image 1: full image
        assert result["source_index"][3].item == 1
        assert result["source_index"][3].target is None

        # Image 1: box 0
        assert result["source_index"][4].item == 1
        assert result["source_index"][4].target == 0

    def test_od_dataset_per_target_only(self, get_mock_od_dataset):
        """Test ObjectDetectionDataset with per_image=False, per_target=True."""
        images = [np.random.random((3, 100, 100)) for _ in range(2)]
        labels = [[0], [1, 0]]
        bboxes = [
            [[10, 10, 50, 50]],
            [[20, 20, 60, 60], [70, 70, 95, 95]],
        ]

        dataset = get_mock_od_dataset(images, labels, bboxes)

        result = compute_stats(
            dataset, stats=ImageStats.PIXEL_MEAN, per_image=False, per_target=True, per_channel=False
        )

        # Should have only boxes: 1 + 2 = 3 results (no full images)
        assert len(result["stats"]["mean"]) == 3
        assert len(result["source_index"]) == 3

        # All should be boxes (no full images)
        assert result["source_index"][0].item == 0
        assert result["source_index"][0].target == 0

        assert result["source_index"][1].item == 1
        assert result["source_index"][1].target == 0

        assert result["source_index"][2].item == 1
        assert result["source_index"][2].target == 1

    def test_od_dataset_per_image_only(self, get_mock_od_dataset):
        """Test ObjectDetectionDataset with per_image=True, per_target=False."""
        images = [np.random.random((3, 100, 100)) for _ in range(2)]
        labels = [[0, 1], [1]]
        bboxes = [
            [[10, 10, 50, 50], [60, 60, 90, 90]],
            [[20, 20, 70, 70]],
        ]

        dataset = get_mock_od_dataset(images, labels, bboxes)

        result = compute_stats(
            dataset, stats=ImageStats.PIXEL_MEAN, per_image=True, per_target=False, per_channel=False
        )

        # Should have only full images: 2 results (no boxes)
        assert len(result["stats"]["mean"]) == 2
        assert len(result["source_index"]) == 2

        # All should be full images
        assert result["source_index"][0].item == 0
        assert result["source_index"][0].target is None

        assert result["source_index"][1].item == 1
        assert result["source_index"][1].target is None

    def test_od_dataset_with_per_channel(self, get_mock_od_dataset):
        """Test ObjectDetectionDataset with per_channel=True."""
        images = [np.random.random((3, 100, 100))]
        labels = [[0]]
        bboxes = [[[10, 10, 50, 50]]]

        dataset = get_mock_od_dataset(images, labels, bboxes)

        result = compute_stats(dataset, stats=ImageStats.PIXEL_MEAN, per_image=True, per_target=True, per_channel=True)

        # Should have 6 results: (1 full image + 1 box) × 3 channels
        assert len(result["stats"]["mean"]) == 6
        assert len(result["source_index"]) == 6

        # Full image - channels 0, 1, 2
        assert result["source_index"][0].item == 0
        assert result["source_index"][0].target is None
        assert result["source_index"][0].channel == 0

        assert result["source_index"][1].item == 0
        assert result["source_index"][1].target is None
        assert result["source_index"][1].channel == 1

        assert result["source_index"][2].item == 0
        assert result["source_index"][2].target is None
        assert result["source_index"][2].channel == 2

        # Box - channels 0, 1, 2
        assert result["source_index"][3].item == 0
        assert result["source_index"][3].target == 0
        assert result["source_index"][3].channel == 0

    def test_od_dataset_with_dimension_stats(self, get_mock_od_dataset):
        """Test ObjectDetectionDataset with dimension statistics for boxes."""
        images = [np.random.random((3, 100, 100))]
        labels = [[0]]
        bboxes = [[[10, 20, 60, 80]]]  # x0=10, y0=20, x1=60, y1=80

        dataset = get_mock_od_dataset(images, labels, bboxes)

        result = compute_stats(dataset, stats=ImageStats.DIMENSION, per_image=False, per_target=True, per_channel=False)

        # Should have 1 result (just the box)
        assert len(result["source_index"]) == 1

        # Check box dimensions
        assert result["stats"]["offset_x"][0] == 10
        assert result["stats"]["offset_y"][0] == 20
        assert result["stats"]["width"][0] == 50
        assert result["stats"]["height"][0] == 60

    def test_od_dataset_override_with_boxes_param(self, get_mock_od_dataset):
        """Test ObjectDetectionDataset with boxes parameter override."""
        images = [np.random.random((3, 100, 100)) for _ in range(2)]
        labels = [[0], [1]]
        bboxes_dataset = [
            [[10, 10, 50, 50]],
            [[20, 20, 70, 70]],
        ]

        dataset = get_mock_od_dataset(images, labels, bboxes_dataset)

        boxes_override = [
            [BoundingBox(5, 5, 25, 25, image_shape=(3, 100, 100))],
            [BoundingBox(30, 30, 80, 80, image_shape=(3, 100, 100))],
        ]

        result = compute_stats(
            dataset,
            boxes=boxes_override,
            stats=ImageStats.DIMENSION,
            per_image=False,
            per_target=True,
            per_channel=False,
        )

        # Should use override boxes
        assert len(result["source_index"]) == 2

        # Check first box dimensions from override
        assert result["stats"]["offset_x"][0] == 5
        assert result["stats"]["offset_y"][0] == 5
        assert result["stats"]["width"][0] == 20
        assert result["stats"]["height"][0] == 20

    def test_od_dataset_empty_boxes(self, get_mock_od_dataset):
        """Test ObjectDetectionDataset with images that have no boxes."""
        images = [np.random.random((3, 100, 100)) for _ in range(2)]
        labels = [[], [0]]
        bboxes = [[], [[10, 10, 50, 50]]]

        dataset = get_mock_od_dataset(images, labels, bboxes)

        result = compute_stats(dataset, stats=ImageStats.PIXEL_MEAN, per_image=True, per_target=True, per_channel=False)

        # Should have: image0 (1 full + 0 boxes) + image1 (1 full + 1 box) = 3 results
        assert len(result["stats"]["mean"]) == 3
        assert len(result["source_index"]) == 3

        # Check object counts
        assert result["object_count"][0] == 0
        assert result["object_count"][1] == 1

    def test_od_dataset_multiple_stats(self, get_mock_od_dataset):
        """Test ObjectDetectionDataset with multiple statistics."""
        images = [np.random.random((3, 100, 100))]
        labels = [[0]]
        bboxes = [[[10, 10, 50, 50]]]

        dataset = get_mock_od_dataset(images, labels, bboxes)

        result = compute_stats(
            dataset,
            stats=ImageStats.PIXEL | ImageStats.VISUAL | ImageStats.DIMENSION,
            per_image=True,
            per_target=True,
            per_channel=False,
        )

        # Check pixel stats
        stats = result["stats"]
        assert "mean" in stats
        assert "std" in stats

        # Check visual stats
        assert "brightness" in stats
        assert "contrast" in stats

        # Check dimension stats
        assert "width" in stats
        assert "height" in stats
        assert "offset_x" in stats
        assert "offset_y" in stats

        # Should have 2 results: full image + 1 box


class TestProgressCallback:
    """Test suite for progress_callback functionality in calculate."""

    def test_progress_callback_called_during_calculate(self):
        """Test that progress_callback is called during calculation."""
        images = [np.random.random((3, 10, 10)) for _ in range(5)]
        callback_calls = []

        def callback(step: int, *, total: int | None = None, desc: str | None = None, extra_info: dict | None = None):
            callback_calls.append({"step": step, "total": total})

        result = compute_stats(images, stats=ImageStats.PIXEL, progress_callback=callback)

        # Callback should have been called for each image
        assert len(callback_calls) == 5
        assert result["image_count"] == 5

        # Verify callbacks have correct step values
        for i, call in enumerate(callback_calls):
            assert call["step"] == i + 1
            assert call["total"] == 5

    def test_progress_callback_not_called_when_none(self):
        """Test that no error occurs when progress_callback is None."""
        images = [np.random.random((3, 10, 10)) for _ in range(3)]

        result = compute_stats(images, stats=ImageStats.PIXEL, progress_callback=None)

        # Should work without error
        assert result["image_count"] == 3

    def test_progress_callback_with_boxes(self):
        """Test that progress_callback works with bounding boxes."""
        images = [np.random.random((3, 100, 100)) for _ in range(3)]
        boxes = [[[10, 10, 50, 50], [20, 20, 60, 60]] for _ in range(3)]
        callback_calls = []

        def callback(step: int, *, total: int | None = None, desc: str | None = None, extra_info: dict | None = None):
            callback_calls.append({"step": step, "total": total})

        result = compute_stats(images, boxes=boxes, stats=ImageStats.DIMENSION, progress_callback=callback)

        # Callback should be called for each image (not each box)
        assert len(callback_calls) == 3
        assert result["image_count"] == 3

        # Verify step counts
        for i, call in enumerate(callback_calls):
            assert call["step"] == i + 1
            assert call["total"] == 3

    def test_progress_callback_with_dataset(self, get_mock_od_dataset):
        """Test that progress_callback works with Dataset input."""
        images = [np.random.random((3, 100, 100)) for _ in range(4)]
        labels = [[0, 1] for _ in range(4)]
        bboxes = [[[10, 10, 50, 50], [20, 20, 60, 60]] for _ in range(4)]

        dataset = get_mock_od_dataset(images, labels, bboxes)
        callback_calls = []

        def callback(step: int, *, total: int | None = None, desc: str | None = None, extra_info: dict | None = None):
            callback_calls.append({"step": step, "total": total})

        result = compute_stats(dataset, stats=ImageStats.PIXEL, progress_callback=callback)

        # Callback should be called for each image
        assert len(callback_calls) == 4
        assert result["image_count"] == 4

        # Verify total is provided for Dataset (which is Sized)
        for call in callback_calls:
            assert call["total"] == 4

    def test_progress_callback_incremental_steps(self):
        """Test that progress_callback receives incremental step counts."""
        images = [np.random.random((3, 10, 10)) for _ in range(10)]
        callback_calls = []

        def callback(step: int, *, total: int | None = None, desc: str | None = None, extra_info: dict | None = None):
            callback_calls.append(step)

        compute_stats(images, stats=ImageStats.PIXEL_BASIC, progress_callback=callback)

        # Steps should be 1, 2, 3, ..., 10
        assert callback_calls == list(range(1, 11))

    def test_calculate_with_empty_dataset(self):
        """Test calculate with empty dataset."""
        images = []
        result = compute_stats(images, stats=ImageStats.PIXEL)

        assert result["image_count"] == 0
        assert len(result["source_index"]) == 0
        assert len(result["object_count"]) == 0
        assert len(result["invalid_box_count"]) == 0

    def test_calculate_determine_channel_indices_error(self):
        """Test _determine_channel_indices raises error for unexpected output (line 190)."""
        from dataeval.core._compute_stats import _determine_channel_indices

        # Create calculator output with unexpected number of elements
        # (not 1 for image-level, not equal to num_channels for per-channel)
        calculator_output = [{"stat1": [1, 2, 3]}]  # 3 values but image has 1 channel
        num_channels = 1

        with pytest.raises(ValueError, match="Processor produced"):
            _determine_channel_indices(calculator_output, num_channels)


@pytest.mark.required
class TestNormalizePixelValues:
    """Tests for the normalize_pixel_values parameter of compute_stats."""

    @pytest.fixture
    def uint8_images(self):
        rng = np.random.RandomState(42)
        return [rng.randint(0, 256, (3, 32, 32)).astype(np.uint8) for _ in range(5)]

    @pytest.fixture
    def mixed_bitdepth_images(self):
        """Two images: one 8-bit, one 16-bit, with similar visual content."""
        rng = np.random.RandomState(42)
        img_8bit = rng.randint(0, 256, (3, 32, 32)).astype(np.uint8)
        img_16bit = (img_8bit.astype(np.uint16)) * 257  # scale 0-255 to 0-65535
        return [img_8bit, img_16bit]

    def test_normalized_mean_is_between_0_and_1(self, uint8_images):
        """When normalized, pixel mean should be in [0, 1] range."""
        result = compute_stats(uint8_images, stats=ImageStats.PIXEL_MEAN, normalize_pixel_values=True)
        means = result["stats"]["mean"]
        assert np.all(means >= 0.0)
        assert np.all(means <= 1.0)

    def test_raw_mean_reflects_original_scale(self, uint8_images):
        """When not normalized, pixel mean should reflect original uint8 scale."""
        result = compute_stats(uint8_images, stats=ImageStats.PIXEL_MEAN, normalize_pixel_values=False)
        means = result["stats"]["mean"]
        # uint8 images with random data should have means around 127, not near 0-1
        assert np.all(means > 1.0), "Raw means for uint8 images should be well above 1.0"
        assert np.all(means < 256.0)

    def test_normalized_and_raw_are_proportional(self, uint8_images):
        """Normalized mean * 255 should approximate the raw mean for uint8 data."""
        norm = compute_stats(uint8_images, stats=ImageStats.PIXEL_MEAN, normalize_pixel_values=True)
        raw = compute_stats(uint8_images, stats=ImageStats.PIXEL_MEAN, normalize_pixel_values=False)
        np.testing.assert_allclose(norm["stats"]["mean"] * 255.0, raw["stats"]["mean"], rtol=1e-4)

    def test_normalization_makes_mixed_bitdepth_comparable(self, mixed_bitdepth_images):
        """Normalized stats of equivalent 8-bit and 16-bit images should be nearly identical."""
        result = compute_stats(mixed_bitdepth_images, stats=ImageStats.PIXEL_MEAN, normalize_pixel_values=True)
        means = result["stats"]["mean"]
        # Both images have the same visual content, so normalized means should match closely
        np.testing.assert_allclose(means[0], means[1], rtol=1e-3)

    def test_raw_stats_differ_across_bitdepths(self, mixed_bitdepth_images):
        """Without normalization, 8-bit and 16-bit images produce very different raw stats."""
        result = compute_stats(mixed_bitdepth_images, stats=ImageStats.PIXEL_MEAN, normalize_pixel_values=False)
        means = result["stats"]["mean"]
        # 16-bit mean should be ~257x larger than 8-bit mean
        assert means[1] / means[0] > 200.0

    def test_std_raw_vs_normalized(self, uint8_images):
        """Standard deviation should also scale with normalization."""
        norm = compute_stats(uint8_images, stats=ImageStats.PIXEL_STD, normalize_pixel_values=True)
        raw = compute_stats(uint8_images, stats=ImageStats.PIXEL_STD, normalize_pixel_values=False)
        np.testing.assert_allclose(norm["stats"]["std"] * 255.0, raw["stats"]["std"], rtol=1e-4)

    def test_histogram_bins_cover_correct_range(self, uint8_images):
        """Histogram should distribute values properly in both modes."""
        norm = compute_stats(uint8_images, stats=ImageStats.PIXEL_HISTOGRAM, normalize_pixel_values=True)
        raw = compute_stats(uint8_images, stats=ImageStats.PIXEL_HISTOGRAM, normalize_pixel_values=False)
        # Both histograms should have 256 bins and non-trivial spread
        for hist in [norm["stats"]["histogram"][0], raw["stats"]["histogram"][0]]:
            assert len(hist) == 256
            assert np.sum(hist > 0) > 10, "Histogram should use many bins, not collapse"

    def test_entropy_consistent_across_modes(self, uint8_images):
        """Entropy should be similar regardless of normalization for uniform-ish data."""
        norm = compute_stats(uint8_images, stats=ImageStats.PIXEL_ENTROPY, normalize_pixel_values=True)
        raw = compute_stats(uint8_images, stats=ImageStats.PIXEL_ENTROPY, normalize_pixel_values=False)
        # Entropy depends on bin distribution, not absolute scale, so should be close
        np.testing.assert_allclose(norm["stats"]["entropy"], raw["stats"]["entropy"], rtol=0.05)

    def test_default_emits_deprecation_warning(self, uint8_images):
        """Calling compute_stats without explicit normalize_pixel_values warns."""
        with pytest.warns(FutureWarning, match="normalize_pixel_values"):
            compute_stats(uint8_images, stats=ImageStats.PIXEL_MEAN)

    def test_explicit_value_no_warning(self, uint8_images):
        """Explicit normalize_pixel_values should not warn."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            compute_stats(uint8_images, stats=ImageStats.PIXEL_MEAN, normalize_pixel_values=True)
            compute_stats(uint8_images, stats=ImageStats.PIXEL_MEAN, normalize_pixel_values=False)


class TestCalculatorEmptyValuesDefault:
    """Cover the base Calculator.get_empty_values() default implementation."""

    def test_base_get_empty_values_returns_empty_dict(self):
        """The base get_empty_values() returns an empty dict (no custom empty values)."""
        image = np.random.rand(3, 10, 10)
        cache = CalculatorCache(image, box=None)
        calculator = DimensionStatCalculator(image, cache)

        # Invoke the base implementation directly (subclasses override it), so the
        # unbound base method's `return {}` executes.
        assert Calculator.get_empty_values(calculator) == {}


class TestVisualStatsAllNanPerChannel:
    """Cover the all-NaN per-channel branch of VisualStatCalculator.percentiles."""

    def test_percentiles_all_nan_per_channel(self):
        """All-NaN image in per-channel mode returns a (channels, len(QUARTILES)) NaN array."""
        image = np.full((3, 10, 10), np.nan)
        cache = CalculatorCache(image, box=None, per_channel=True)
        calculator = VisualStatCalculator(image, cache, per_channel=True)

        result = calculator.percentiles

        assert result.shape == (3, len(QUARTILES))
        assert np.isnan(result).all()


class TestPixelStatsAllNanPerChannel:
    """Cover the per-channel branch of PixelStatCalculator._nan_list()."""

    def test_nan_list_all_nan_per_channel(self):
        """All-NaN image in per-channel mode returns one NaN per channel."""
        image = np.full((3, 10, 10), np.nan)
        cache = CalculatorCache(image, box=None, per_channel=True)
        calculator = PixelStatCalculator(image, cache, per_channel=True)

        # _mean() short-circuits to _nan_list() when the cache is all-NaN.
        result = calculator._mean()

        assert len(result) == 3
        assert all(np.isnan(v) for v in result)


class TestDependencyRemoval:
    """Test that computed dependencies are not returned in the output."""

    def test_entropy_does_not_return_histogram(self):
        images = [np.random.random((10, 10))]
        result = compute_stats(images, stats=ImageStats.PIXEL_ENTROPY, normalize_pixel_values=False)
        assert "entropy" in result["stats"]
        assert "histogram" not in result["stats"]

    def test_brightness_does_not_return_percentiles(self):
        images = [np.random.random((10, 10))]
        result = compute_stats(images, stats=ImageStats.VISUAL_BRIGHTNESS, normalize_pixel_values=False)
        assert "brightness" in result["stats"]
        assert "percentiles" not in result["stats"]


class TestPerBackground:
    """Test per_background: statistics over the pixels an image's boxes do not cover."""

    @staticmethod
    def _image_with_bright_box(shape=(3, 20, 20), box=(0, 0, 10, 10)):
        """An image whose boxed region is brighter than everything around it."""
        rng = np.random.default_rng(0)
        image = (rng.random(shape) * 100).astype(np.uint8)
        x0, y0, x1, y1 = box
        image[:, y0:y1, x0:x1] = 250
        return image

    def test_background_stats_share_the_image_row(self):
        """Background values land on the item's existing row, under prefixed names."""
        images = [self._image_with_bright_box()]
        boxes = [[(0, 0, 10, 10)]]

        result = compute_stats(
            images,
            boxes=boxes,
            stats=ImageStats.PIXEL_MEAN,
            per_background=True,
            normalize_pixel_values=False,
        )

        # One row for the image, one for its box - the background added no row.
        assert len(result["source_index"]) == 2
        assert result["source_index"][0] == SourceIndex(0, None, None)
        assert result["source_index"][1] == SourceIndex(0, 0, None)
        assert set(result["stats"]) == {"mean", "background_mean", "background_fraction"}

    def test_background_excludes_the_boxed_pixels(self):
        """The background mean is taken over exactly the unboxed pixels."""
        images = [self._image_with_bright_box()]
        boxes = [[(0, 0, 10, 10)]]

        result = compute_stats(
            images,
            boxes=boxes,
            stats=ImageStats.PIXEL_MEAN,
            per_background=True,
            per_image=False,
            normalize_pixel_values=False,
        )

        mask = np.zeros((20, 20), dtype=bool)
        mask[0:10, 0:10] = True
        expected = images[0][:, ~mask].mean()

        assert result["stats"]["background_mean"][0] == pytest.approx(expected, abs=1e-3)
        # The bright box is gone, so the background sits well below the whole image.
        assert result["stats"]["background_mean"][0] < images[0].mean()

    def test_background_fraction_counts_overlap_once(self):
        """Overlapping boxes are unioned, so a twice-covered pixel is still one pixel."""
        images = [np.zeros((3, 10, 10), dtype=np.uint8)]
        boxes = [[(0, 0, 6, 6), (4, 4, 10, 10)]]

        result = compute_stats(
            images,
            boxes=boxes,
            stats=ImageStats.PIXEL_MEAN,
            per_background=True,
            normalize_pixel_values=False,
        )

        # 36 + 36 - 4 overlapping = 68 covered of 100.
        assert result["stats"]["background_fraction"][0] == pytest.approx(0.32)

    def test_background_fraction_is_one_without_boxes(self):
        """An image with no detections has a background equal to the whole image."""
        images = [self._image_with_bright_box(), self._image_with_bright_box()]
        boxes = [[(0, 0, 10, 10)], []]

        result = compute_stats(
            images,
            boxes=boxes,
            stats=ImageStats.PIXEL_MEAN,
            per_background=True,
            normalize_pixel_values=False,
        )

        image_rows = [i for i, s in enumerate(result["source_index"]) if s.target is None]
        empty = image_rows[1]
        assert result["stats"]["background_fraction"][empty] == pytest.approx(1.0)
        assert result["stats"]["background_mean"][empty] == pytest.approx(result["stats"]["mean"][empty], abs=1e-3)

    def test_fully_covered_image_yields_nan(self):
        """Boxes covering everything leave nothing to measure."""
        images = [self._image_with_bright_box()]
        boxes = [[(0, 0, 20, 20)]]

        result = compute_stats(
            images,
            boxes=boxes,
            stats=ImageStats.PIXEL | ImageStats.VISUAL,
            per_background=True,
            normalize_pixel_values=False,
        )

        assert result["stats"]["background_fraction"][0] == pytest.approx(0.0)
        for name in ("background_mean", "background_brightness", "background_missing", "background_zeros"):
            assert np.isnan(result["stats"][name][0])
        # The unmasked statistics on the same row are unaffected.
        assert not np.isnan(result["stats"]["mean"][0])

    def test_box_rows_carry_no_background_values(self):
        """Background is a property of the image, so box rows hold nulls for it."""
        images = [self._image_with_bright_box()]
        boxes = [[(0, 0, 10, 10), (12, 12, 16, 16)]]

        result = compute_stats(
            images,
            boxes=boxes,
            stats=ImageStats.PIXEL_MEAN,
            per_background=True,
            normalize_pixel_values=False,
        )

        # Every stat array stays one-to-one with the source index.
        assert all(len(values) == len(result["source_index"]) for values in result["stats"].values())
        for i, source in enumerate(result["source_index"]):
            if source.target is not None:
                assert np.isnan(result["stats"]["background_mean"][i])
                assert np.isnan(result["stats"]["background_fraction"][i])

    def test_missing_and_zeros_discount_the_mask(self):
        """The mask is not missing data, and is in neither half of a fraction."""
        images = [np.zeros((1, 10, 10), dtype=np.float64)]
        images[0][:, 0:5, :] = np.nan  # half the image genuinely missing
        boxes = [[(0, 0, 10, 5)]]  # ... and that same half is boxed out

        result = compute_stats(
            images,
            boxes=boxes,
            stats=ImageStats.PIXEL_MISSING | ImageStats.PIXEL_ZEROS,
            per_background=True,
            normalize_pixel_values=False,
        )

        # The whole image is half NaN and half zeros.
        assert result["stats"]["missing"][0] == pytest.approx(0.5)
        assert result["stats"]["zeros"][0] == pytest.approx(0.5)
        # The background is the other half: no missing data, and all of it zeros.
        assert result["stats"]["background_missing"][0] == pytest.approx(0.0)
        assert result["stats"]["background_zeros"][0] == pytest.approx(1.0)

    def test_per_channel_places_fraction_on_the_channel_none_row(self):
        """A single-valued statistic keeps its channel=None row in per-channel mode."""
        images = [self._image_with_bright_box()]
        boxes = [[(0, 0, 10, 10)]]

        result = compute_stats(
            images,
            boxes=boxes,
            stats=ImageStats.PIXEL_MEAN,
            per_background=True,
            per_target=False,
            per_channel=True,
            normalize_pixel_values=False,
        )

        by_channel = {s.channel: i for i, s in enumerate(result["source_index"])}
        assert set(by_channel) == {None, 0, 1, 2}
        assert result["stats"]["background_fraction"][by_channel[None]] == pytest.approx(0.75)
        for channel in (0, 1, 2):
            assert not np.isnan(result["stats"]["background_mean"][by_channel[channel]])

    def test_background_only_omits_the_unmasked_stats(self):
        """per_image=False with per_background=True yields the background row alone."""
        images = [self._image_with_bright_box()]
        boxes = [[(0, 0, 10, 10)]]

        result = compute_stats(
            images,
            boxes=boxes,
            stats=ImageStats.PIXEL_MEAN,
            per_image=False,
            per_target=False,
            per_background=True,
            normalize_pixel_values=False,
        )

        assert set(result["stats"]) == {"background_mean", "background_fraction"}
        assert len(result["source_index"]) == 1
        assert result["source_index"][0] == SourceIndex(0, None, None)

    def test_hash_and_dimension_stats_are_skipped_for_background(self):
        """Statistics that cannot describe a masked region are not computed for it."""
        images = [self._image_with_bright_box()]
        boxes = [[(0, 0, 10, 10)]]

        result = compute_stats(
            images,
            boxes=boxes,
            stats=ImageStats.HASH_XXHASH | ImageStats.DIMENSION_WIDTH | ImageStats.PIXEL_MEAN,
            per_background=True,
            normalize_pixel_values=False,
        )

        # Still computed for the image and its boxes...
        assert "xxhash" in result["stats"]
        assert "width" in result["stats"]
        # ... but they have no background counterpart.
        assert "background_xxhash" not in result["stats"]
        assert "background_width" not in result["stats"]
        assert "background_mean" in result["stats"]

    def test_no_applicable_stats_warns(self):
        """Asking only for statistics the background cannot carry is worth saying."""
        images = [self._image_with_bright_box()]
        boxes = [[(0, 0, 10, 10)]]

        with pytest.warns(UserWarning, match="none of the requested statistics apply to a background region"):
            result = compute_stats(
                images,
                boxes=boxes,
                stats=ImageStats.DIMENSION_WIDTH,
                per_background=True,
                normalize_pixel_values=False,
            )

        assert set(result["stats"]) == {"width", "background_fraction"}

    def test_two_dimensional_image_is_masked(self):
        """A (H, W) image has no channel axis to reach the crop path by, but is still masked.

        Regression test: the background row carries no box of its own, so a guard keyed
        on "has a box, or has channels" let single-channel images through unmasked and
        returned the whole image's statistics under the background's name.
        """
        image = np.zeros((20, 20), dtype=np.uint8)
        image[0:10, 0:10] = 255

        result = compute_stats(
            [image],
            boxes=[[(0, 0, 10, 10)]],
            stats=ImageStats.PIXEL_MEAN | ImageStats.PIXEL_ZEROS,
            per_background=True,
            normalize_pixel_values=False,
        )

        assert result["stats"]["mean"][0] == pytest.approx(63.75)
        # Everything bright was boxed out, so the background is the zeros that remain.
        assert result["stats"]["background_mean"][0] == pytest.approx(0.0)
        assert result["stats"]["background_zeros"][0] == pytest.approx(1.0)
        assert result["stats"]["background_fraction"][0] == pytest.approx(0.75)

    def test_an_unannotated_image_is_all_background(self):
        """Nothing annotated means nothing masked, so the background is the whole image.

        Already the answer an unannotated image inside an object-detection dataset gets;
        a dataset carrying no boxes at all is the same situation, one level up.
        """
        images = [np.random.random((3, 10, 10))]

        result = compute_stats(
            images,
            stats=ImageStats.PIXEL_MEAN,
            per_background=True,
            normalize_pixel_values=False,
        )["stats"]

        assert set(result) == {"mean", "background_mean", "background_fraction"}
        assert result["background_fraction"][0] == pytest.approx(1.0)
        assert result["background_mean"][0] == pytest.approx(result["mean"][0])

    def test_the_column_set_follows_the_arguments_not_the_data(self):
        """What makes two datasets combinable: identical arguments, identical columns.

        While `per_background` degraded on box-less data, a boxed and an unboxed dataset
        run with the same call produced different column sets, and combining them either
        dropped columns silently or — once that was made an error — refused outright.
        """
        images = np.random.default_rng(0).integers(0, 256, (4, 3, 8, 8), np.uint8)
        kwargs = {
            "stats": ImageStats.PIXEL_MEAN,
            "per_background": True,
            "per_target": False,
            "normalize_pixel_values": False,
        }

        boxed = compute_stats(images, boxes=[[(0, 0, 4, 4)]] * 4, **kwargs)
        unboxed = compute_stats(images, **kwargs)

        assert set(boxed["stats"]) == set(unboxed["stats"])
        stats, source_index, _ = combine_stats_results([boxed, unboxed])
        assert len(source_index) == 8
        assert set(stats) == {"mean", "background_mean", "background_fraction"}


class TestBitDepthAnchoring:
    """Every view of one datum is scaled and binned against the whole datum's range.

    A box's own extremes are not the image's, so letting each region infer its own bit
    depth scales them by different denominators and bins them over different ranges —
    which makes two regions' statistics differ when the pixels do not. Anchoring on the
    datum is what makes a per-target statistic comparable against its image's.
    """

    @staticmethod
    def _dark_box_image():
        """A 12-bit image whose top-left box holds only values an 8-bit image could."""
        image = np.full((1, 20, 20), 3000, dtype=np.uint16)
        image[:, 0:10, 0:10] = np.arange(100, dtype=np.uint16).reshape(10, 10)
        return image

    def test_normalized_box_uses_the_image_range(self):
        """A dark box is divided by the image's maximum, not its own."""
        image = self._dark_box_image()

        result = compute_stats(
            [image],
            boxes=[[(0, 0, 10, 10)]],
            stats=ImageStats.PIXEL_MEAN,
            per_image=False,
            normalize_pixel_values=True,
        )

        # The image is 12-bit (max 3000 < 4096), so every region scales by 4095.
        assert result["stats"]["mean"][0] == pytest.approx(np.arange(100).mean() / 4095, rel=1e-4)

    def test_box_entropy_is_binned_over_the_image_range(self):
        """The box's histogram spans the image's range, so it is not spread over 256 bins."""
        image = self._dark_box_image()

        result = compute_stats(
            [image],
            boxes=[[(0, 0, 10, 10)]],
            stats=ImageStats.PIXEL_ENTROPY,
            per_image=False,
            normalize_pixel_values=False,
        )

        # 100 distinct values inside [0, 4095] fall in far fewer than 100 of the 256 bins,
        # so the entropy is well below the log(100) an own-range binning would give.
        assert result["stats"]["entropy"][0] < np.log(100) - 0.5

    def test_image_and_box_agree_when_the_box_is_the_image(self):
        """A box covering the whole image must reproduce the image's own statistics."""
        image = self._dark_box_image()

        result = compute_stats(
            [image],
            boxes=[[(0, 0, 20, 20)]],
            stats=ImageStats.PIXEL_MEAN | ImageStats.PIXEL_ENTROPY,
            normalize_pixel_values=True,
        )

        for name in ("mean", "entropy"):
            assert result["stats"][name][0] == pytest.approx(result["stats"][name][1], rel=1e-6)

    def test_background_is_scaled_like_its_image(self):
        """Masking removes the image's brightest pixels without changing the scale."""
        image = self._dark_box_image()

        result = compute_stats(
            [image],
            # Box out the bright region, leaving only the dark corner as background.
            boxes=[[(0, 10, 20, 20)]],
            stats=ImageStats.PIXEL_MEAN,
            per_background=True,
            normalize_pixel_values=True,
        )
        boxed_out = compute_stats(
            [image],
            boxes=[[(0, 10, 20, 20)]],
            stats=ImageStats.PIXEL_MEAN,
            per_image=False,
            normalize_pixel_values=True,
        )

        # Background and box are disjoint halves of one image, both divided by 4095, so
        # their means average back to the whole image's.
        whole = result["stats"]["mean"][0]
        assert (result["stats"]["background_mean"][0] + boxed_out["stats"]["mean"][0]) / 2 == pytest.approx(
            whole, rel=1e-4
        )


class TestPerceptualVisualStatistics:
    """`PIXEL` reports the data; `VISUAL` reports the picture.

    A visual statistic stands in for how an image looks to a person, which is a position
    between black and white rather than a value in whatever units the sensor wrote. So it
    resolves a full-scale reference always — independently of `normalize_pixel_values`,
    which scopes to the pixel family — and reports NaN where no such reference exists.
    """

    _VISUAL = ImageStats.VISUAL & ~ImageStats.VISUAL_PERCENTILES

    @staticmethod
    def _encodings():
        """One picture, in the four spellings a caller might hand over."""
        u8 = np.random.default_rng(0).integers(0, 256, (3, 32, 32), dtype=np.uint8)
        return {
            "uint8": u8,
            "uint16": u8.astype(np.uint16) * 257,
            "float01": u8 / 255.0,
            "float255": u8.astype(np.float64),
        }

    @pytest.mark.parametrize("normalize", [True, False])
    def test_one_picture_reads_the_same_at_every_encoding(self, normalize):
        results = {
            name: compute_stats([image], stats=self._VISUAL, normalize_pixel_values=normalize)["stats"]
            for name, image in self._encodings().items()
        }

        reference = results["uint8"]
        for name, result in results.items():
            for stat in reference:
                assert result[stat][0] == pytest.approx(reference[stat][0], rel=1e-5), (
                    f"{stat} differs between uint8 and {name}"
                )

    def test_normalize_pixel_values_does_not_reach_them(self):
        """The flag scopes to the pixel family; a visual statistic already has its own scale."""
        image = self._encodings()["uint8"]

        normalized = compute_stats([image], stats=self._VISUAL, normalize_pixel_values=True)["stats"]
        raw = compute_stats([image], stats=self._VISUAL, normalize_pixel_values=False)["stats"]

        for stat in raw:
            assert normalized[stat][0] == pytest.approx(raw[stat][0], rel=1e-6)

    def test_eight_bit_input_takes_the_identity_path(self):
        """The display range *is* 8-bit's range, so the common case is not touched at all."""
        cache = CalculatorCache(self._encodings()["uint8"])

        assert cache.perceptual is cache.image, "an 8-bit image should not be copied to be read"

    def test_data_with_no_reference_has_no_reading(self):
        """A band carrying elevation has values, but 'how bright is it' has no answer."""
        elevation = np.random.default_rng(0).normal(0, 500, (1, 16, 16))

        result = compute_stats([elevation], stats=self._VISUAL | ImageStats.PIXEL_MEAN, normalize_pixel_values=False)[
            "stats"
        ]

        assert all(np.isnan(result[stat][0]) for stat in ("brightness", "contrast", "darkness", "sharpness"))
        assert not np.isnan(result["mean"][0]), "pixel statistics are unaffected by the absent reference"

    def test_a_declared_range_restores_the_reading(self):
        elevation = np.random.default_rng(0).normal(0, 500, (1, 16, 16))

        result = compute_stats(
            [elevation], stats=self._VISUAL, normalize_pixel_values=False, value_range=(-2000.0, 2000.0)
        )["stats"]

        assert all(np.isfinite(result[stat][0]) for stat in ("brightness", "contrast", "darkness", "sharpness"))
        # Values sat well inside the declared interval, so they land mid-range rather than
        # pinned to either end of it.
        assert 0.0 < result["brightness"][0] < 255.0

    def test_outlier_flags_do_not_move(self):
        """The 255x uniform scale-up is invisible to any threshold computed from the data."""
        rng = np.random.default_rng(0)
        images = [rng.integers(0, 256, (3, 32, 32), dtype=np.uint8) for _ in range(20)]
        images[7] = np.full((3, 32, 32), 250, dtype=np.uint8)  # a conspicuously bright one

        result = compute_stats(images, stats=ImageStats.VISUAL_BRIGHTNESS, normalize_pixel_values=True)["stats"]
        brightness = np.asarray(result["brightness"], dtype=np.float64)

        # Standardizing is what every non-constant threshold does first, and it is exactly
        # what a uniform rescale cancels out of.
        z = (brightness - brightness.mean()) / brightness.std()
        assert int(np.argmax(np.abs(z))) == 7
        assert np.abs(z[7]) > 3.0


class TestValueRangeAnchor:
    """A depth is decoded where the data was encoded, and withheld where it was measured.

    The two conventional float spellings of visible imagery worked by accident before the
    split — a ``[0, 1]`` image inferred ``depth=1, pmax=1``, so ``rescale`` divided by 1 and
    the histogram spanned ``(0, 1)``: right answers from wrong reasoning. Making the
    reasoning explicit must not move either number.
    """

    @staticmethod
    def _image():
        return np.random.default_rng(0).integers(0, 256, (3, 16, 16), dtype=np.uint8)

    _STATS = ImageStats.PIXEL_MEAN | ImageStats.PIXEL_ENTROPY | ImageStats.DIMENSION_DEPTH

    @pytest.mark.parametrize("normalize", [True, False])
    def test_normalized_float_is_unchanged(self, normalize):
        """`ToTensor`-style [0, 1] float: interval [0, 1], and the depth it always reported."""
        result = compute_stats([self._image() / 255.0], stats=self._STATS, normalize_pixel_values=normalize)["stats"]

        assert result["depth"][0] == 1
        # Already normalized, so normalizing again is the identity either way.
        assert result["mean"][0] == pytest.approx(self._image().mean() / 255.0, rel=1e-5)

    @pytest.mark.parametrize("normalize", [True, False])
    def test_float_boxed_8bit_matches_the_integer_it_holds(self, normalize):
        """An 8-bit image handed over in a float array reads as the 8-bit image it is."""
        boxed = compute_stats([self._image().astype(np.float64)], stats=self._STATS, normalize_pixel_values=normalize)[
            "stats"
        ]
        native = compute_stats([self._image()], stats=self._STATS, normalize_pixel_values=normalize)["stats"]

        for name in ("depth", "mean", "entropy"):
            assert boxed[name][0] == pytest.approx(native[name][0], rel=1e-6)

    def test_interpolated_8bit_float_still_reads_as_8bit(self):
        """A resize leaves 8-bit values non-integral; that does not stop them being 8-bit."""
        result = compute_stats(
            [self._image() * 0.7861], stats=ImageStats.DIMENSION_DEPTH, normalize_pixel_values=False
        )["stats"]

        assert result["depth"][0] == 8

    def test_undeclared_float_reports_no_depth(self):
        """A band whose range is the sensor's rather than a file format's has no depth to report."""
        elevation = np.random.default_rng(0).normal(0, 500, (1, 16, 16))

        result = compute_stats([elevation], stats=ImageStats.DIMENSION_DEPTH, normalize_pixel_values=False)["stats"]

        assert np.isnan(result["depth"][0]), "a fabricated depth is a wrong answer, not an imprecise one"

    @pytest.mark.parametrize(
        ("stats", "name", "normalize"),
        [
            (ImageStats.PIXEL_ENTROPY, "entropy", False),
            (ImageStats.PIXEL_HISTOGRAM, "histogram", False),
            (ImageStats.PIXEL_MEAN, "mean", True),
        ],
    )
    def test_statistics_needing_an_interval_refuse_to_guess(self, stats, name, normalize, caplog):
        """Previously such data was binned over [0, 1] with every value outside it.

        NaN rather than an error, matching every other unmeasurable view: the caller may
        have declared ranges for the band groups they care about and never asked for this
        one, and a single unmeasurable datum should not end a run over the rest. Said out
        loud, though, because an all-NaN column is easy to miss.
        """
        signed = np.random.default_rng(0).normal(0, 10, (1, 16, 16))

        with caplog.at_level(logging.WARNING, logger="dataeval.core._compute_stats"):
            result = compute_stats([signed], stats=stats, normalize_pixel_values=normalize)["stats"]

        assert np.all(np.isnan(result[name][0]))
        assert any("no value range could be established" in r.getMessage() for r in caplog.records)
        assert any("value_range=(low, high)" in r.getMessage() for r in caplog.records)

    def test_an_unmeasurable_datum_does_not_end_the_run(self):
        """The reason this is a warning: one bad image must not lose the other 99."""
        rng = np.random.default_rng(0)
        images = [rng.integers(0, 256, (1, 16, 16), np.uint8).astype(np.float64) for _ in range(3)]
        images[1] = rng.normal(0, 10, (1, 16, 16))

        result = compute_stats(images, stats=ImageStats.PIXEL_ENTROPY, normalize_pixel_values=False)["stats"]

        assert np.isfinite(result["entropy"][0])
        assert np.isnan(result["entropy"][1])
        assert np.isfinite(result["entropy"][2])

    def test_a_group_range_rescues_only_its_own_group(self):
        """The gap that made the raise untenable.

        A caller declaring `value_range` per group had their run ended by the always-on
        unnamed view, which they never asked for and cannot declare a range for when the
        bands genuinely disagree.
        """
        from dataeval.utils.preprocessing import ChannelGroup

        rng = np.random.default_rng(0)
        cube = np.stack([rng.normal(0, 500, (16, 16)), rng.normal(0, 50, (16, 16))])

        result = compute_stats(
            [cube],
            stats=ImageStats.PIXEL_MEAN | ImageStats.PIXEL_ENTROPY,
            normalize_pixel_values=False,
            channels={"tight": ChannelGroup(1, value_range=(-200.0, 200.0))},
        )["stats"]

        assert np.isfinite(result["tight_entropy"][0]), "the group declared an interval"
        assert np.isnan(result["entropy"][0]), "the unnamed view has none, and says so"
        assert np.isfinite(result["mean"][0]), "an unnormalized mean never needed one"

    def test_a_declared_range_is_what_they_are_measured_against(self):
        signed = np.random.default_rng(0).normal(0, 10, (1, 16, 16))

        result = compute_stats(
            [signed],
            stats=ImageStats.PIXEL_MEAN | ImageStats.PIXEL_ENTROPY | ImageStats.DIMENSION_DEPTH,
            normalize_pixel_values=True,
            value_range=(-50.0, 50.0),
        )["stats"]

        # Every value lands inside the declared interval, so normalizing puts them in [0, 1]
        # around the midpoint rather than pinning them all into one histogram bin.
        assert result["mean"][0] == pytest.approx((signed.mean() + 50.0) / 100.0, rel=1e-4)
        assert result["entropy"][0] > 0.0
        assert np.isnan(result["depth"][0]), "a declaration is a measurement, not an encoding"

    def test_a_malformed_declaration_fails_at_the_call(self):
        with pytest.raises(ValueError, match="value range must be"):
            compute_stats([self._image()], stats=ImageStats.PIXEL_MEAN, value_range=(1.0, 0.0))


class TestRescaleAnchor:
    """`rescale` accepts the range to scale from, which is what the cache passes it."""

    def test_defaults_to_the_arrays_own_range(self):
        # Dark enough to infer 1-bit, which is already the target depth, so unchanged.
        crop = np.array([[0, 1]], dtype=np.uint8)
        np.testing.assert_allclose(rescale(crop), [[0, 1]])

    def test_explicit_range_anchors_elsewhere(self):
        crop = np.array([[0, 1]], dtype=np.uint8)
        anchor = get_value_range(np.array([[0, 255]], dtype=np.uint8))
        np.testing.assert_allclose(rescale(crop, value_range=anchor), [[0.0, 1 / 255]])


class TestBackgroundEdgeCases:
    """Cases the background pass has to answer for rather than fall through."""

    def test_fully_covered_image_has_no_zero_entropy(self):
        """Regression: entropy and histogram once read an empty histogram as flat data."""
        image = (np.random.default_rng(0).random((3, 20, 20)) * 255).astype(np.uint8)

        result = compute_stats(
            [image],
            boxes=[[(0, 0, 20, 20)]],
            stats=ImageStats.PIXEL,
            per_background=True,
            normalize_pixel_values=False,
        )

        assert np.isnan(result["stats"]["background_entropy"][0])
        assert np.isnan(np.asarray(result["stats"]["background_histogram"][0], dtype=float)).all()

    def test_empty_datum_reports_no_fraction_and_stays_quiet(self):
        """A datum with no pixels has no background to report, and nothing to warn about."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = compute_stats(
                [np.zeros((3, 0, 0))],
                boxes=[[(0, 0, 1, 1)]],
                stats=ImageStats.PIXEL_MEAN,
                per_background=True,
                normalize_pixel_values=False,
            )

        assert np.isnan(result["stats"]["background_fraction"][0])

    def test_maskable_statistics_come_from_the_statistics(self):
        """Which statistics survive masking is each statistic's own declaration."""
        assert PixelStatCalculator.flags_for_view(ImageStats.PIXEL, ViewKind.MASK) == ImageStats.PIXEL
        assert VisualStatCalculator.flags_for_view(ImageStats.VISUAL, ViewKind.MASK) == ImageStats.VISUAL
        # Hashes are not stable under NaN; geometry is unchanged by a mask.
        assert not HashStatCalculator.flags_for_view(ImageStats.HASH, ViewKind.MASK)
        assert not DimensionStatCalculator.flags_for_view(ImageStats.DIMENSION, ViewKind.MASK)

    def test_one_calculator_can_disagree_with_itself(self):
        """The reason the declaration is per statistic rather than per class.

        Dropping bands does not narrow a bounding box, so ``rgb_width`` would restate the
        plain width under a new name. It does change which encoding is being read, so a
        group's depth is genuinely its own.
        """
        banded = DimensionStatCalculator.flags_for_view(ImageStats.DIMENSION, ViewKind.BAND)

        assert banded == ImageStats.DIMENSION_DEPTH

    def test_hashes_are_band_variant_but_not_maskable(self):
        """The pairing the old per-calculator predicate could not express."""
        assert HashStatCalculator.flags_for_view(ImageStats.HASH, ViewKind.BAND) == ImageStats.HASH
        assert not HashStatCalculator.flags_for_view(ImageStats.HASH, ViewKind.MASK)

    def test_a_new_statistic_defaults_to_the_whole_view_only(self):
        """A statistic is never computed over a derived view before it has been checked."""

        class Unchecked(Calculator[ImageStats]):
            def get_applicable_flags(self) -> ImageStats:
                return ImageStats.PIXEL

            def get_handlers(self):
                return {ImageStats.PIXEL_MEAN: Handler("mean", lambda: [0.0])}

        assert Unchecked.flags_for_view(ImageStats.PIXEL, ViewKind.WHOLE) == ImageStats.PIXEL_MEAN
        assert not Unchecked.flags_for_view(ImageStats.PIXEL, ViewKind.MASK)
        assert not Unchecked.flags_for_view(ImageStats.PIXEL, ViewKind.BAND)


@pytest.mark.required
class TestChannelGroups:
    """Named band groups are a *view* of an item, so they land as columns on its row.

    The band-axis counterpart of `per_background`, which does the same thing to the spatial
    axes. Rows would need a third source-index level to land on, and there is none — which
    is why per-channel statistics reach nothing downstream.
    """

    @staticmethod
    def _cube():
        """RGB in 0-255 and NIR in 30000-40000, stacked into one uint16 array.

        The case the design exists for, and the one that breaks a whole-datum range anchor:
        the visible bands are 8-bit data inside a 16-bit container.
        """
        rng = np.random.default_rng(0)
        cube = np.empty((4, 32, 32), np.uint16)
        cube[:3] = rng.integers(0, 256, (3, 32, 32))
        cube[3] = rng.integers(30000, 40000, (32, 32))
        return cube

    _GROUPS = {"rgb": [0, 1, 2], "nir": 3}

    def test_groups_land_as_prefixed_columns(self):
        result = compute_stats(
            [self._cube()],
            stats=ImageStats.PIXEL_MEAN,
            normalize_pixel_values=False,
            channels=self._GROUPS,
        )["stats"]

        assert set(result) == {"mean", "rgb_mean", "nir_mean"}
        assert len(result["mean"]) == len(result["rgb_mean"]) == 1

    def test_each_group_is_anchored_on_its_own_range(self):
        """Bands of one cube are different measurements, so they cannot share a denominator."""
        result = compute_stats(
            [self._cube()],
            stats=ImageStats.DIMENSION_DEPTH,
            normalize_pixel_values=False,
            channels=self._GROUPS,
        )["stats"]

        assert result["rgb_depth"][0] == 8, "the visible bands are 8-bit data in a 16-bit container"
        assert result["nir_depth"][0] == 16
        assert result["depth"][0] == 16, "the cube as a whole still reports its container"

    def test_a_group_is_reduced_over_jointly(self):
        """`rgb_mean` is one number over three bands, not three numbers."""
        cube = self._cube()

        result = compute_stats(
            [cube], stats=ImageStats.PIXEL_MEAN, normalize_pixel_values=False, channels={"rgb": [0, 1, 2]}
        )["stats"]

        assert result["rgb_mean"][0] == pytest.approx(cube[:3].mean(), rel=1e-4)

    def test_geometry_is_not_banded(self):
        """Dropping bands does not narrow a bounding box, so `rgb_width` would restate `width`."""
        result = compute_stats(
            [self._cube()],
            stats=ImageStats.DIMENSION_WIDTH | ImageStats.DIMENSION_CHANNELS,
            normalize_pixel_values=False,
            channels=self._GROUPS,
        )["stats"]

        assert set(result) == {"width", "channels"}

    def test_hashes_are_band_variant(self):
        """The fix for `Duplicates` on multispectral data.

        Hashing the whole cube runs a grayscale conversion that guesses between CMYK and
        RGBA at four channels; `rgb_xxhash` is the digest a caller actually wants.
        """
        result = compute_stats(
            [self._cube()], stats=ImageStats.HASH_XXHASH, normalize_pixel_values=False, channels=self._GROUPS
        )["stats"]

        assert len({result["xxhash"][0], result["rgb_xxhash"][0], result["nir_xxhash"][0]}) == 3

    def test_the_unnamed_view_survives_a_mapping(self):
        """Adding channels= to a pipeline reading `brightness` must not remove `brightness`."""
        without = compute_stats([self._cube()], stats=ImageStats.PIXEL_MEAN, normalize_pixel_values=False)["stats"]
        with_groups = compute_stats(
            [self._cube()], stats=ImageStats.PIXEL_MEAN, normalize_pixel_values=False, channels=self._GROUPS
        )["stats"]

        assert with_groups["mean"][0] == without["mean"][0]

    def test_a_group_range_overrides_the_call_level_one(self):
        """Bands of one cube are different measurements, so a group may need its own interval."""
        from dataeval.utils.preprocessing import ChannelGroup

        rng = np.random.default_rng(0)
        # Band 0 spans roughly [-1500, 1500]; band 1 is an order of magnitude tighter.
        cube = np.stack([rng.normal(0, 500, (16, 16)), rng.normal(0, 50, (16, 16))])

        result = compute_stats(
            [cube],
            stats=ImageStats.PIXEL_ENTROPY,
            normalize_pixel_values=False,
            value_range=(-2000.0, 2000.0),
            channels={"tight": ChannelGroup(1, value_range=(-200.0, 200.0))},
        )["stats"]

        # Binned over its own interval the tight band fills the histogram; over the cube's
        # it would collapse into the middle few bins and read as near-zero entropy.
        assert result["tight_entropy"][0] > result["entropy"][0]

    def test_a_group_inherits_the_call_level_range_when_it_declares_none(self):
        rng = np.random.default_rng(0)
        cube = np.stack([rng.normal(0, 500, (16, 16))] * 2)

        result = compute_stats(
            [cube],
            stats=ImageStats.PIXEL_ENTROPY,
            normalize_pixel_values=False,
            value_range=(-2000.0, 2000.0),
            channels={"first": 0},
        )["stats"]

        assert np.isfinite(result["first_entropy"][0])


@pytest.mark.required
class TestUnsatisfiableChannelGroups:
    """A group the datum cannot supply is NaN'd whole, and its column still exists.

    All-or-nothing: a group spanning bands 2-5 must not mean "bands 2-3" on a 4-band image
    and "bands 2-5" on an 8-band one under one column name. And it must be *substituted*
    rather than skipped, or the column vanishes for that datum and every later array
    misaligns against the source index.
    """

    @staticmethod
    def _ragged():
        """One 4-band image and one 2-band image, in that order."""
        rng = np.random.default_rng(0)
        return [rng.integers(0, 256, (4, 8, 8), np.uint8), rng.integers(0, 256, (2, 8, 8), np.uint8)]

    def test_a_missing_group_is_nan_not_a_missing_column(self):
        result = compute_stats(
            self._ragged(), stats=ImageStats.PIXEL_MEAN, normalize_pixel_values=False, channels={"nir": 3}
        )

        assert "nir_mean" in result["stats"]
        assert np.isfinite(result["stats"]["nir_mean"][0])
        assert np.isnan(result["stats"]["nir_mean"][1]), "the 2-band image has no band 3"

    def test_every_column_stays_aligned_with_the_source_index(self):
        """The failure mode substitution exists to prevent: a short array, silently offset."""
        result = compute_stats(
            self._ragged(),
            stats=ImageStats.PIXEL_MEAN | ImageStats.VISUAL_BRIGHTNESS,
            normalize_pixel_values=False,
            channels={"nir": 3, "rgb": [0, 1, 2]},
        )

        rows = len(result["source_index"])
        for name, values in result["stats"].items():
            assert len(values) == rows, f"{name} is {len(values)} long against {rows} rows"

    def test_partial_coverage_is_all_or_nothing(self):
        """Not reduced over the bands that happen to be present."""
        result = compute_stats(
            self._ragged(),
            stats=ImageStats.PIXEL_MEAN,
            normalize_pixel_values=False,
            channels={"wide": [0, 1, 2, 3]},
        )["stats"]

        assert np.isfinite(result["wide_mean"][0])
        assert np.isnan(result["wide_mean"][1]), "bands 0-1 are present but the group named four"

    def test_an_unsatisfiable_group_has_no_hash(self):
        """Hashing NaN yields the same digest every time, making every absence a duplicate."""
        result = compute_stats(
            self._ragged(), stats=ImageStats.HASH_XXHASH, normalize_pixel_values=False, channels={"nir": 3}
        )["stats"]

        assert result["nir_xxhash"][0] != ""
        assert result["nir_xxhash"][1] == ""

    def test_a_vector_valued_statistic_keeps_its_shape(self):
        """`histogram` must come back as 256 NaN bins, not a scalar NaN.

        A padding path would look up the unprefixed `histogram` for a column named
        `nir_histogram`, miss, and substitute a scalar — producing a ragged object array.
        Running the real calculators over NaN pixels sidesteps the whole class of problem.
        """
        result = compute_stats(
            self._ragged(), stats=ImageStats.PIXEL_HISTOGRAM, normalize_pixel_values=False, channels={"nir": 3}
        )["stats"]

        assert result["nir_histogram"].shape == (2, 256)
        assert np.isnan(result["nir_histogram"][1]).all()


@pytest.mark.required
class TestChannelGroupComposition:
    """Region first, then band: `background_nir_brightness` is a real quantity."""

    def test_groups_compose_with_the_background(self):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, (4, 16, 16), np.uint8)

        result = compute_stats(
            [image],
            boxes=[[(0, 0, 8, 8)]],
            stats=ImageStats.PIXEL_MEAN,
            per_background=True,
            normalize_pixel_values=False,
            channels={"nir": 3},
        )["stats"]

        assert "background_nir_mean" in result
        assert np.isfinite(result["background_nir_mean"][0])

    def test_the_background_keeps_its_own_unbanded_column(self):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, (4, 16, 16), np.uint8)

        result = compute_stats(
            [image],
            boxes=[[(0, 0, 8, 8)]],
            stats=ImageStats.PIXEL_MEAN,
            per_background=True,
            normalize_pixel_values=False,
            channels={"nir": 3},
        )["stats"]

        assert {"mean", "nir_mean", "background_mean", "background_nir_mean"} <= set(result)

    def test_a_masked_region_still_has_no_hash(self):
        """MASK and BAND intersect: a hash answers for a band group but not for a mask."""
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, (4, 16, 16), np.uint8)

        result = compute_stats(
            [image],
            boxes=[[(0, 0, 8, 8)]],
            stats=ImageStats.HASH_XXHASH,
            per_background=True,
            normalize_pixel_values=False,
            channels={"nir": 3},
        )["stats"]

        assert "nir_xxhash" in result
        assert not any(name.startswith("background_") for name in result if name != "background_fraction")


@pytest.mark.required
class TestChannelGroupValidation:
    """Group names are checked at the call, where the mistake is.

    Every input to the check is known before an image is read, so a bad name fails here
    rather than as a confusing rename several layers downstream.
    """

    _IMAGE = [np.zeros((4, 8, 8), np.uint8)]

    @pytest.mark.parametrize("name", ["unit", "instance", "sequence", "track", "background"])
    def test_reserved_names_are_rejected(self, name):
        """`instance_brightness` would be indistinguishable from a level-qualified column."""
        with pytest.raises(ValueError, match="reserved"):
            compute_stats(self._IMAGE, stats=ImageStats.PIXEL_MEAN, channels={name: 0}, normalize_pixel_values=False)

    def test_a_name_that_would_collide_with_a_statistic_is_rejected(self):
        """`distance` + `center` is `distance_center`, which already names a statistic."""
        with pytest.raises(ValueError, match="already name statistics"):
            compute_stats(
                self._IMAGE,
                stats=ImageStats.DIMENSION_CENTER | ImageStats.DIMENSION_DISTANCE_CENTER,
                channels={"distance": 0},
                normalize_pixel_values=False,
            )

    @pytest.mark.parametrize("name", ["", "not an identifier", "3bands"])
    def test_names_must_be_identifiers(self, name):
        with pytest.raises(ValueError, match="valid identifiers"):
            compute_stats(self._IMAGE, stats=ImageStats.PIXEL_MEAN, channels={name: 0}, normalize_pixel_values=False)

    def test_a_mapping_with_no_band_variant_statistic_warns(self):
        """Naming bands and asking only for geometry returns no band columns at all."""
        with pytest.warns(UserWarning, match="none of the requested statistics vary"):
            compute_stats(
                self._IMAGE,
                stats=ImageStats.DIMENSION_WIDTH,
                channels={"nir": 3},
                normalize_pixel_values=False,
            )


@pytest.mark.required
class TestPerChannelDeprecation:
    """The row shape is kept and deprecated rather than rebuilt badly.

    Column names discovered per datum cannot be reconciled across ragged data — the
    aggregation appends by name, so a name absent from one datum shortens its array. The
    row path has no such problem, so it survives until the shape goes.
    """

    _IMAGE = [np.zeros((3, 8, 8), np.uint8)]

    def test_channels_true_is_the_row_path(self):
        with pytest.warns(DeprecationWarning, match="Per-channel rows are deprecated"):
            result = compute_stats(
                self._IMAGE, stats=ImageStats.PIXEL_MEAN, channels=True, normalize_pixel_values=False
            )

        assert any(s.channel is not None for s in result["source_index"])

    def test_per_channel_warns_and_is_unchanged(self):
        with pytest.warns(DeprecationWarning, match="Per-channel rows are deprecated"):
            old = compute_stats(
                self._IMAGE, stats=ImageStats.PIXEL_MEAN, per_channel=True, normalize_pixel_values=False
            )
        with pytest.warns(DeprecationWarning, match="Per-channel rows are deprecated"):
            shim = compute_stats(self._IMAGE, stats=ImageStats.PIXEL_MEAN, channels=True, normalize_pixel_values=False)

        assert [s.channel for s in old["source_index"]] == [s.channel for s in shim["source_index"]]

    def test_the_message_carries_the_migration(self):
        with pytest.warns(DeprecationWarning, match=r"channels=\{'r': 0, 'g': 1, 'b': 2\}"):
            compute_stats(self._IMAGE, stats=ImageStats.PIXEL_MEAN, per_channel=True, normalize_pixel_values=False)


@pytest.mark.required
class TestWideBandWarning:
    """The unnamed view means *the image as a picture*, only defined for mono or RGB.

    Said rather than enforced: a cap would take the dimension statistics — well defined at
    any band count — down with it, and existing 4-band callers keep today's answer.
    """

    @staticmethod
    def _warned(caplog):
        return any("measured as a single picture" in record.getMessage() for record in caplog.records)

    def test_visual_statistics_on_a_wide_datum_warn(self, caplog):
        image = np.zeros((6, 8, 8), np.uint8)

        with caplog.at_level(logging.WARNING, logger="dataeval.core._compute_stats"):
            compute_stats([image], stats=ImageStats.VISUAL_BRIGHTNESS, normalize_pixel_values=False)

        assert self._warned(caplog)

    def test_naming_the_bands_answers_the_question(self, caplog):
        """A caller who named their groups has already said which bands are a picture."""
        image = np.zeros((6, 8, 8), np.uint8)

        with caplog.at_level(logging.WARNING, logger="dataeval.core._compute_stats"):
            compute_stats(
                [image],
                stats=ImageStats.VISUAL_BRIGHTNESS,
                normalize_pixel_values=False,
                channels={"rgb": [0, 1, 2]},
            )

        assert not self._warned(caplog)

    def test_three_band_data_never_warns(self, caplog):
        image = np.zeros((3, 8, 8), np.uint8)

        with caplog.at_level(logging.WARNING, logger="dataeval.core._compute_stats"):
            compute_stats([image], stats=ImageStats.VISUAL | ImageStats.HASH, normalize_pixel_values=False)

        assert not self._warned(caplog)

    def test_dimension_statistics_are_never_capped(self, caplog):
        """Warn, do not cap — a cap would take geometry down with the visual statistics."""
        image = np.zeros((6, 8, 8), np.uint8)

        with caplog.at_level(logging.WARNING, logger="dataeval.core._compute_stats"):
            result = compute_stats([image], stats=ImageStats.DIMENSION_WIDTH, normalize_pixel_values=False)

        assert result["stats"]["width"][0] == 8.0
        assert not self._warned(caplog)


@pytest.mark.required
class TestCommonCaseUnaffected:
    """1- and 3-channel callers must notice nothing: no argument, no warning, no change.

    This feature is for power users with special image formats. Every element of it is
    measured against this.
    """

    @pytest.mark.parametrize("channels", [1, 3])
    def test_results_are_identical_to_before_the_feature(self, channels, recwarn):
        image = np.random.default_rng(0).integers(0, 256, (channels, 16, 16), dtype=np.uint8)

        result = compute_stats(
            [image],
            stats=ImageStats.PIXEL | ImageStats.VISUAL | ImageStats.DIMENSION,
            normalize_pixel_values=False,
        )

        assert not any(name.count("_") and name.split("_")[0] in ("rgb", "nir") for name in result["stats"])
        assert not result.get("warnings")
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning | UserWarning)]


@pytest.mark.required
class TestUnmeasuredViewsAnswerConsistently:
    """One rule for a view that holds no data: NaN, except for the statistic about absence.

    Every statistic here is a claim about values, and there are no values to make it about
    — so NaN. `missing` is the exception by construction: it measures the *presence* of
    data rather than the data, so it is precisely the one that still has an answer, and
    1.0 is the signal a reader needs. Hashes answer with the empty string for the same
    reason they cannot answer at all — see `TestUnsatisfiableChannelGroups`.
    """

    _STATS = (
        ImageStats.PIXEL_MEAN
        | ImageStats.PIXEL_STD
        | ImageStats.PIXEL_ENTROPY
        | ImageStats.PIXEL_ZEROS
        | ImageStats.PIXEL_MISSING
        | ImageStats.VISUAL_BRIGHTNESS
    )

    def test_an_absent_band_group(self):
        ragged = [np.zeros((4, 8, 8), np.uint8), np.zeros((2, 8, 8), np.uint8)]

        result = compute_stats(ragged, stats=self._STATS, normalize_pixel_values=False, channels={"nir": 3})["stats"]

        for name in ("nir_mean", "nir_std", "nir_entropy", "nir_zeros", "nir_brightness"):
            assert np.isnan(result[name][1]), f"{name} claims something about values that are not there"
        assert result["nir_missing"][1] == pytest.approx(1.0), "absence is what `missing` is for"

    def test_an_out_of_bounds_box(self):
        """The same rule, reached by a different route — and `zeros` was wrong here too."""
        images = [np.zeros((3, 8, 8), np.uint8)]

        result = compute_stats(
            images,
            boxes=[[(100, 100, 110, 110)]],
            stats=self._STATS,
            per_image=False,
            normalize_pixel_values=False,
        )["stats"]

        assert np.isnan(result["zeros"][0]), "an all-zero image would report 1.0; a box off it reported 0.0"
        assert result["missing"][0] == pytest.approx(1.0)
        for name in ("mean", "std", "entropy", "brightness"):
            assert np.isnan(result[name][0])

    def test_a_measured_view_still_reports_zeros(self):
        """The guard must not swallow a real all-zero measurement."""
        result = compute_stats(
            [np.zeros((3, 8, 8), np.uint8)], stats=ImageStats.PIXEL_ZEROS, normalize_pixel_values=False
        )["stats"]

        assert result["zeros"][0] == pytest.approx(1.0)


@pytest.mark.required
class TestPresenceStatisticsReadTheRawView:
    """`missing` and `zeros` are claims about the stored values, not about the scaled copy.

    Both are counted over `CalculatorCache.image` in the aggregate path and so must be
    counted over the same view per channel. `scaled` cannot answer either: it is all-NaN
    where no interval could be established, and it shifts by `pmin`, which moves a raw zero
    off zero for any range that does not start there.
    """

    def test_missing_is_zero_for_present_but_unmeasurable_data(self):
        """Data with no readable range is still present, so nothing is missing."""
        images = np.random.default_rng(0).random((2, 3, 8, 8)) * 1000.0  # float beyond [0, 255]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            per_channel = compute_stats(
                images, stats=ImageStats.PIXEL_MISSING, normalize_pixel_values=True, per_channel=True
            )["stats"]
            aggregate = compute_stats(
                images, stats=ImageStats.PIXEL_MISSING, normalize_pixel_values=True, per_channel=False
            )["stats"]

        assert np.all(np.asarray(per_channel["missing"]) == 0.0), "an unreadable range is not missing data"
        assert np.all(np.asarray(aggregate["missing"]) == 0.0)

    def test_zeros_agrees_across_layouts_under_a_declared_range(self):
        """A declared range starting below zero must not move which pixels count as zero."""
        images = np.zeros((1, 3, 4, 4))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            per_channel = compute_stats(
                images,
                stats=ImageStats.PIXEL_ZEROS,
                normalize_pixel_values=True,
                value_range=(-100.0, 100.0),
                per_channel=True,
            )["stats"]
            aggregate = compute_stats(
                images,
                stats=ImageStats.PIXEL_ZEROS,
                normalize_pixel_values=True,
                value_range=(-100.0, 100.0),
                per_channel=False,
            )["stats"]

        assert np.all(np.asarray(per_channel["zeros"]) == pytest.approx(1.0))
        assert np.all(np.asarray(aggregate["zeros"]) == pytest.approx(1.0))


class TestRegressionsAgainstV1_0:
    """Calls that worked before this branch and must keep working.

    Each of these passed on v1.0.6 and on `main`, broke somewhere in the channel-views
    work, and was found only after the fact — the full suite stayed green through all
    three. Pinned here so the next refactor of the view machinery has to answer for them.
    """

    def test_per_background_on_non_spatial_data(self):
        """`per_background=True` over 1-D data must not reach the mask builder.

        Phase 1 dropped `per_background = per_background and has_boxes` deliberately, so
        an unannotated image would still report a background. That reasoning covers
        box-less *image* data; 1-D data has no image plane to carve a background out of,
        and `boxes_to_mask` reads `shape[-2:]` unconditionally.
        """
        result = compute_stats(
            [np.random.default_rng(0).random(32)],
            stats=ImageStats.PIXEL_MEAN,
            per_background=True,
            normalize_pixel_values=False,
        )

        assert "mean" in result["stats"]
        assert not np.isnan(result["stats"]["mean"]).all()

    def test_two_dimensional_data_reports_one_channel(self):
        """A NaN fallback must be sized by the channel count, not by `image.shape[0]`.

        For 2-D data `shape[0]` is the height. The fallbacks only ran on all-NaN input
        before the value-range work, which reshapes to one channel first; the new
        unmeasurable-range gates made them reachable for ordinary 2-D input, where
        emitting H values for a 1-channel datum fails reconciliation outright.
        """
        image = np.linspace(-50.0, 50.0, 64).reshape(8, 8)  # 2-D, negative, so no readable range

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pixel = compute_stats([image], stats=ImageStats.PIXEL_MEAN, per_channel=True, normalize_pixel_values=True)[
                "stats"
            ]
            visual = compute_stats(
                [image], stats=ImageStats.VISUAL_BRIGHTNESS, per_channel=True, normalize_pixel_values=False
            )["stats"]

        # One value per datum, not one per row of pixels.
        assert len(pixel["mean"]) == 1
        assert len(visual["brightness"]) == 1

    def test_empty_result_combines_without_a_name_mismatch(self):
        """A split that matched nothing has no columns, which is not a disagreement.

        `combine_stats_results` raises on differing statistic names so band groups cannot
        be silently dropped. A zero-row result carries an empty name set, which is not the
        caller having asked for different flags.
        """

        def stats_for(n: int):
            return compute_stats(np.zeros((n, 1, 8, 8)), stats=ImageStats.HASH_XXHASH, normalize_pixel_values=False)

        stats, source_index, steps = combine_stats_results([stats_for(3), stats_for(0), stats_for(2)])

        # The empty result contributes a boundary but no rows, so later items stay put.
        assert steps == [3, 3, 5]
        assert len(source_index) == 5
        assert len(stats["xxhash"]) == 5

    def test_empty_result_does_not_erase_the_statistics(self):
        """Regression on the *older* behavior: intersecting away every column.

        Before the raise, an empty result intersected the name set to nothing, so the
        combined map held no statistics at all and every downstream search ran over an
        empty table and reported a normal-looking empty answer.
        """
        stats, _, _ = combine_stats_results([
            compute_stats(np.zeros((3, 1, 8, 8)), stats=ImageStats.HASH_XXHASH, normalize_pixel_values=False),
            compute_stats(np.zeros((0, 1, 8, 8)), stats=ImageStats.HASH_XXHASH, normalize_pixel_values=False),
        ])

        assert "xxhash" in stats, "the populated result's statistics must survive an empty sibling"
        assert len(stats["xxhash"]) == 3

    def test_unreadable_range_warns_for_every_family_it_nans(self, caplog):
        """The warning must name every statistic it makes NaN, not just the pixel ones.

        `VISUAL` resolves the display range and `DIMENSION_DEPTH` reports the decoded
        encoding, so both answer NaN without an interval — and both were silent, which the
        concepts page explicitly promises against.
        """
        image = np.random.default_rng(0).uniform(-2000.0, 2000.0, (1, 8, 8))

        with caplog.at_level(logging.WARNING, logger="dataeval.core"):
            result = compute_stats(
                [image],
                stats=ImageStats.VISUAL_BRIGHTNESS | ImageStats.DIMENSION_DEPTH,
                normalize_pixel_values=False,
            )

        assert np.isnan(result["stats"]["brightness"]).all()
        assert np.isnan(result["stats"]["depth"]).all()
        assert "VISUAL_BRIGHTNESS" in caplog.text
        assert "DIMENSION_DEPTH" in caplog.text

    def test_missing_is_never_named_as_unmeasurable(self, caplog):
        """`missing` reads the raw view, so it always answers and must not be promised NaN."""
        image = np.random.default_rng(0).random((1, 8, 8)) * 1000.0

        with caplog.at_level(logging.WARNING, logger="dataeval.core"):
            result = compute_stats([image], stats=ImageStats.PIXEL, normalize_pixel_values=True)

        assert np.all(np.asarray(result["stats"]["missing"]) == 0.0)
        assert "PIXEL_MISSING" not in caplog.text


class TestReviewFindings:
    """Defects found by adversarial review of this branch, each verified before fixing."""

    def test_visual_statistics_stay_on_the_display_range(self):
        """A declared interval narrower than the data must not push a reading past 255.

        `edge_filter` clips at 0-255 internally, so without this `sharpness` saturated
        while `brightness` and `darkness` ran free — the family disagreeing about the scale
        it documents itself as being on.
        """
        image = np.linspace(0.0, 1000.0, 192).reshape(3, 8, 8)

        stats = compute_stats(
            [image],
            stats=ImageStats.VISUAL_BRIGHTNESS | ImageStats.VISUAL_DARKNESS | ImageStats.VISUAL_PERCENTILES,
            normalize_pixel_values=False,
            value_range=(0.0, 100.0),
        )["stats"]

        for name in ("brightness", "darkness"):
            assert 0.0 <= float(stats[name][0]) <= 255.0, f"{name} left the display range"
        assert np.nanmax(np.asarray(stats["percentiles"], dtype=np.float64)) <= 255.0

    def test_eight_bit_perceptual_view_is_still_the_identity(self):
        """Clipping must not cost the common case a copy."""
        image = (np.random.default_rng(0).random((3, 8, 8)) * 255).astype(np.uint8)
        cache = CalculatorCache(image)

        assert cache.perceptual is cache.image

    def test_wide_band_warning_is_silent_under_per_channel(self, caplog):
        """`per_channel=True` already measures each band, so nothing was averaged.

        The warning describes the opposite of what happened and points at `channels=`,
        which cannot be combined with `per_channel` anyway.
        """
        image = (np.random.default_rng(0).random((4, 16, 16)) * 255).astype(np.uint8)

        with caplog.at_level(logging.WARNING, logger="dataeval.core"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compute_stats([image], stats=ImageStats.VISUAL_BRIGHTNESS, per_channel=True, normalize_pixel_values=False)
        assert "single picture" not in caplog.text

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="dataeval.core"):
            compute_stats([image], stats=ImageStats.VISUAL_BRIGHTNESS, normalize_pixel_values=False)
        assert "single picture" in caplog.text, "it must still fire for an unnamed wide-band view"

    def test_channel_indices_accept_a_numpy_array(self):
        """`np.arange(3)` is an ordinary way to spell a band selection."""
        from dataeval.utils.preprocessing import ChannelGroup

        assert ChannelGroup(np.arange(3)).indices == (0, 1, 2)
        assert ChannelGroup([np.int64(0), np.int64(2)]).indices == (0, 2)

        with pytest.raises(ValueError, match="must be 1-D and of integer dtype"):
            ChannelGroup(np.array([[0, 1]]))
        with pytest.raises(ValueError, match="must be 1-D and of integer dtype"):
            ChannelGroup(np.array([0.5, 1.5]))


class TestHistogramRangeCoverage:
    """`np.histogram` drops values outside its range; that must not pass unremarked."""

    def test_declared_range_narrower_than_the_data_is_reported(self, caplog):
        """The bins give no sign of it, and the entropy derived from them reads as whole."""
        image = np.arange(400.0).reshape(1, 20, 20)

        with caplog.at_level(logging.WARNING, logger="dataeval.core"):
            result = compute_stats(
                [image],
                stats=ImageStats.PIXEL_HISTOGRAM | ImageStats.PIXEL_ENTROPY,
                normalize_pixel_values=False,
                value_range=(0.0, 100.0),
            )

        counted = int(np.asarray(result["stats"]["histogram"]).sum())
        assert counted < 400, "the setup must actually truncate for this to be testing anything"
        assert "fall outside the histogram range" in caplog.text
        assert f"{400 - counted} of 400" in caplog.text

    @pytest.mark.parametrize(
        "image",
        [
            (np.random.default_rng(0).random((3, 16, 16)) * 255).astype(np.uint8),
            (np.random.default_rng(0).random((3, 16, 16)) * 4095).astype(np.uint16),
            np.random.default_rng(0).random((3, 16, 16)),
        ],
        ids=["uint8", "uint16", "float_unit"],
    )
    def test_ordinary_imagery_never_reports_truncation(self, image, caplog):
        """A decoded range covers its data by construction, so this must stay silent."""
        with caplog.at_level(logging.WARNING, logger="dataeval.core"):
            compute_stats([image], stats=ImageStats.PIXEL_HISTOGRAM, normalize_pixel_values=False)

        assert "fall outside the histogram range" not in caplog.text
