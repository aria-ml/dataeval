from unittest.mock import patch

import numpy as np
import pytest

from dataeval.core import calculate
from dataeval.core._calculate import CalculatorCache
from dataeval.core._calculators._dimensionstats import DimensionStatCalculator
from dataeval.core._calculators._hashstats import HashStatCalculator
from dataeval.flags import ImageStats
from dataeval.utils.preprocessing import BoundingBox


class TestPixelStats:
    @pytest.mark.parametrize("n_channels", [1, 3])
    def test_process_basic_stats(self, n_channels):
        """Test pixel statistics calculation."""
        # Create deterministic image
        images = [np.random.random((n_channels, 10, 10))]

        result = calculate(images, None, stats=ImageStats.PIXEL, per_channel=False)

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
        assert result["stats"]["mean"].dtype == np.float16
        assert len(result["stats"]["histogram"][0]) == 256

    def test_process_with_nans(self):
        """Test pixel statistics with NaN values."""
        images = [np.array([[[np.nan, 0.5], [0.5, 0.5]]])]

        result = calculate(images, None, stats=ImageStats.PIXEL, per_channel=False)
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
                ]
            )
        ]

        result = calculate(images, None, stats=ImageStats.PIXEL_MISSING, per_channel=False)

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
                ]
            )
        ]

        result = calculate(images, None, stats=ImageStats.PIXEL_MISSING, per_channel=True)

        # Per-channel mode should return list with one value per channel
        assert len(result["stats"]["missing"]) == 3
        assert result["stats"]["missing"][0] == pytest.approx(0.25, abs=1e-4)
        assert result["stats"]["missing"][1] == pytest.approx(0.5, abs=1e-4)
        assert result["stats"]["missing"][2] == pytest.approx(0.0, abs=1e-4)

    def test_missing_single_channel_image(self):
        """Test missing calculation for single-channel image."""
        # Single channel (1, 3, 3) with 2 NaNs out of 9 pixels
        images = [np.array([[[np.nan, 1.0, 1.0], [1.0, np.nan, 1.0], [1.0, 1.0, 1.0]]])]

        result = calculate(images, None, stats=ImageStats.PIXEL_MISSING, per_channel=False)

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
                ]
            )
        ]

        result = calculate(images, None, stats=ImageStats.PIXEL_ZEROS, per_channel=False)

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
                ]
            )
        ]

        result = calculate(images, None, stats=ImageStats.PIXEL_ZEROS, per_channel=True)

        # Per-channel mode should return list with one value per channel
        assert len(result["stats"]["zeros"]) == 3
        assert result["stats"]["zeros"][0] == pytest.approx(0.25, abs=1e-4)
        assert result["stats"]["zeros"][1] == pytest.approx(0.5, abs=1e-4)
        assert result["stats"]["zeros"][2] == pytest.approx(0.0, abs=1e-4)

    def test_zeros_single_channel_image(self):
        """Test zeros calculation for single-channel image."""
        # Single channel (1, 3, 3) with 2 zeros out of 9 pixels
        images = [np.array([[[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]])]

        result = calculate(images, None, stats=ImageStats.PIXEL_ZEROS, per_channel=False)

        # 2 zeros / 9 total = 0.222...
        expected_zeros = 2 / 9
        assert result["stats"]["zeros"][0] == pytest.approx(expected_zeros, abs=1e-4)

    def test_zeros_all_zeros_image(self):
        """Test zeros calculation when entire image is zeros."""
        images = [np.zeros((3, 10, 10))]

        result = calculate(images, None, stats=ImageStats.PIXEL_ZEROS, per_channel=False)

        # All pixels are zero, so should be 1.0
        assert result["stats"]["zeros"][0] == pytest.approx(1.0, abs=1e-4)

    def test_missing_all_nans_image(self):
        """Test missing calculation when entire image is NaN."""
        images = [np.full((3, 10, 10), np.nan)]

        result = calculate(images, None, stats=ImageStats.PIXEL_MISSING, per_channel=False)

        # All pixels are NaN, so should be 1.0
        assert result["stats"]["missing"][0] == pytest.approx(1.0, abs=1e-4)


class TestVisualStats:
    @pytest.mark.parametrize("n_channels", [1, 3])
    def test_process_visual_stats(self, n_channels):
        """Test visual statistics calculation."""
        images = [np.random.random((n_channels, 10, 10))]

        result = calculate(images, None, stats=ImageStats.VISUAL, per_channel=False)

        assert "brightness" in result["stats"]
        assert "contrast" in result["stats"]
        assert "darkness" in result["stats"]
        assert "sharpness" in result["stats"]
        assert "percentiles" in result["stats"]

        assert len(result["stats"]["brightness"]) == 1
        assert result["stats"]["brightness"].dtype == np.float16
        assert len(result["stats"]["percentiles"][0]) == 5  # QUARTILES length


class TestPixelStatsPerChannel:
    @pytest.mark.parametrize("n_channels", [1, 3])
    def test_process_per_channel(self, n_channels):
        """Test per-channel pixel statistics."""
        images = [np.random.random((n_channels, 10, 10))]

        result = calculate(images, None, stats=ImageStats.PIXEL, per_channel=True)

        assert len(result["stats"]["mean"]) == n_channels
        assert len(result["stats"]["std"]) == n_channels
        assert len(result["stats"]["histogram"]) == n_channels
        assert len(result["stats"]["histogram"][0]) == 256


class TestVisualStatsPerChannel:
    @pytest.mark.parametrize("n_channels", [1, 3])
    def test_process_visual_per_channel(self, n_channels):
        """Test per-channel visual statistics."""
        images = [np.random.random((n_channels, 10, 10))]

        result = calculate(images, None, stats=ImageStats.VISUAL, per_channel=True)

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
    @patch("dataeval.core._hash.xxhash")
    @patch("dataeval.core._hash.pchash")
    def test_process_hashes(self, mock_pchash, mock_xxhash):
        """Test hash statistics calculation."""
        mock_xxhash.return_value = "xxhash_result"
        mock_pchash.return_value = "pchash_result"

        image = np.random.rand(3, 50, 50)
        datum_calculator = CalculatorCache(image, None)
        calculator = HashStatCalculator(image, datum_calculator)

        stats = calculator.compute(ImageStats.HASH)

        assert stats["xxhash"][0] == "xxhash_result"
        assert stats["pchash"][0] == "pchash_result"


class TestPerImagePerBox:
    """Test per_image and per_target parameter combinations."""

    def test_per_image_only_no_boxes(self):
        """Test per_image=True with no boxes provided."""
        images = [np.random.random((3, 10, 10))]

        result = calculate(
            images, None, stats=ImageStats.PIXEL_MEAN, per_image=True, per_target=True, per_channel=False
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
            ]
        ]

        result = calculate(
            images, boxes, stats=ImageStats.PIXEL_MEAN, per_image=True, per_target=True, per_channel=False
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
            ]
        ]

        result = calculate(
            images, boxes, stats=ImageStats.PIXEL_MEAN, per_image=False, per_target=True, per_channel=False
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
            ]
        ]

        result = calculate(
            images, boxes, stats=ImageStats.PIXEL_MEAN, per_image=True, per_target=False, per_channel=False
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

        result = calculate(
            images, boxes, stats=ImageStats.PIXEL_MEAN, per_image=True, per_target=True, per_channel=True
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

        result = calculate(
            images, boxes, stats=ImageStats.PIXEL_MEAN, per_image=True, per_target=True, per_channel=False
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

        with pytest.raises(ValueError, match="At least one of 'per_image' or 'per_target' must be True"):
            calculate(images, None, stats=ImageStats.PIXEL_MEAN, per_image=False, per_target=False, per_channel=False)

    def test_object_count_tracking(self):
        """Test that object_count is correctly tracked with per_image and per_target."""
        images = [np.random.random((3, 100, 100))]
        boxes = [
            [
                BoundingBox(0, 0, 50, 50, image_shape=(3, 100, 100)),
                BoundingBox(50, 50, 100, 100, image_shape=(3, 100, 100)),
            ]
        ]

        result = calculate(
            images, boxes, stats=ImageStats.PIXEL_MEAN, per_image=True, per_target=True, per_channel=False
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

        result = calculate(data, None, stats=ImageStats.PIXEL, per_channel=False)

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
        assert result["stats"]["mean"].dtype == np.float16
        assert len(result["stats"]["histogram"][0]) == 256

    def test_2d_data_pixel_stats(self):
        """Test pixel statistics calculation with 2D data (single channel image)."""
        # Create 2D data (shape: (height, width))
        data = [np.random.random((10, 10))]

        result = calculate(data, None, stats=ImageStats.PIXEL, per_channel=False)

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
        assert result["stats"]["mean"].dtype == np.float16
        assert len(result["stats"]["histogram"][0]) == 256

    def test_1d_data_with_nans(self):
        """Test pixel statistics with 1D data containing NaN values."""
        data = [np.array([np.nan, 0.5, 0.5, 0.5, np.nan])]

        result = calculate(data, None, stats=ImageStats.PIXEL, per_channel=False)
        assert result["stats"]["missing"][0] > 0

    def test_1d_data_per_channel(self):
        """Test that 1D data is treated as single channel when per_channel=True."""
        data = [np.random.random(100)]

        result = calculate(data, None, stats=ImageStats.PIXEL, per_channel=True)

        # Should be treated as 1 channel
        assert len(result["stats"]["mean"]) == 1
        assert len(result["stats"]["std"]) == 1
        assert len(result["stats"]["histogram"]) == 1

    def test_2d_data_per_channel(self):
        """Test that 2D data is treated as single channel when per_channel=True."""
        data = [np.random.random((10, 10))]

        result = calculate(data, None, stats=ImageStats.PIXEL, per_channel=True)

        # Should be treated as 1 channel
        assert len(result["stats"]["mean"]) == 1
        assert len(result["stats"]["std"]) == 1
        assert len(result["stats"]["histogram"]) == 1


class TestLowerDimensionalVisualStats:
    """Test visual statistics with lower dimensional data (1D and 2D)."""

    def test_1d_data_visual_stats(self):
        """Test visual statistics calculation with 1D data."""
        data = [np.random.random(100)]

        result = calculate(data, None, stats=ImageStats.VISUAL, per_channel=False)

        assert "brightness" in result["stats"]
        assert "contrast" in result["stats"]
        assert "darkness" in result["stats"]
        assert "sharpness" in result["stats"]
        assert "percentiles" in result["stats"]

        assert len(result["stats"]["brightness"]) == 1
        assert result["stats"]["brightness"].dtype == np.float16
        assert len(result["stats"]["percentiles"][0]) == 5  # QUARTILES length
        # Sharpness should be NaN for 1D data
        assert np.isnan(result["stats"]["sharpness"][0])

    def test_2d_data_visual_stats(self):
        """Test visual statistics calculation with 2D data."""
        data = [np.random.random((10, 10))]

        result = calculate(data, None, stats=ImageStats.VISUAL, per_channel=False)

        assert "brightness" in result["stats"]
        assert "contrast" in result["stats"]
        assert "darkness" in result["stats"]
        assert "sharpness" in result["stats"]
        assert "percentiles" in result["stats"]

        assert len(result["stats"]["brightness"]) == 1
        assert result["stats"]["brightness"].dtype == np.float16
        assert len(result["stats"]["percentiles"][0]) == 5
        # Sharpness should be computed for 2D data
        assert result["stats"]["sharpness"].dtype == np.float16
        assert not np.isnan(result["stats"]["sharpness"][0])

    def test_1d_data_visual_stats_per_channel(self):
        """Test visual statistics with 1D data and per_channel=True."""
        data = [np.random.random(100)]

        result = calculate(data, None, stats=ImageStats.VISUAL, per_channel=True)

        # Should be treated as 1 channel
        assert len(result["stats"]["brightness"]) == 1
        assert len(result["stats"]["contrast"]) == 1
        assert len(result["stats"]["sharpness"]) == 1
        # Sharpness should be NaN for 1D data
        assert np.isnan(result["stats"]["sharpness"][0])

    def test_2d_data_visual_stats_per_channel(self):
        """Test visual statistics with 2D data and per_channel=True."""
        data = [np.random.random((10, 10))]

        result = calculate(data, None, stats=ImageStats.VISUAL, per_channel=True)

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
        """Test dimension statistics via calculate() with 1D data."""
        data = [np.random.random(100)]

        result = calculate(data, None, stats=ImageStats.DIMENSION, per_channel=False)

        assert "width" in result["stats"]
        assert "height" in result["stats"]
        assert "channels" in result["stats"]
        assert "size" in result["stats"]

        assert result["stats"]["width"][0] == 100
        assert np.isnan(result["stats"]["height"][0])
        assert result["stats"]["channels"][0] == 1
        assert result["stats"]["size"][0] == 100

    def test_calculate_2d_dimension_stats(self):
        """Test dimension statistics via calculate() with 2D data."""
        data = [np.random.random((10, 20))]

        result = calculate(data, None, stats=ImageStats.DIMENSION, per_channel=False)

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

    @patch("dataeval.core._hash.xxhash")
    @patch("dataeval.core._hash.pchash")
    def test_1d_data_hashes(self, mock_pchash, mock_xxhash):
        """Test hash statistics calculation with 1D data."""
        mock_xxhash.return_value = "xxhash_1d_result"
        mock_pchash.return_value = "pchash_1d_result"

        data = np.random.rand(50)
        datum_calculator = CalculatorCache(data, None)
        calculator = HashStatCalculator(data, datum_calculator)

        stats = calculator.compute(ImageStats.HASH)

        assert stats["xxhash"][0] == "xxhash_1d_result"
        assert stats["pchash"][0] == "pchash_1d_result"

    @patch("dataeval.core._hash.xxhash")
    @patch("dataeval.core._hash.pchash")
    def test_2d_data_hashes(self, mock_pchash, mock_xxhash):
        """Test hash statistics calculation with 2D data."""
        mock_xxhash.return_value = "xxhash_2d_result"
        mock_pchash.return_value = "pchash_2d_result"

        data = np.random.rand(10, 10)
        datum_calculator = CalculatorCache(data, None)
        calculator = HashStatCalculator(data, datum_calculator)

        stats = calculator.compute(ImageStats.HASH)

        assert stats["xxhash"][0] == "xxhash_2d_result"
        assert stats["pchash"][0] == "pchash_2d_result"

    @patch("dataeval.core._hash.xxhash")
    @patch("dataeval.core._hash.pchash")
    def test_calculate_1d_hash_stats(self, mock_pchash, mock_xxhash):
        """Test hash statistics via calculate() with 1D data."""
        mock_xxhash.return_value = "xxhash_calc_result"
        mock_pchash.return_value = "pchash_calc_result"

        data = [np.random.random(100)]

        result = calculate(data, None, stats=ImageStats.HASH, per_channel=False)

        assert "xxhash" in result["stats"]
        assert "pchash" in result["stats"]
        assert result["stats"]["xxhash"][0] == "xxhash_calc_result"
        assert result["stats"]["pchash"][0] == "pchash_calc_result"

    def test_1d_data_pchash_warning(self, caplog):
        """Test that pchash emits a warning for 1D data."""
        import logging

        data = np.random.rand(50)
        datum_calculator = CalculatorCache(data, None)
        calculator = HashStatCalculator(data, datum_calculator)

        with caplog.at_level(logging.WARNING):
            stats = calculator.compute(ImageStats.HASH)

        assert "Perceptual hashing requires spatial data" in caplog.text

        # pchash should return empty string for 1D data
        assert stats["pchash"][0] == ""
        # xxhash should still work
        assert stats["xxhash"][0] != ""

    def test_1d_data_pchash_returns_empty_via_calculate(self):
        """Test that pchash returns empty string for 1D data via calculate().

        Note: The warning is emitted but not captured due to multiprocessing.
        We test the behavior (empty string return) instead.
        """
        data = [np.random.random(100)]

        result = calculate(data, None, stats=ImageStats.HASH, per_channel=False)

        # pchash should return empty string for 1D data
        assert result["stats"]["pchash"][0] == ""
        # xxhash should still work
        assert result["stats"]["xxhash"][0] != ""

    def test_2d_small_image_pchash_warning(self, caplog):
        """Test that pchash emits a warning for images smaller than 9x9."""
        import logging

        # Create a 5x5 image (smaller than required 9x9)
        data = np.random.rand(5, 5)
        datum_calculator = CalculatorCache(data, None)
        calculator = HashStatCalculator(data, datum_calculator)

        with caplog.at_level(logging.WARNING):
            stats = calculator.compute(ImageStats.HASH)

        assert "Image too small for perceptual hashing" in caplog.text

        # pchash should return empty string for small images
        assert stats["pchash"][0] == ""
        # xxhash should still work
        assert stats["xxhash"][0] != ""


class TestImageClassificationDataset:
    """Test calculate() with ImageClassificationDataset input."""

    def test_ic_dataset_without_boxes(self, get_mock_ic_dataset):
        """Test ImageClassificationDataset processing without boxes."""
        images = [np.random.random((3, 100, 100)) for _ in range(3)]
        labels = [0, 1, 0]

        dataset = get_mock_ic_dataset(images, labels)

        result = calculate(dataset, stats=ImageStats.PIXEL_MEAN, per_image=True, per_target=True, per_channel=False)

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

        result = calculate(
            dataset, boxes=boxes, stats=ImageStats.PIXEL_MEAN, per_image=True, per_target=True, per_channel=False
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

        result = calculate(dataset, stats=ImageStats.PIXEL_MEAN, per_image=True, per_target=True, per_channel=True)

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

        result = calculate(dataset, stats=ImageStats.PIXEL | ImageStats.VISUAL, per_image=True, per_channel=False)
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
    """Test calculate() with ObjectDetectionDataset input."""

    def test_od_dataset_with_boxes(self, get_mock_od_dataset):
        """Test ObjectDetectionDataset automatically processes boxes."""
        images = [np.random.random((3, 100, 100)) for _ in range(2)]
        labels = [[0, 1], [1]]
        bboxes = [
            [[10, 10, 50, 50], [60, 60, 90, 90]],
            [[20, 20, 70, 70]],
        ]

        dataset = get_mock_od_dataset(images, labels, bboxes)

        result = calculate(dataset, stats=ImageStats.PIXEL_MEAN, per_image=True, per_target=True, per_channel=False)

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

        result = calculate(dataset, stats=ImageStats.PIXEL_MEAN, per_image=False, per_target=True, per_channel=False)

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

        result = calculate(dataset, stats=ImageStats.PIXEL_MEAN, per_image=True, per_target=False, per_channel=False)

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

        result = calculate(dataset, stats=ImageStats.PIXEL_MEAN, per_image=True, per_target=True, per_channel=True)

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

        result = calculate(dataset, stats=ImageStats.DIMENSION, per_image=False, per_target=True, per_channel=False)

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

        result = calculate(
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

        result = calculate(dataset, stats=ImageStats.PIXEL_MEAN, per_image=True, per_target=True, per_channel=False)

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

        result = calculate(
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
        """Test that progress_callback is called during calculation"""
        images = [np.random.random((3, 10, 10)) for _ in range(5)]
        callback_calls = []

        def callback(step: int, *, total: int | None = None, desc: str | None = None, extra_info: dict | None = None):
            callback_calls.append({"step": step, "total": total})

        result = calculate(images, None, stats=ImageStats.PIXEL, progress_callback=callback)

        # Callback should have been called for each image
        assert len(callback_calls) == 5
        assert result["image_count"] == 5

        # Verify callbacks have correct step values
        for i, call in enumerate(callback_calls):
            assert call["step"] == i + 1
            assert call["total"] == 5

    def test_progress_callback_not_called_when_none(self):
        """Test that no error occurs when progress_callback is None"""
        images = [np.random.random((3, 10, 10)) for _ in range(3)]

        result = calculate(images, None, stats=ImageStats.PIXEL, progress_callback=None)

        # Should work without error
        assert result["image_count"] == 3

    def test_progress_callback_with_boxes(self):
        """Test that progress_callback works with bounding boxes"""
        images = [np.random.random((3, 100, 100)) for _ in range(3)]
        boxes = [[[10, 10, 50, 50], [20, 20, 60, 60]] for _ in range(3)]
        callback_calls = []

        def callback(step: int, *, total: int | None = None, desc: str | None = None, extra_info: dict | None = None):
            callback_calls.append({"step": step, "total": total})

        result = calculate(images, boxes, stats=ImageStats.DIMENSION, progress_callback=callback)

        # Callback should be called for each image (not each box)
        assert len(callback_calls) == 3
        assert result["image_count"] == 3

        # Verify step counts
        for i, call in enumerate(callback_calls):
            assert call["step"] == i + 1
            assert call["total"] == 3

    def test_progress_callback_with_dataset(self, get_mock_od_dataset):
        """Test that progress_callback works with Dataset input"""
        images = [np.random.random((3, 100, 100)) for _ in range(4)]
        labels = [[0, 1] for _ in range(4)]
        bboxes = [[[10, 10, 50, 50], [20, 20, 60, 60]] for _ in range(4)]

        dataset = get_mock_od_dataset(images, labels, bboxes)
        callback_calls = []

        def callback(step: int, *, total: int | None = None, desc: str | None = None, extra_info: dict | None = None):
            callback_calls.append({"step": step, "total": total})

        result = calculate(dataset, stats=ImageStats.PIXEL, progress_callback=callback)

        # Callback should be called for each image
        assert len(callback_calls) == 4
        assert result["image_count"] == 4

        # Verify total is provided for Dataset (which is Sized)
        for call in callback_calls:
            assert call["total"] == 4

    def test_progress_callback_incremental_steps(self):
        """Test that progress_callback receives incremental step counts"""
        images = [np.random.random((3, 10, 10)) for _ in range(10)]
        callback_calls = []

        def callback(step: int, *, total: int | None = None, desc: str | None = None, extra_info: dict | None = None):
            callback_calls.append(step)

        calculate(images, None, stats=ImageStats.PIXEL_BASIC, progress_callback=callback)

        # Steps should be 1, 2, 3, ..., 10
        assert callback_calls == list(range(1, 11))

    def test_calculate_with_empty_dataset(self):
        """Test calculate with empty dataset"""
        images = []
        result = calculate(images, None, stats=ImageStats.PIXEL)

        assert result["image_count"] == 0
        assert len(result["source_index"]) == 0
        assert len(result["object_count"]) == 0
        assert len(result["invalid_box_count"]) == 0

    def test_calculate_determine_channel_indices_error(self):
        """Test _determine_channel_indices raises error for unexpected output (line 190)"""
        from dataeval.core._calculate import _determine_channel_indices

        # Create calculator output with unexpected number of elements
        # (not 1 for image-level, not equal to num_channels for per-channel)
        calculator_output = [{"stat1": [1, 2, 3]}]  # 3 values but image has 1 channel
        num_channels = 1

        with pytest.raises(ValueError, match="Processor produced"):
            _determine_channel_indices(calculator_output, num_channels)
