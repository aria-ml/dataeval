from unittest.mock import patch

import numpy as np
import pytest

from dataeval.core import calculate
from dataeval.core._calculate import CalculatorCache
from dataeval.core._calculators._imagestats import DimensionStatCalculator, HashStatCalculator
from dataeval.core.flags import ImageStats
from dataeval.utils._boundingbox import BoundingBox


class TestPixelStats:
    @pytest.mark.parametrize("n_channels", [1, 3])
    def test_process_basic_stats(self, n_channels):
        """Test pixel statistics calculation."""
        # Create deterministic image
        images = [np.random.random((n_channels, 10, 10))]

        result = calculate(images, None, stats=ImageStats.PIXEL, per_channel=False)

        assert "mean" in result
        assert "std" in result
        assert "var" in result
        assert "skew" in result
        assert "kurtosis" in result
        assert "entropy" in result
        assert "missing" in result
        assert "zeros" in result
        assert "histogram" in result

        assert len(result["mean"]) == 1
        assert type(result["mean"][0]) is float
        assert len(result["histogram"][0]) == 256

    def test_process_with_nans(self):
        """Test pixel statistics with NaN values."""
        images = [np.array([[[np.nan, 0.5], [0.5, 0.5]]])]

        result = calculate(images, None, stats=ImageStats.PIXEL, per_channel=False)
        assert result["missing"][0] > 0


class TestVisualStats:
    @pytest.mark.parametrize("n_channels", [1, 3])
    def test_process_visual_stats(self, n_channels):
        """Test visual statistics calculation."""
        images = [np.random.random((n_channels, 10, 10))]

        result = calculate(images, None, stats=ImageStats.VISUAL, per_channel=False)

        assert "brightness" in result
        assert "contrast" in result
        assert "darkness" in result
        assert "sharpness" in result
        assert "percentiles" in result

        assert len(result["brightness"]) == 1
        assert type(result["brightness"][0]) is float
        assert len(result["percentiles"][0]) == 5  # QUARTILES length


class TestPixelStatsPerChannel:
    @pytest.mark.parametrize("n_channels", [1, 3])
    def test_process_per_channel(self, n_channels):
        """Test per-channel pixel statistics."""
        images = [np.random.random((n_channels, 10, 10))]

        result = calculate(images, None, stats=ImageStats.PIXEL, per_channel=True)

        assert len(result["mean"]) == n_channels
        assert len(result["std"]) == n_channels
        assert len(result["histogram"]) == n_channels
        assert len(result["histogram"][0]) == 256


class TestVisualStatsPerChannel:
    @pytest.mark.parametrize("n_channels", [1, 3])
    def test_process_visual_per_channel(self, n_channels):
        """Test per-channel visual statistics."""
        images = [np.random.random((n_channels, 10, 10))]

        result = calculate(images, None, stats=ImageStats.VISUAL, per_channel=True)

        assert len(result["brightness"]) == n_channels
        assert len(result["contrast"]) == n_channels
        assert len(result["percentiles"]) == n_channels
        assert len(result["percentiles"][0]) == 5


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
        assert stats["aspect_ratio"][0] == pytest.approx(50 / 60)
        assert len(stats["center"][0]) == 2

    def test_process_invalid_box(self):
        """Test dimension stats with invalid bounding box."""
        image = np.random.rand(3, 100, 100)
        box = BoundingBox(-1, -1, 0, 0, image_shape=image.shape)
        datum_calculator = CalculatorCache(image, box)
        calculator = DimensionStatCalculator(image, datum_calculator)
        stats = calculator.compute(ImageStats.DIMENSION)

        assert stats["invalid_box"][0] is True


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
    """Test per_image and per_box parameter combinations."""

    def test_per_image_only_no_boxes(self):
        """Test per_image=True with no boxes provided."""
        images = [np.random.random((3, 10, 10))]

        result = calculate(images, None, stats=ImageStats.PIXEL_MEAN, per_image=True, per_box=True, per_channel=False)

        # Should have 1 result (full image)
        assert len(result["mean"]) == 1
        assert len(result["source_index"]) == 1
        assert result["source_index"][0].image == 0
        assert result["source_index"][0].box is None
        assert result["source_index"][0].channel is None

    def test_per_image_and_per_box_with_boxes(self):
        """Test per_image=True and per_box=True with boxes provided."""
        images = [np.random.random((3, 100, 100))]
        boxes = [
            [
                BoundingBox(0, 0, 50, 50, image_shape=(3, 100, 100)),
                BoundingBox(50, 50, 100, 100, image_shape=(3, 100, 100)),
            ]
        ]

        result = calculate(images, boxes, stats=ImageStats.PIXEL_MEAN, per_image=True, per_box=True, per_channel=False)

        # Should have 3 results: full image + 2 boxes
        assert len(result["mean"]) == 3
        assert len(result["source_index"]) == 3

        # First should be full image
        assert result["source_index"][0].image == 0
        assert result["source_index"][0].box is None

        # Next two should be boxes
        assert result["source_index"][1].image == 0
        assert result["source_index"][1].box == 0
        assert result["source_index"][2].image == 0
        assert result["source_index"][2].box == 1

    def test_per_box_only_with_boxes(self):
        """Test per_image=False and per_box=True with boxes provided."""
        images = [np.random.random((3, 100, 100))]
        boxes = [
            [
                BoundingBox(0, 0, 50, 50, image_shape=(3, 100, 100)),
                BoundingBox(50, 50, 100, 100, image_shape=(3, 100, 100)),
            ]
        ]

        result = calculate(images, boxes, stats=ImageStats.PIXEL_MEAN, per_image=False, per_box=True, per_channel=False)

        # Should have 2 results: only boxes
        assert len(result["mean"]) == 2
        assert len(result["source_index"]) == 2

        # Both should be boxes (no full image)
        assert result["source_index"][0].image == 0
        assert result["source_index"][0].box == 0
        assert result["source_index"][1].image == 0
        assert result["source_index"][1].box == 1

    def test_per_image_only_with_boxes_ignored(self):
        """Test per_image=True and per_box=False with boxes provided (boxes ignored)."""
        images = [np.random.random((3, 100, 100))]
        boxes = [
            [
                BoundingBox(0, 0, 50, 50, image_shape=(3, 100, 100)),
                BoundingBox(50, 50, 100, 100, image_shape=(3, 100, 100)),
            ]
        ]

        result = calculate(images, boxes, stats=ImageStats.PIXEL_MEAN, per_image=True, per_box=False, per_channel=False)

        # Should have 1 result: only full image (boxes ignored)
        assert len(result["mean"]) == 1
        assert len(result["source_index"]) == 1

        # Should be full image
        assert result["source_index"][0].image == 0
        assert result["source_index"][0].box is None

    def test_per_image_and_per_box_with_per_channel(self):
        """Test per_image=True, per_box=True, and per_channel=True."""
        images = [np.random.random((3, 100, 100))]
        boxes = [[BoundingBox(0, 0, 50, 50, image_shape=(3, 100, 100))]]

        result = calculate(images, boxes, stats=ImageStats.PIXEL_MEAN, per_image=True, per_box=True, per_channel=True)

        # Should have 6 results: (full image + 1 box) × 3 channels = 6
        assert len(result["mean"]) == 6
        assert len(result["source_index"]) == 6

        # Check structure: full image channels first, then box channels
        # Full image - channel 0, 1, 2
        assert result["source_index"][0].image == 0
        assert result["source_index"][0].box is None
        assert result["source_index"][0].channel == 0

        assert result["source_index"][1].image == 0
        assert result["source_index"][1].box is None
        assert result["source_index"][1].channel == 1

        assert result["source_index"][2].image == 0
        assert result["source_index"][2].box is None
        assert result["source_index"][2].channel == 2

        # Box - channel 0, 1, 2
        assert result["source_index"][3].image == 0
        assert result["source_index"][3].box == 0
        assert result["source_index"][3].channel == 0

        assert result["source_index"][4].image == 0
        assert result["source_index"][4].box == 0
        assert result["source_index"][4].channel == 1

        assert result["source_index"][5].image == 0
        assert result["source_index"][5].box == 0
        assert result["source_index"][5].channel == 2

    def test_multiple_images_per_image_and_per_box(self):
        """Test multiple images with per_image=True and per_box=True."""
        images = [np.random.random((3, 100, 100)), np.random.random((3, 100, 100))]
        boxes = [
            [BoundingBox(0, 0, 50, 50, image_shape=(3, 100, 100))],
            [
                BoundingBox(25, 25, 75, 75, image_shape=(3, 100, 100)),
                BoundingBox(50, 50, 100, 100, image_shape=(3, 100, 100)),
            ],
        ]

        result = calculate(images, boxes, stats=ImageStats.PIXEL_MEAN, per_image=True, per_box=True, per_channel=False)

        # Should have 5 results: image0 (1 full + 1 box) + image1 (1 full + 2 boxes)
        assert len(result["mean"]) == 5
        assert len(result["source_index"]) == 5

        # Image 0: full image
        assert result["source_index"][0].image == 0
        assert result["source_index"][0].box is None

        # Image 0: box 0
        assert result["source_index"][1].image == 0
        assert result["source_index"][1].box == 0

        # Image 1: full image
        assert result["source_index"][2].image == 1
        assert result["source_index"][2].box is None

        # Image 1: box 0
        assert result["source_index"][3].image == 1
        assert result["source_index"][3].box == 0

        # Image 1: box 1
        assert result["source_index"][4].image == 1
        assert result["source_index"][4].box == 1

    def test_invalid_both_false_raises_error(self):
        """Test that per_image=False and per_box=False raises ValueError."""
        images = [np.random.random((3, 10, 10))]

        with pytest.raises(ValueError, match="At least one of 'per_image' or 'per_box' must be True"):
            calculate(images, None, stats=ImageStats.PIXEL_MEAN, per_image=False, per_box=False, per_channel=False)

    def test_object_count_tracking(self):
        """Test that object_count is correctly tracked with per_image and per_box."""
        images = [np.random.random((3, 100, 100))]
        boxes = [
            [
                BoundingBox(0, 0, 50, 50, image_shape=(3, 100, 100)),
                BoundingBox(50, 50, 100, 100, image_shape=(3, 100, 100)),
            ]
        ]

        result = calculate(images, boxes, stats=ImageStats.PIXEL_MEAN, per_image=True, per_box=True, per_channel=False)

        # Object count should be 2 (number of boxes)
        assert result["object_count"][0] == 2
        assert result["image_count"] == 1
