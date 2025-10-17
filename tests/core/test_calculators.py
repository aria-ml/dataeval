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
