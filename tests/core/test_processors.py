from unittest.mock import patch

import numpy as np
import pytest

from dataeval.core.processors._dimensionstats import DimensionStatsProcessor
from dataeval.core.processors._hashstats import HashStatsProcessor
from dataeval.core.processors._pixelstats import PixelStatsPerChannelProcessor, PixelStatsProcessor
from dataeval.core.processors._visualstats import VisualStatsPerChannelProcessor, VisualStatsProcessor
from dataeval.utils._boundingbox import BoundingBox


class TestPixelStatsProcessor:
    @pytest.mark.parametrize("n_channels", [1, 3])
    def test_process_basic_stats_single_channel(self, n_channels):
        """Test pixel statistics calculation."""
        # Create deterministic image
        image = np.random.random((n_channels, 10, 10))  # nx10x10 image
        processor = PixelStatsProcessor(image, None)

        with patch.object(processor, "scaled", image):
            stats = processor.process()

        assert "mean" in stats
        assert "std" in stats
        assert "var" in stats
        assert "skew" in stats
        assert "kurtosis" in stats
        assert "entropy" in stats
        assert "missing" in stats
        assert "zeros" in stats
        assert "histogram" in stats

        assert len(stats["mean"]) == 1
        assert type(stats["mean"][0]) is float
        assert len(stats["histogram"][0]) == 256

    def test_process_with_nans(self):
        """Test pixel statistics with NaN values."""
        image = np.array([[[np.nan, 0.5], [0.5, 0.5]]])
        processor = PixelStatsProcessor(image, None)

        stats = processor.process()
        assert stats["missing"][0] > 0


class TestVisualStatsProcessor:
    @pytest.mark.parametrize("n_channels", [1, 3])
    def test_process_visual_stats(self, n_channels):
        """Test visual statistics calculation."""
        image = np.random.random((n_channels, 10, 10))  # nx10x10 image
        processor = VisualStatsProcessor(image, None)

        stats = processor.process()

        assert "brightness" in stats
        assert "contrast" in stats
        assert "darkness" in stats
        assert "sharpness" in stats
        assert "percentiles" in stats

        assert len(stats["brightness"]) == 1
        assert type(stats["brightness"][0]) is float
        assert len(stats["percentiles"][0]) == 5  # QUARTILES length


class TestPixelStatsPerChannelProcessor:
    @pytest.mark.parametrize("n_channels", [1, 3])
    def test_process_per_channel(self, n_channels):
        """Test per-channel pixel statistics."""
        image = np.random.random((n_channels, 10, 10))  # nx10x10 image
        processor = PixelStatsPerChannelProcessor(image, None)

        stats = processor.process()

        assert len(stats["mean"]) == n_channels
        assert len(stats["std"]) == n_channels
        assert len(stats["histogram"]) == n_channels
        assert len(stats["histogram"][0]) == 256


class TestVisualStatsPerChannelProcessor:
    @pytest.mark.parametrize("n_channels", [1, 3])
    def test_process_visual_per_channel(self, n_channels):
        """Test per-channel visual statistics."""
        image = np.random.random((n_channels, 10, 10))  # nx10x10 image
        processor = VisualStatsPerChannelProcessor(image, None)

        stats = processor.process()

        assert len(stats["brightness"]) == n_channels
        assert len(stats["contrast"]) == n_channels
        assert len(stats["percentiles"]) == n_channels
        assert len(stats["percentiles"][0]) == 5


class TestDimensionStatsProcessor:
    def test_process_dimensions(self):
        """Test dimension statistics calculation."""
        image = np.random.rand(3, 100, 150)
        box = BoundingBox(10, 20, 60, 80, image_shape=image.shape)
        processor = DimensionStatsProcessor(image, box)

        stats = processor.process()

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
        processor = DimensionStatsProcessor(image, box)
        stats = processor.process()

        assert stats["invalid_box"][0] is True


class TestHashStatsProcessor:
    @patch("dataeval.core.processors._hashstats.xxhash")
    @patch("dataeval.core.processors._hashstats.pchash")
    def test_process_hashes(self, mock_pchash, mock_xxhash):
        """Test hash statistics calculation."""
        mock_xxhash.return_value = "xxhash_result"
        mock_pchash.return_value = "pchash_result"

        image = np.random.rand(3, 50, 50)
        processor = HashStatsProcessor(image, None)

        stats = processor.process()

        assert stats["xxhash"][0] == "xxhash_result"
        assert stats["pchash"][0] == "pchash_result"
