from unittest.mock import patch

import numpy as np
import pytest

from dataeval.core._imagestats import (
    BaseProcessor,
    DimensionStatsProcessor,
    HashStatsProcessor,
    PixelPerChannelStatsProcessor,
    PixelStatsProcessor,
    ProcessorResult,
    StatsProcessorOutput,
    VisualPerChannelStatsProcessor,
    VisualStatsProcessor,
    _aggregate,
    _collect_processor_stats,
    _determine_channel_indices,
    _process_single,
    _reconcile_stats,
    _sort,
    process,
)
from dataeval.outputs._stats import SourceIndex
from dataeval.utils._boundingbox import BoundingBox


class TestBaseProcessor:
    def test_init_with_box(self):
        """Test BaseProcessor initialization with bounding box."""
        image = np.random.rand(3, 100, 100)
        box = BoundingBox(10, 10, 50, 50, image_shape=image.shape)
        processor = BaseProcessor(image, box)

        assert processor.width == 100
        assert processor.height == 100
        assert processor.shape == (3, 100, 100)
        assert processor.box == box

    def test_init_without_box(self):
        """Test BaseProcessor initialization without bounding box."""
        image = np.random.rand(3, 100, 100)
        processor = BaseProcessor(image, None)

        expected_box = BoundingBox(0, 0, 100, 100, image_shape=image.shape)
        assert processor.box.x0 == expected_box.x0
        assert processor.box.y0 == expected_box.y0
        assert processor.box.x1 == expected_box.x1
        assert processor.box.y1 == expected_box.y1

    @patch("dataeval.core._imagestats.clip_and_pad")
    @patch("dataeval.core._imagestats.normalize_image_shape")
    def test_image_property(self, mock_normalize, mock_clip):
        """Test image property caching and processing."""
        image = np.random.rand(3, 100, 100)
        mock_normalize.return_value = image
        mock_clip.return_value = image

        processor = BaseProcessor(image, None)
        result1 = processor.image
        result2 = processor.image

        assert result1 is result2  # Cached property
        mock_normalize.assert_called_once()
        mock_clip.assert_called_once()


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


class TestPixelPerChannelStatsProcessor:
    @pytest.mark.parametrize("n_channels", [1, 3])
    def test_process_per_channel(self, n_channels):
        """Test per-channel pixel statistics."""
        image = np.random.random((n_channels, 10, 10))  # nx10x10 image
        processor = PixelPerChannelStatsProcessor(image, None)

        stats = processor.process()

        assert len(stats["mean"]) == n_channels
        assert len(stats["std"]) == n_channels
        assert len(stats["histogram"]) == n_channels
        assert len(stats["histogram"][0]) == 256


class TestVisualPerChannelStatsProcessor:
    @pytest.mark.parametrize("n_channels", [1, 3])
    def test_process_visual_per_channel(self, n_channels):
        """Test per-channel visual statistics."""
        image = np.random.random((n_channels, 10, 10))  # nx10x10 image
        processor = VisualPerChannelStatsProcessor(image, None)

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
    @patch("dataeval.core._imagestats.xxhash")
    @patch("dataeval.core._imagestats.pchash")
    def test_process_hashes(self, mock_pchash, mock_xxhash):
        """Test hash statistics calculation."""
        mock_xxhash.return_value = "xxhash_result"
        mock_pchash.return_value = "pchash_result"

        image = np.random.rand(3, 50, 50)
        processor = HashStatsProcessor(image, None)

        stats = processor.process()

        assert stats["xxhash"][0] == "xxhash_result"
        assert stats["pchash"][0] == "pchash_result"


class TestHelperFunctions:
    def test_collect_processor_stats(self):
        """Test processor stats collection."""
        image = np.random.rand(3, 50, 50)
        processors = [PixelStatsProcessor, HashStatsProcessor]

        stats_list = _collect_processor_stats(processors, image, None)

        assert len(stats_list) == 2
        assert "mean" in stats_list[0]
        assert "xxhash" in stats_list[1]

    def test_determine_channel_indices_single_value(self):
        """Test channel index determination for single values."""
        processor_stats = [{"stat1": [0.5], "stat2": [0.3]}]

        indices = _determine_channel_indices(processor_stats, 3)

        assert indices == [None]

    def test_determine_channel_indices_per_channel(self):
        """Test channel index determination for per-channel values."""
        processor_stats = [{"stat1": [0.1, 0.2, 0.3]}]

        indices = _determine_channel_indices(processor_stats, 3)

        assert indices == [0, 1, 2]

    def test_determine_channel_indices_mixed(self):
        """Test channel index determination for mixed processors."""
        processor_stats = [
            {"stat1": [0.5]},  # Single value
            {"stat2": [0.1, 0.2, 0.3]},  # Per-channel
        ]

        indices = _determine_channel_indices(processor_stats, 3)

        assert indices == [None, 0, 1, 2]

    def test_determine_channel_indices_invalid(self):
        """Test channel index determination with invalid number of values."""
        processor_stats = [{"stat1": [0.1, 0.2]}]  # 2 values for 3-channel image

        with pytest.raises(ValueError, match="Processor produced 2 values"):
            _determine_channel_indices(processor_stats, 3)

    def test_reconcile_stats(self):
        """Test statistics reconciliation."""
        processor_stats = [{"single_stat": [0.5]}, {"channel_stat": [0.1, 0.2, 0.3]}]
        sorted_channels = [None, 0, 1, 2]

        reconciled = _reconcile_stats(processor_stats, sorted_channels)

        assert reconciled["single_stat"] == [0.5, None, None, None]
        assert reconciled["channel_stat"] == [None, 0.1, 0.2, 0.3]

    def test_sort(self):
        """Test results sorting."""
        source_indices = [
            SourceIndex(1, None, None),
            SourceIndex(0, 1, 0),
            SourceIndex(0, None, None),
            SourceIndex(0, 1, 1),
        ]
        stats = {"stat1": [1, 2, 3, 4]}

        sorted_indices, sorted_stats = _sort(source_indices, stats)

        assert [idx.image for idx in sorted_indices] == [0, 0, 0, 1]
        assert sorted_stats["stat1"] == [3, 2, 4, 1]

    def test_aggregate(self):
        """Test result aggregation."""
        result = StatsProcessorOutput(
            results=[ProcessorResult(source_indices=[SourceIndex(0, None, None)], stats={"stat1": [0.5]})],
            object_count=5,
            invalid_box_count=2,
            warnings_list=["Warning message"],
        )

        source_indices = []
        aggregated_stats = {}
        object_count = {}
        invalid_box_count = {}
        warning_list = []

        _aggregate(result, source_indices, aggregated_stats, object_count, invalid_box_count, warning_list)

        assert len(source_indices) == 1
        assert aggregated_stats["stat1"] == [0.5]
        assert object_count[0] == 5
        assert invalid_box_count[0] == 2
        assert warning_list == ["Warning message"]

    def test_aggregate_empty_source_indices(self):
        """Test aggregation with empty source indices."""
        result = StatsProcessorOutput(
            results=[
                ProcessorResult(
                    source_indices=[],  # Empty source indices
                    stats={"stat1": []},  # Empty stats to match
                )
            ],
            object_count=3,
            invalid_box_count=1,
            warnings_list=["Warning with empty indices"],
        )

        source_indices = []
        aggregated_stats = {}
        object_count = {}
        invalid_box_count = {}
        warning_list = []

        # Should not raise IndexError
        _aggregate(result, source_indices, aggregated_stats, object_count, invalid_box_count, warning_list)

        assert len(source_indices) == 0
        assert aggregated_stats["stat1"] == []
        assert len(object_count) == 0  # No image index available
        assert len(invalid_box_count) == 0  # No image index available
        assert warning_list == ["Warning with empty indices"]


class TestProcessSingle:
    def test_process_single_no_boxes(self):
        """Test processing single image without boxes."""
        image = np.random.rand(3, 50, 50)
        processors = [PixelStatsProcessor]

        result = _process_single(0, image, None, processors)

        assert len(result.results) == 1
        assert result.object_count == 0
        assert result.invalid_box_count == 0

    def test_process_single_with_boxes(self):
        """Test processing single image with boxes."""
        image = np.random.rand(3, 100, 100)
        boxes = [BoundingBox(0, 0, 50, 50, image_shape=image.shape)]
        processors = [PixelStatsProcessor]

        result = _process_single(0, image, boxes, processors)

        assert len(result.results) == 1
        assert result.object_count == 1

    def test_process_single_invalid_box(self):
        """Test processing with invalid bounding box."""
        image = np.random.rand(3, 50, 50)
        boxes = [BoundingBox(-1, -1, 0, 0, image_shape=image.shape)]
        processors = [PixelStatsProcessor]

        result = _process_single(0, image, boxes, processors)  # type: ignore

        assert result.invalid_box_count == 1
        assert len(result.warnings_list) == 1


class TestProcessMain:
    def test_process_single_processor(self):
        """Test main process function with single processor."""
        images = [np.random.rand(3, 50, 50)]

        with patch("dataeval.core._imagestats.get_max_processes", return_value=1):
            result = process(images, None, PixelStatsProcessor)

        assert "mean" in result
        assert "image_count" in result
        assert result["image_count"] == 1

    def test_process_multiple_processors(self):
        """Test main process function with multiple processors."""
        images = [np.random.rand(3, 50, 50)]
        processors = [PixelStatsProcessor, HashStatsProcessor]

        with patch("dataeval.core._imagestats.get_max_processes", return_value=1):
            result = process(images, None, processors)

        assert "mean" in result
        assert "xxhash" in result
        assert result["image_count"] == 1

    def test_process_with_boxes(self):
        """Test process function with bounding boxes."""
        images = [np.random.rand(3, 100, 100)]
        boxes = [[(10, 10, 50, 50)]]

        with patch("dataeval.core._imagestats.get_max_processes", return_value=1):
            result = process(images, boxes, PixelStatsProcessor)

        assert "object_count" in result
        assert result["object_count"][0] == 1

    @patch("dataeval.core._imagestats.warnings.warn")
    def test_process_with_warnings(self, mock_warn):
        """Test process function with warning generation."""
        images = [np.random.rand(3, 50, 50)]

        # Mock invalid box that generates warning
        with patch("dataeval.core._imagestats._process_single") as mock_process:
            mock_result = StatsProcessorOutput(
                results=[ProcessorResult([], {})], object_count=0, invalid_box_count=1, warnings_list=["Test warning"]
            )
            mock_process.return_value = mock_result

            with patch("dataeval.core._imagestats.get_max_processes", return_value=1):
                process(images, None, PixelStatsProcessor)

        mock_warn.assert_called_with("Test warning", UserWarning)

    def test_process_empty_images(self):
        """Test process function with empty image list."""
        images = []

        with patch("dataeval.core._imagestats.get_max_processes", return_value=1):
            result = process(images, None, PixelStatsProcessor)

        assert result["image_count"] == 0
