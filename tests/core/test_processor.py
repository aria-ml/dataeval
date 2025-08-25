from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dataeval.core._processor import (
    BaseProcessor,
    ProcessorOutput,
    ProcessorResult,
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


class MockFooStatsProcessor(BaseProcessor):
    def process(self) -> dict[str, list[Any]]:
        return {
            "foo": ["foo"],
        }


class MockBarStatsProcessor(BaseProcessor):
    def process(self) -> dict[str, list[Any]]:
        return {
            "bar": ["bar"],
        }


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

    @patch("dataeval.core._processor.clip_and_pad")
    @patch("dataeval.core._processor.normalize_image_shape")
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


class TestHelperFunctions:
    def test_collect_processor_stats(self):
        """Test processor stats collection."""
        image = np.random.rand(3, 50, 50)
        processors = [MockFooStatsProcessor, MockBarStatsProcessor]

        stats_list = _collect_processor_stats(processors, image, None)

        assert len(stats_list) == 2
        assert "foo" in stats_list[0]
        assert "bar" in stats_list[1]

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
        result = ProcessorOutput(
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
        result = ProcessorOutput(
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
        processors = [MockFooStatsProcessor]

        result = _process_single(0, image, None, processors)

        assert len(result.results) == 1
        assert result.object_count == 0
        assert result.invalid_box_count == 0

    def test_process_single_with_boxes(self):
        """Test processing single image with boxes."""
        image = np.random.rand(3, 100, 100)
        boxes = [BoundingBox(0, 0, 50, 50, image_shape=image.shape)]
        processors = [MockFooStatsProcessor]

        result = _process_single(0, image, boxes, processors)

        assert len(result.results) == 1
        assert result.object_count == 1

    def test_process_single_invalid_box(self):
        """Test processing with invalid bounding box."""
        image = np.random.rand(3, 50, 50)
        boxes = [BoundingBox(-1, -1, 0, 0, image_shape=image.shape)]
        processors = [MockFooStatsProcessor]

        result = _process_single(0, image, boxes, processors)  # type: ignore

        assert result.invalid_box_count == 1
        assert len(result.warnings_list) == 1


class TestProcessMain:
    def test_process_single_processor(self):
        """Test main process function with single processor."""
        images = [np.random.rand(3, 50, 50)]

        with patch("dataeval.core._processor.get_max_processes", return_value=1):
            result = process(images, None, MockFooStatsProcessor)

        assert "foo" in result
        assert result["image_count"] == 1

    def test_process_multiple_processors(self):
        """Test main process function with multiple processors."""
        images = [np.random.rand(3, 50, 50)]
        processors = [MockFooStatsProcessor, MockBarStatsProcessor]

        with patch("dataeval.core._processor.get_max_processes", return_value=1):
            result = process(images, None, processors)

        assert "foo" in result
        assert "bar" in result
        assert result["image_count"] == 1

    def test_process_with_boxes(self):
        """Test process function with bounding boxes."""
        images = [np.random.rand(3, 100, 100)]
        boxes = [[(10, 10, 50, 50)]]

        with patch("dataeval.core._processor.get_max_processes", return_value=1):
            result = process(images, boxes, MockFooStatsProcessor)

        assert "object_count" in result
        assert result["object_count"][0] == 1

    @patch("dataeval.core._processor.warnings.warn")
    def test_process_with_warnings(self, mock_warn):
        """Test process function with warning generation."""
        images = [np.random.rand(3, 50, 50)]

        # Mock invalid box that generates warning
        with patch("dataeval.core._processor._process_single") as mock_process:
            mock_result = ProcessorOutput(
                results=[ProcessorResult([], {})], object_count=0, invalid_box_count=1, warnings_list=["Test warning"]
            )
            mock_process.return_value = mock_result

            with patch("dataeval.core._processor.get_max_processes", return_value=1):
                process(images, None, MockFooStatsProcessor)

        mock_warn.assert_called_with("Test warning", UserWarning)

    def test_process_empty_images(self):
        """Test process function with empty image list."""
        images = []

        with patch("dataeval.core._processor.get_max_processes", return_value=1):
            result = process(images, None, MockFooStatsProcessor)

        assert result["image_count"] == 0

    def test_process_progress_callback(self):
        """Test main process function with progress callback."""
        images = [np.random.rand(3, 50, 50)]

        callback = MagicMock()

        with patch("dataeval.core._processor.get_max_processes", return_value=1):
            process(images, None, MockFooStatsProcessor, progress_callback=callback)

        assert callback.called
