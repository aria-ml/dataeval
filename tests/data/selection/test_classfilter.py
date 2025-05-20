from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from dataeval.data._metadata import Metadata
from dataeval.data._selection import Select
from dataeval.data.selections._classfilter import ClassFilter, _try_mask_object


@pytest.mark.required
class TestTryMaskObject:
    mask = np.array([True, False, False, True])

    @pytest.mark.parametrize(
        "obj, expected",
        [
            ("test", "test"),
            (np.array([1, 2, 3, 4]), np.array([1, 4])),
            ([1, 2, 3, 4], [1, 4]),
        ],
    )
    def test_nonmaskable(self, obj, expected):
        result = _try_mask_object(obj, self.mask)
        assert len(result) == len(expected)
        assert all(result[i] == expected[i] for i in range(len(result)))


@dataclass
class MockDetectionTarget:
    boxes: NDArray[np.float32]
    labels: NDArray[np.intp]
    scores: NDArray[np.float32]


@dataclass
class MockSegmentationTarget:
    mask: NDArray[np.intp]
    labels: NDArray[np.intp]
    scores: NDArray[np.float32]


@pytest.fixture(scope="module")
def mock_detection_dataset():
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 4

    # Create mock data items:
    # Item 0: contains classes [0, 1, 2]
    # Item 1: contains classes [0, 0]
    # Item 2: contains classes [1, 2]
    # Item 3: contains classes [2, 2]

    items = [
        # Item 0: Three boxes with labels 0, 1, 2
        (
            "image_0",
            MockDetectionTarget(
                boxes=np.array([[0, 0, 10, 10], [10, 10, 20, 20], [30, 30, 40, 40]]),
                labels=np.array([0, 1, 2]),
                scores=np.array([0.9, 0.8, 0.7]),
            ),
            {"id": 0, "bbox_metadata": ["box1", "box2", "box3"], "a": {"b": {"nested": [0, 0, 0]}}},
        ),
        # Item 1: Two boxes with labels 0, 0
        (
            "image_1",
            MockDetectionTarget(
                boxes=np.array([[0, 0, 10, 10], [10, 10, 20, 20]]),
                labels=np.array([0, 0]),
                scores=np.array([0.9, 0.8]),
            ),
            {"id": 1, "bbox_metadata": ["box1", "box2"], "a": {"b": {"nested": [1, 1]}}},
        ),
        # Item 2: Two boxes with labels 1, 2
        (
            "image_2",
            MockDetectionTarget(
                boxes=np.array([[0, 0, 10, 10], [10, 10, 20, 20]]),
                labels=np.array([1, 2]),
                scores=np.array([0.9, 0.8]),
            ),
            {"id": 2, "bbox_metadata": ["box1", "box2"], "a": {"b": {"nested": [2, 2]}}},
        ),
        # Item 3: Two boxes with labels 2, 2
        (
            "image_3",
            MockDetectionTarget(
                boxes=np.array([[0, 0, 10, 10], [10, 10, 20, 20]]),
                labels=np.array([2, 2]),
                scores=np.array([0.9, 0.8]),
            ),
            {"id": 3, "bbox_metadata": ["box1", "box2"], "a": {"b": {"nested": [3, 3]}}},
        ),
    ]

    mock_dataset.__getitem__.side_effect = lambda idx: items[idx]
    return mock_dataset


@pytest.fixture(scope="module")
def mock_segmentation_dataset():
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 4

    # Create mock data items similar to detection but with masks
    items = [
        # Item 0: Three masks with labels 0, 1, 2
        (
            "image_0",
            MockSegmentationTarget(
                mask=np.array([[[1, 0, 0], [0, 0, 0]], [[0, 1, 0], [0, 0, 0]], [[0, 0, 1], [0, 0, 0]]]),
                labels=np.array([0, 1, 2]),
                scores=np.array([0.9, 0.8, 0.7]),
            ),
            {"id": 0, "mask_metadata": ["mask1", "mask2", "mask3"]},
        ),
        # Item 1: Two masks with labels 0, 0
        (
            "image_1",
            MockSegmentationTarget(
                mask=np.array([[[1, 0], [0, 0]], [[0, 1], [0, 0]]]),
                labels=np.array([0, 0]),
                scores=np.array([0.9, 0.8]),
            ),
            {"id": 1, "mask_metadata": ["mask1", "mask2"]},
        ),
        # Item 2: Two masks with labels 1, 2
        (
            "image_2",
            MockSegmentationTarget(
                mask=np.array([[[1, 0], [0, 0]], [[0, 1], [0, 0]]]),
                labels=np.array([1, 2]),
                scores=np.array([0.9, 0.8]),
            ),
            {"id": 2, "mask_metadata": ["mask1", "mask2"]},
        ),
        # Item 3: Two masks with labels 2, 2
        (
            "image_3",
            MockSegmentationTarget(
                mask=np.array([[[1, 0], [0, 0]], [[0, 1], [0, 0]]]),
                labels=np.array([2, 2]),
                scores=np.array([0.9, 0.8]),
            ),
            {"id": 3, "mask_metadata": ["mask1", "mask2"]},
        ),
    ]

    mock_dataset.__getitem__.side_effect = lambda idx: items[idx]
    return mock_dataset


@pytest.mark.required
class TestObjectDetectionSelections:
    def test_detection_filter_classes_only(self, mock_detection_dataset):
        """Test filtering a detection dataset by class."""
        class_filter = ClassFilter(classes=(0, 1), filter_detections=True)
        select = Select(mock_detection_dataset, selections=[class_filter])

        # Items 0, 1, 2 have classes 0 or 1, so they should be included
        assert len(select) == 3

        # Check that transformations are applied correctly
        result0 = select[0]
        image, target, metadata = result0

        # Item 0 had 3 boxes, but only 2 are in classes (0, 1)
        assert len(target.boxes) == 2
        assert len(target.labels) == 2
        assert np.array_equal(target.labels, np.array([0, 1]))

        # Check metadata was also properly filtered
        assert len(metadata["bbox_metadata"]) == 2
        assert metadata["bbox_metadata"] == ["box1", "box2"]

        # Check that item 1 is not filtered (both boxes are class 0)
        result1 = select[1]
        _, target1, metadata1 = result1
        assert len(target1.boxes) == 2
        assert len(metadata1["bbox_metadata"]) == 2

    def test_detection_filter_no_filtering(self, mock_detection_dataset):
        """Test with filter_detections=False to keep all detections in included images."""
        class_filter = ClassFilter(classes=(0,), filter_detections=False)
        select = Select(mock_detection_dataset, selections=[class_filter])

        # Only items 0 and 1 have class 0
        assert len(select) == 2

        # But all detections should remain
        result0 = select[0]
        _, target, _ = result0

        # Item 0 had 3 boxes, and all should remain
        assert len(target.boxes) == 3
        assert len(target.labels) == 3
        assert np.array_equal(target.labels, np.array([0, 1, 2]))

    def test_detection_filter_index_alignment(self, mock_detection_dataset):
        """Test that filtering preserves correct index alignment between images, targets, and metadata."""
        # Only select class 2, which appears in items 0, 2, and 3
        class_filter = ClassFilter(classes=(2,), filter_detections=True)
        select = Select(mock_detection_dataset, selections=[class_filter])

        # Check that only items with class 2 are included
        assert len(select) == 3

        # For each of the selected items, verify that:
        # 1. Only detections of class 2 remain
        # 2. The correct image is returned
        # 3. The metadata is properly aligned and filtered

        # First item should be the original item 0
        result0 = select[0]
        image0, target0, metadata0 = result0

        # Verify it's the correct image
        assert image0 == "image_0"

        # Verify only class 2 detections remain
        assert len(target0.boxes) == 1
        assert np.array_equal(target0.labels, np.array([2]))
        assert np.array_equal(target0.scores, np.array([0.7]))

        # Verify metadata is aligned (should only have the third box metadata)
        assert metadata0["id"] == 0
        assert len(metadata0["bbox_metadata"]) == 1
        assert metadata0["bbox_metadata"] == ["box3"]
        assert metadata0["a"]["b"]["nested"] == [0]

        # Second item should be the original item 2
        result1 = select[1]
        image1, target1, metadata1 = result1

        # Verify it's the correct image
        assert image1 == "image_2"

        # Verify only class 2 detections remain
        assert len(target1.boxes) == 1
        assert np.array_equal(target1.labels, np.array([2]))
        assert np.array_equal(target1.scores, np.array([0.8]))

        # Verify metadata is aligned
        assert metadata1["id"] == 2
        assert len(metadata1["bbox_metadata"]) == 1
        assert metadata1["bbox_metadata"] == ["box2"]

        # Third item should be the original item 3
        result2 = select[2]
        image2, target2, metadata2 = result2

        # Verify it's the correct image
        assert image2 == "image_3"

        # Verify both boxes remain (both are class 2)
        assert len(target2.boxes) == 2
        assert np.array_equal(target2.labels, np.array([2, 2]))
        assert np.array_equal(target2.scores, np.array([0.9, 0.8]))

        # Verify metadata is aligned
        assert metadata2["id"] == 3
        assert len(metadata2["bbox_metadata"]) == 2
        assert metadata2["bbox_metadata"] == ["box1", "box2"]

        # Verify nested metadata is processed by Metadata class
        md = Metadata(select)
        assert md.dataframe["nested"].to_list() == [0, 2, 3, 3]
