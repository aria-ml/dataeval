from unittest.mock import MagicMock

import numpy as np
import pytest

# from dataeval.typing import ObjectDetectionTarget, SegmentationTarget
from dataeval.utils.data._selection import Select
from dataeval.utils.data.selections._classfilter import ClassFilter
from dataeval.utils.data.selections._indices import Indices
from dataeval.utils.data.selections._limit import Limit
from dataeval.utils.data.selections._reverse import Reverse
from dataeval.utils.data.selections._shuffle import Shuffle


class MockDetectionTarget:
    def __init__(self, boxes, labels, scores=None):
        self.boxes = boxes
        self.labels = labels
        self.scores = scores


class MockSegmentationTarget:
    def __init__(self, mask, labels, scores=None):
        self.mask = mask
        self.labels = labels
        self.scores = scores


def one_hot(label: int):
    oh = np.zeros(3)
    oh[label] = 1
    return oh


@pytest.fixture(scope="module")
def mock_dataset():
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 10
    mock_dataset.__getitem__.side_effect = lambda idx: (idx, one_hot(idx % 3), {"id": idx})
    return mock_dataset



@pytest.fixture
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
                scores=np.array([0.9, 0.8, 0.7])
            ),
            {"id": 0, "bbox_metadata": ["box1", "box2", "box3"]}
        ),
        # Item 1: Two boxes with labels 0, 0
        (
            "image_1", 
            MockDetectionTarget(
                boxes=np.array([[0, 0, 10, 10], [10, 10, 20, 20]]),
                labels=np.array([0, 0]),
                scores=np.array([0.9, 0.8])
            ),
            {"id": 1, "bbox_metadata": ["box1", "box2"]}
        ),
        # Item 2: Two boxes with labels 1, 2
        (
            "image_2", 
            MockDetectionTarget(
                boxes=np.array([[0, 0, 10, 10], [10, 10, 20, 20]]),
                labels=np.array([1, 2]),
                scores=np.array([0.9, 0.8])
            ),
            {"id": 2, "bbox_metadata": ["box1", "box2"]}
        ),
        # Item 3: Two boxes with labels 2, 2
        (
            "image_3", 
            MockDetectionTarget(
                boxes=np.array([[0, 0, 10, 10], [10, 10, 20, 20]]),
                labels=np.array([2, 2]),
                scores=np.array([0.9, 0.8])
            ),
            {"id": 3, "bbox_metadata": ["box1", "box2"]}
        ),
    ]
    
    mock_dataset.__getitem__.side_effect = lambda idx: items[idx]
    return mock_dataset


@pytest.fixture
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
                scores=np.array([0.9, 0.8, 0.7])
            ),
            {"id": 0, "mask_metadata": ["mask1", "mask2", "mask3"]}
        ),
        # Item 1: Two masks with labels 0, 0
        (
            "image_1", 
            MockSegmentationTarget(
                mask=np.array([[[1, 0], [0, 0]], [[0, 1], [0, 0]]]),
                labels=np.array([0, 0]),
                scores=np.array([0.9, 0.8])
            ),
            {"id": 1, "mask_metadata": ["mask1", "mask2"]}
        ),
        # Item 2: Two masks with labels 1, 2
        (
            "image_2", 
            MockSegmentationTarget(
                mask=np.array([[[1, 0], [0, 0]], [[0, 1], [0, 0]]]),
                labels=np.array([1, 2]),
                scores=np.array([0.9, 0.8])
            ),
            {"id": 2, "mask_metadata": ["mask1", "mask2"]}
        ),
        # Item 3: Two masks with labels 2, 2
        (
            "image_3", 
            MockSegmentationTarget(
                mask=np.array([[[1, 0], [0, 0]], [[0, 1], [0, 0]]]),
                labels=np.array([2, 2]),
                scores=np.array([0.9, 0.8])
            ),
            {"id": 3, "mask_metadata": ["mask1", "mask2"]}
        ),
    ]
    
    mock_dataset.__getitem__.side_effect = lambda idx: items[idx]
    return mock_dataset



@pytest.mark.required
class TestSelectionClasses:
    def test_classfilter_classes_only(self, mock_dataset):
        # Test ClassFilter classes
        class_filter = ClassFilter(classes=(0, 1))
        select = Select(mock_dataset, selections=[class_filter])
        assert len(select) == 7
        counts = {0: 0, 1: 0}
        for _, target, _ in select:
            label = int(np.argmax(target))
            counts[label] = counts[label] + 1
        assert counts == {0: 4, 1: 3}
        assert "ClassFilter(classes=(0, 1), balance=False" in str(select)

    def test_classfilter_balance_only(self, mock_dataset):
        # Test ClassFilter balance
        class_filter = ClassFilter(balance=True)
        select = Select(mock_dataset, selections=[class_filter])
        assert len(select) == 9
        counts = {0: 0, 1: 0, 2: 0}
        for _, target, _ in select:
            label = int(np.argmax(target))
            counts[label] = counts[label] + 1
        assert counts == {0: 3, 1: 3, 2: 3}
        assert "ClassFilter(classes=None, balance=True" in str(select)

    def test_classfilter_classes_and_balance(self, mock_dataset):
        # Test ClassFilter balance
        class_filter = ClassFilter(classes=[0, 1], balance=True)
        select = Select(mock_dataset, selections=[class_filter])
        assert len(select) == 6
        counts = {0: 0, 1: 0}
        for _, target, _ in select:
            label = int(np.argmax(target))
            counts[label] = counts[label] + 1
        assert counts == {0: 3, 1: 3}
        assert "ClassFilter(classes=[0, 1], balance=True" in str(select)

    def test_classfilter_with_unsupported_target(self):
        class MockTarget:
            def __init__(self, label: int):
                self.label = label

        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset.__getitem__.side_effect = lambda idx: (f"data_{idx}", MockTarget(idx), {"id": idx})

        class_filter = ClassFilter(classes=[0])
        with pytest.raises(TypeError):
            Select(mock_dataset, selections=[class_filter])

    def test_classfilter_with_nothing(self, mock_dataset):
        # Test ClassFilter with no params
        class_filter = ClassFilter()
        select = Select(mock_dataset, selections=class_filter)
        assert len(select) == 10

    def test_classfilter_with_limit(self, mock_dataset):
        # Test ClassFilter balance
        class_filter = ClassFilter(classes=[0, 1], balance=True)
        limit = Limit(size=5)
        select = Select(mock_dataset, selections=[limit, class_filter])
        assert len(select) == 5
        assert np.array_equal(select._selection, np.array([0, 1, 3, 4, 6]))
        # counts = {0: 0, 1: 0}
        # for _, target, _ in select:
        #     label = int(np.argmax(target))
        #     counts[label] = counts[label] + 1
        # assert counts == {0: 2, 1: 2}
        assert "ClassFilter(classes=[0, 1], balance=True" in str(select)
        assert "Limit(size=5)" in str(select)

    def test_limit(self, mock_dataset):
        # Test Limit
        limit = Limit(size=5)
        select = Select(mock_dataset, selections=[limit])
        assert len(select) == 5
        assert "Limit(size=5)" in str(select)

    def test_reverse(self, mock_dataset):
        # Test Reverse
        reverse = Reverse()
        select = Select(mock_dataset, selections=[reverse])
        expected_order = list(range(9, -1, -1))
        for i, (data, _, _) in enumerate(select):
            assert data == expected_order[i]
        assert "Reverse()" in str(select)

    def test_shuffle(self, mock_dataset):
        # Test Shuffle
        shuffle = Shuffle(seed=0)
        select = Select(mock_dataset, selections=[shuffle])
        # Since shuffle is random, we just check if the length is correct
        assert len(select) == 10
        # Check if the shuffled order is not the same as the original order
        original_order = [f"data_{i}" for i in range(10)]
        shuffled_order = [data for data, _, _ in select]
        assert original_order != shuffled_order
        assert "Shuffle(seed=0)" in str(select)

    def test_indices(self, mock_dataset):
        indices = Indices([12, 10, 8, 6, 4, 2, 0])
        select = Select(mock_dataset, indices)
        assert len(select) == 5
        assert select._selection == [8, 6, 4, 2, 0]
        assert "Indices(indices=[12, 10, 8, 6, 4, 2, 0])" in str(select)

    def test_indices_repeats(self, mock_dataset):
        indices = Indices([12, 12, 4, 4, 12, 12, 0])
        select = Select(mock_dataset, indices)
        assert len(select) == 3
        assert select._selection == [4, 4, 0]
        assert "Indices(indices=[12, 12, 4, 4, 12, 12, 0])" in str(select)

    def test_indices_with_classfilter(self, mock_dataset):
        class_filter = ClassFilter(classes=[0, 1], balance=False)
        indices = Indices([12, 10, 8, 6, 4, 2, 0])
        select = Select(mock_dataset, [indices, class_filter])
        assert len(select) == 3
        assert select._selection == [6, 4, 0]
        assert "ClassFilter(classes=[0, 1], balance=False" in str(select)
        assert "Indices(indices=[12, 10, 8, 6, 4, 2, 0])" in str(select)

    def test_indices_with_classfilter_layered(self, mock_dataset):
        class_filter = ClassFilter(classes=[0, 1], balance=False)
        select_cf = Select(mock_dataset, class_filter)
        assert len(select_cf) == 7
        indices = Indices([12, 10, 8, 6, 4, 2, 0])
        select = Select(select_cf, indices)
        assert len(select) == 4
        assert select._selection == [6, 4, 2, 0]
        assert "ClassFilter(classes=[0, 1], balance=False" in str(select_cf)
        assert "Indices(indices=[12, 10, 8, 6, 4, 2, 0])" in str(select)


@pytest.mark.required
class TestObjectDetectionSelections:
    def test_detection_filter_classes_only(self, mock_detection_dataset):
        """Test filtering a detection dataset by class."""
        class_filter = ClassFilter(classes=(0, 1))
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

    def test_detection_balancing(self, mock_detection_dataset):
        """Test balancing with detection targets."""
        class_filter = ClassFilter(classes=(0, 1, 2), balance=True)
        select = Select(mock_detection_dataset, selections=[class_filter])
        
        # All items are included, but class counts should be balanced
        # Class 0 appears in 2 images, class 1 in 2 images, class 2 in 3 images
        # With balance=True, we should keep 2 images per class
        
        assert len(select) == 3  # 2 images per class, but with duplicates
        
        # Count actual class occurrences
        class_counts = {0: 0, 1: 0, 2: 0}
        seen_images = set()
        
        for i in range(len(select)):
            image, target, metadata = select[i]
            seen_images.add(image)
            
            for label in target.labels:
                if int(label) in class_counts:
                    class_counts[int(label)] += 1
        
        # We might have duplicates in our balanced set
        # But each class should appear at least twice
        assert class_counts[0] >= 2
        assert class_counts[1] >= 2
        assert class_counts[2] >= 2

    def test_detection_multiclass_scores(self, mock_detection_dataset):
        """Test with multi-class scores instead of simple labels."""
        # Create a dataset with multi-class scores
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 2
        
        items = [
            # Item 0: Two boxes with multi-class scores 
            (
                "image_0", 
                MockDetectionTarget(
                    boxes=np.array([[0, 0, 10, 10], [10, 10, 20, 20]]),
                    labels=np.array([0, 1]),  # Just placeholders, using scores instead
                    scores=np.array([[0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])  # Class 1, Class 2
                ),
                {"id": 0}
            ),
            # Item 1: One box with multi-class scores
            (
                "image_1", 
                MockDetectionTarget(
                    boxes=np.array([[0, 0, 10, 10]]),
                    labels=np.array([0]),  # Placeholder
                    scores=np.array([[0.8, 0.1, 0.1]])  # Class 0
                ),
                {"id": 1}
            ),
        ]
        
        mock_dataset.__getitem__.side_effect = lambda idx: items[idx]
        
        # Filter only class 0
        class_filter = ClassFilter(classes=(0,))
        select = Select(mock_dataset, selections=[class_filter])
        
        # Only item 1 has highest score for class 0
        assert len(select) == 1
        image, target, metadata = select[0]
        assert image == "image_1"

@pytest.mark.required
class TestSegmentationSelections:
    def test_segmentation_filter_classes(self, mock_segmentation_dataset):
        """Test filtering a segmentation dataset by class."""
        class_filter = ClassFilter(classes=(0, 1))
        select = Select(mock_segmentation_dataset, selections=[class_filter])
        
        # Items 0, 1, 2 have classes 0 or 1, so they should be included
        assert len(select) == 3
        
        # Check that transformations are applied correctly
        result0 = select[0]
        image, target, metadata = result0
        
        # Item 0 had 3 masks, but only 2 are in classes (0, 1)
        assert target.mask.shape[0] == 2  # First dimension should be number of masks
        assert len(target.labels) == 2
        assert np.array_equal(target.labels, np.array([0, 1]))
        
        # Check metadata was also properly filtered
        assert len(metadata["mask_metadata"]) == 2
        assert metadata["mask_metadata"] == ["mask1", "mask2"]

    def test_segmentation_no_filtering(self, mock_segmentation_dataset):
        """Test with filter_detections=False to keep all masks in included images."""
        class_filter = ClassFilter(classes=(0,), filter_detections=False)
        select = Select(mock_segmentation_dataset, selections=[class_filter])
        
        # Only items 0 and 1 have class 0
        assert len(select) == 2
        
        # But all masks should remain
        result0 = select[0]
        _, target, _ = result0
        
        # Item 0 had 3 masks, and all should remain
        assert target.mask.shape[0] == 3
        assert len(target.labels) == 3
        assert np.array_equal(target.labels, np.array([0, 1, 2]))

    def test_segmentation_with_balance(self, mock_segmentation_dataset):
        """Test balancing with segmentation targets."""
        class_filter = ClassFilter(classes=(0, 1, 2), balance=True)
        select = Select(mock_segmentation_dataset, selections=[class_filter])
        
        # Each class should have equal representation
        assert len(select) == 3  # 2 images per class, with duplicates
        
        # Count class occurrences
        class_counts = {0: 0, 1: 0, 2: 0}
        
        for i in range(len(select)):
            _, target, _ = select[i]
            for label in target.labels:
                if int(label) in class_counts:
                    class_counts[int(label)] += 1
        
        # Each class should have at least 2 instances
        assert class_counts[0] >= 2
        assert class_counts[1] >= 2
        assert class_counts[2] >= 2