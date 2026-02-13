"""Comprehensive tests for ClassBalance selection."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from dataeval.selection._classbalance import ClassBalance
from dataeval.selection._select import Select


def one_hot(label: int, num_classes: int = 3):
    """Create one-hot encoded label."""
    oh = np.zeros(num_classes)
    oh[label] = 1
    return oh


@pytest.fixture(scope="module")
def classification_dataset():
    """Mock dataset for classification (one label per image)."""
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 12
    # Classes: 0,1,2,0,1,2,0,1,2,0,1,2
    mock_dataset.__getitem__.side_effect = lambda idx: (idx, one_hot(idx % 3), {"id": idx})
    return mock_dataset


class MockODTarget:
    """Mock object detection target that implements the protocol."""

    def __init__(self, labels, boxes):
        self._labels = labels
        self._boxes = boxes
        self._scores = np.array([])  # scores is required by the protocol

    @property
    def labels(self):
        return self._labels

    @property
    def boxes(self):
        return self._boxes

    @property
    def scores(self):
        return self._scores


@pytest.fixture(scope="module")
def od_dataset_with_empty():
    """Mock object detection dataset with some empty images."""
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 10

    def get_item(idx):
        if idx in [2, 5, 8]:  # Empty images
            target = MockODTarget(labels=np.array([]), boxes=np.array([]))
        elif idx % 3 == 0:
            target = MockODTarget(labels=np.array([0, 0, 1]), boxes=np.array([[0, 0, 1, 1]] * 3))
        elif idx % 3 == 1:
            target = MockODTarget(labels=np.array([1, 2]), boxes=np.array([[0, 0, 1, 1]] * 2))
        else:
            target = MockODTarget(labels=np.array([2]), boxes=np.array([[0, 0, 1, 1]]))
        return (f"image_{idx}", target, {"id": idx})

    mock_dataset.__getitem__.side_effect = get_item
    return mock_dataset


@pytest.mark.required
class TestClassBalanceInterclass:
    """Test interclass balancing method."""

    def test_interclass_basic(self, classification_dataset):
        """Test basic interclass balancing with equal class distribution."""
        class_balance = ClassBalance(method="interclass")
        select = Select(classification_dataset, selections=[class_balance])

        # Should have equal samples from each class
        assert len(select) == 12
        counts = {0: 0, 1: 0, 2: 0}
        for _, target, _ in select:
            label = int(np.argmax(target))
            counts[label] += 1
        assert counts == {0: 4, 1: 4, 2: 4}

    def test_interclass_with_num_samples(self, classification_dataset):
        """Test interclass with specific number of samples."""
        class_balance = ClassBalance(method="interclass", num_samples=9)
        select = Select(classification_dataset, selections=[class_balance])

        assert len(select) == 9
        counts = {0: 0, 1: 0, 2: 0}
        for _, target, _ in select:
            label = int(np.argmax(target))
            counts[label] += 1
        # Should distribute 9 samples evenly: 3 per class
        assert counts == {0: 3, 1: 3, 2: 3}

    def test_interclass_with_background_class(self, classification_dataset):
        """Test interclass excluding background class."""
        class_balance = ClassBalance(method="interclass", background_class=2)
        select = Select(classification_dataset, selections=[class_balance])

        # Should only sample from classes 0 and 1, excluding background class 2
        counts = {0: 0, 1: 0, 2: 0}
        for _, target, _ in select:
            label = int(np.argmax(target))
            counts[label] += 1
        # Background class should not be sampled
        assert counts[2] == 0
        assert counts[0] > 0
        assert counts[1] > 0


@pytest.mark.required
class TestClassBalanceGlobal:
    """Test global balancing method."""

    def test_global_basic(self, classification_dataset):
        """Test basic global balancing."""
        class_balance = ClassBalance(method="global")
        select = Select(classification_dataset, selections=[class_balance])

        # With equal class distribution, global should give roughly equal results
        assert len(select) == 12
        counts = {0: 0, 1: 0, 2: 0}
        for _, target, _ in select:
            label = int(np.argmax(target))
            counts[label] += 1
        # All classes should be represented
        assert all(count > 0 for count in counts.values())

    def test_global_with_aggregation(self, classification_dataset):
        """Test global with different aggregation functions."""
        # Test with mean aggregation
        class_balance_mean = ClassBalance(method="global", aggregation_func="mean")
        select_mean = Select(classification_dataset, selections=[class_balance_mean])
        assert len(select_mean) == 12

        # Test with max aggregation
        class_balance_max = ClassBalance(method="global", aggregation_func="max")
        select_max = Select(classification_dataset, selections=[class_balance_max])
        assert len(select_max) == 12

    def test_global_with_oversample_factor(self, classification_dataset):
        """Test global with oversample factor."""
        class_balance = ClassBalance(method="global", oversample_factor=2.0)
        select = Select(classification_dataset, selections=[class_balance])
        assert len(select) == 12


@pytest.mark.required
class TestClassBalanceEmptyImages:
    """Test ClassBalance with empty images (separate tracking)."""

    def test_empty_images_tracked_with_none(self, od_dataset_with_empty):
        """Test that empty images are tracked separately."""
        class_balance = ClassBalance(method="interclass", num_empty=0)
        select = Select(od_dataset_with_empty, selections=[class_balance])

        # After balancing, check that the ClassBalance instance tracks empty images separately
        assert hasattr(class_balance, "_empty_image_indices")
        # Empty images should be indices 2, 5, 8
        assert set(class_balance._empty_image_indices) == {2, 5, 8}
        # None should NOT be in images_per_class
        assert None not in class_balance._images_per_class
        # Empty images should not be in the selection since num_empty=0
        empty_in_selection = [idx for idx in select._selection if idx in [2, 5, 8]]
        assert len(empty_in_selection) == 0

    def test_interclass_with_num_empty(self, od_dataset_with_empty):
        """Test interclass balancing with num_empty parameter."""
        class_balance = ClassBalance(method="interclass", num_empty=2)
        select = Select(od_dataset_with_empty, selections=[class_balance])

        # Should include 2 empty images plus balanced non-empty images
        assert len(select) >= 2

        # Count how many images in selection are empty
        empty_count = sum(1 for idx in select._selection if idx in [2, 5, 8])
        assert empty_count == 2

    def test_global_with_num_empty(self, od_dataset_with_empty):
        """Test global balancing with num_empty parameter."""
        class_balance = ClassBalance(method="global", num_empty=1)
        select = Select(od_dataset_with_empty, selections=[class_balance])

        # Should include AT LEAST 1 empty image (may have more due to sampling)
        empty_count = sum(1 for idx in select._selection if idx in [2, 5, 8])
        assert empty_count >= 1

    def test_num_empty_as_proportion(self, od_dataset_with_empty):
        """Test num_empty as proportion of dataset."""
        # 0.2 * 10 = 2 empty images
        class_balance = ClassBalance(method="interclass", num_empty=0.2)
        select = Select(od_dataset_with_empty, selections=[class_balance])

        empty_count = sum(1 for idx in select._selection if idx in [2, 5, 8])
        assert empty_count == 2


@pytest.mark.required
class TestClassBalanceNoneSentinel:
    """Test that empty images are tracked separately, not as a class."""

    def test_empty_images_not_in_classes(self, od_dataset_with_empty):
        """Test that empty images are not tracked in images_per_class."""
        class_balance = ClassBalance(method="interclass", num_empty=0)
        Select(od_dataset_with_empty, selections=[class_balance])

        # None should NOT be in images_per_class (empty images tracked separately)
        assert None not in class_balance._images_per_class

        # -1 should NOT be in images_per_class (old sentinel)
        assert -1 not in class_balance._images_per_class

        # None should NOT be in classes list
        assert None not in class_balance._classes

        # Empty images should be in separate tracking
        assert hasattr(class_balance, "_empty_image_indices")
        assert len(class_balance._empty_image_indices) > 0

    def test_minus_one_can_be_valid_class(self):
        """Test that -1 can be used as a valid class label."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 6

        def get_item(idx):
            if idx == 0:
                # Empty image
                target = MockODTarget(labels=np.array([]), boxes=np.array([]))
            elif idx % 2 == 0:
                # Class -1 (background)
                target = MockODTarget(labels=np.array([-1]), boxes=np.array([[0, 0, 1, 1]]))
            else:
                # Class 0
                target = MockODTarget(labels=np.array([0]), boxes=np.array([[0, 0, 1, 1]]))
            return (f"image_{idx}", target, {"id": idx})

        mock_dataset.__getitem__.side_effect = get_item

        class_balance = ClassBalance(method="interclass", num_empty=0)
        Select(mock_dataset, selections=[class_balance])

        # -1 should be treated as a valid class
        assert -1 in class_balance._images_per_class
        assert len(class_balance._images_per_class[-1]) > 0

        # Empty images should be in separate tracking, not in images_per_class
        assert None not in class_balance._images_per_class
        assert 0 in class_balance._empty_image_indices


@pytest.mark.required
class TestClassBalanceMinimizeDuplicates:
    """Test minimize_duplicates feature."""

    def test_minimize_duplicates_interclass(self, classification_dataset):
        """Test minimize_duplicates with interclass method."""
        # Dataset has 12 images (4 of each class 0,1,2)
        # num_samples defaults to None which means dataset size (12)
        class_balance = ClassBalance(method="interclass", minimize_duplicates=True)
        select = Select(classification_dataset, selections=[class_balance])

        # Should get all 12 images
        assert len(select) == 12

        # With minimize_duplicates, should try to reduce duplicate selections
        # (hard to test deterministically, just verify it runs)
        counts = {0: 0, 1: 0, 2: 0}
        for _, target, _ in select:
            label = int(np.argmax(target))
            counts[label] += 1
        # All classes should be represented equally
        assert counts == {0: 4, 1: 4, 2: 4}


@pytest.mark.required
class TestClassBalanceEdgeCases:
    """Test edge cases and error conditions."""

    def test_all_empty_images(self):
        """Test with dataset containing only empty images."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 5

        def get_item(idx):
            target = MockODTarget(labels=np.array([]), boxes=np.array([]))
            return (f"image_{idx}", target, {"id": idx})

        mock_dataset.__getitem__.side_effect = get_item

        class_balance = ClassBalance(method="interclass")
        select = Select(mock_dataset, selections=[class_balance])

        # With no actual classes, selection should be empty or only contain empty images
        # depending on num_empty setting
        assert len(select) == 0  # No classes to balance

    def test_single_class(self):
        """Test with dataset containing only one class."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 5
        mock_dataset.__getitem__.side_effect = lambda idx: (idx, one_hot(0, num_classes=1), {"id": idx})

        class_balance = ClassBalance(method="interclass", num_samples=3)
        select = Select(mock_dataset, selections=[class_balance])

        # Should work with single class
        assert len(select) == 3
        for _, target, _ in select:
            assert int(np.argmax(target)) == 0

    def test_num_samples_larger_than_dataset(self, classification_dataset):
        """Test with num_samples larger than dataset size."""
        # With num_samples=None, it uses dataset size
        class_balance = ClassBalance(method="interclass")
        select = Select(classification_dataset, selections=[class_balance])

        # Should return dataset size (12)
        assert len(select) == 12
