"""Tests for the core label_stats function."""

import pytest

from dataeval.core import label_stats


@pytest.mark.required
class TestLabelStats:
    """Test the core label_stats function with simplified inputs."""

    def test_label_stats_basic(self):
        """Test basic label counting with multiple images and classes."""
        labels = [[0, 0, 0, 0, 0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
        index2label = {0: "class_0", 1: "class_1", 2: "class_2", 3: "class_3"}
        stats = label_stats(labels, index2label)

        assert stats["label_counts_per_class"] == {0: 8, 1: 3, 2: 2, 3: 1}
        assert stats["class_count"] == 4
        assert stats["label_count"] == 14
        assert stats["image_indices_per_class"] == {0: [0, 1, 2, 3], 1: [1, 2, 3], 2: [2, 3], 3: [3]}
        assert stats["image_counts_per_class"] == {0: 4, 1: 3, 2: 2, 3: 1}
        assert stats["label_counts_per_image"] == [5, 2, 3, 4]
        assert stats["classes_per_image"] == [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
        assert stats["image_count"] == 4
        assert stats["index2label"] == {0: "class_0", 1: "class_1", 2: "class_2", 3: "class_3"}
        assert stats["empty_image_count"] == 0
        assert stats["empty_image_indices"] == []

    def test_label_stats_empty_targets(self):
        """Test handling of images with empty targets tracked separately."""
        labels = [[0, 0, 0, 0, 0], [], [0, 1, 2, 3], []]
        index2label = {0: "class_0", 1: "class_1", 2: "class_2", 3: "class_3"}
        stats = label_stats(labels, index2label)

        assert stats["label_counts_per_class"] == {0: 6, 1: 1, 2: 1, 3: 1}
        assert stats["class_count"] == 4
        assert stats["label_count"] == 9
        assert stats["image_indices_per_class"] == {0: [0, 2], 1: [2], 2: [2], 3: [2]}
        assert stats["image_counts_per_class"] == {0: 2, 1: 1, 2: 1, 3: 1}
        assert stats["label_counts_per_image"] == [5, 0, 4, 0]
        assert stats["classes_per_image"] == [[0], [], [0, 1, 2, 3], []]  # Empty images have empty lists
        assert stats["image_count"] == 4
        assert stats["empty_image_indices"] == [1, 3]
        assert stats["empty_image_count"] == 2
        assert None not in stats["image_indices_per_class"]

    def test_label_stats_no_index2label(self):
        """Test that class names are auto-generated when index2label is None."""
        labels = [[0, 1], [2]]
        stats = label_stats(labels, index2label=None)

        assert stats["index2label"] == {0: "0", 1: "1", 2: "2"}
        assert stats["label_counts_per_class"] == {0: 1, 1: 1, 2: 1}

    def test_label_stats_single_class(self):
        """Test with only a single class."""
        labels = [[0], [0], [0]]
        index2label = {0: "cat"}
        stats = label_stats(labels, index2label)

        assert stats["label_counts_per_class"] == {0: 3}
        assert stats["class_count"] == 1
        assert stats["label_count"] == 3
        assert stats["image_indices_per_class"] == {0: [0, 1, 2]}
        assert stats["image_counts_per_class"] == {0: 3}
        assert stats["label_counts_per_image"] == [1, 1, 1]
        assert stats["image_count"] == 3
        assert stats["index2label"] == {0: "cat"}
        assert stats["empty_image_count"] == 0

    def test_label_stats_all_empty(self):
        """Test with all empty labels - tracked separately."""
        labels = [[], [], []]
        index2label = {}
        stats = label_stats(labels, index2label)

        assert stats["label_counts_per_class"] == {}
        assert stats["class_count"] == 0
        assert stats["label_count"] == 0
        assert stats["image_indices_per_class"] == {}
        assert stats["image_counts_per_class"] == {}
        assert stats["label_counts_per_image"] == [0, 0, 0]
        assert stats["image_count"] == 3
        assert stats["empty_image_indices"] == [0, 1, 2]
        assert stats["empty_image_count"] == 3
        assert stats["index2label"] == {}
        assert None not in stats["image_indices_per_class"]

    def test_label_stats_object_detection(self):
        """Test with object detection-style data (multiple labels per image)."""
        labels = [[0, 0, 1], [1, 2], [], [0, 1, 2, 3]]
        index2label = {0: "horse", 1: "cow", 2: "sheep", 3: "pig"}
        stats = label_stats(labels, index2label)

        assert stats["label_counts_per_class"] == {0: 3, 1: 3, 2: 2, 3: 1}
        assert stats["label_counts_per_image"] == [3, 2, 0, 4]
        assert stats["image_counts_per_class"] == {0: 2, 1: 3, 2: 2, 3: 1}
        assert stats["image_indices_per_class"] == {0: [0, 3], 1: [0, 1, 3], 2: [1, 3], 3: [3]}
        assert stats["empty_image_indices"] == [2]
        assert stats["empty_image_count"] == 1
        assert None not in stats["image_indices_per_class"]
        # Check that actual class names are in the mapping
        assert stats["index2label"] == {0: "horse", 1: "cow", 2: "sheep", 3: "pig"}

    def test_label_stats_image_classification(self):
        """Test with image classification-style data (one label per image)."""
        labels = [[0], [1], [2], [0]]
        index2label = {0: "cat", 1: "dog", 2: "bird"}
        stats = label_stats(labels, index2label)

        assert stats["label_counts_per_class"] == {0: 2, 1: 1, 2: 1}
        assert stats["label_counts_per_image"] == [1, 1, 1, 1]
        assert stats["image_counts_per_class"] == {0: 2, 1: 1, 2: 1}
        assert stats["index2label"] == {0: "cat", 1: "dog", 2: "bird"}
        assert stats["empty_image_count"] == 0

    def test_label_stats_none_sentinel_mixed(self):
        """Test separate tracking with mixed empty and non-empty images."""
        labels = [[0, 1], [], [2, 3], [], [0]]
        index2label = {0: "a", 1: "b", 2: "c", 3: "d"}
        stats = label_stats(labels, index2label)

        # Empty images should be tracked separately
        assert stats["empty_image_indices"] == [1, 3]
        assert stats["empty_image_count"] == 2
        assert None not in stats["image_indices_per_class"]

        # Non-empty images should have normal class tracking
        assert stats["image_indices_per_class"][0] == [0, 4]
        assert stats["image_indices_per_class"][1] == [0]
        assert stats["image_indices_per_class"][2] == [2]
        assert stats["image_indices_per_class"][3] == [2]

        # Label counts should not include None
        assert None not in stats["label_counts_per_class"]
        assert stats["label_count"] == 5  # Total labels excluding empty images
        assert stats["class_count"] == 4  # Should not count empty images

    def test_label_stats_negative_one_as_valid_class(self):
        """Test that -1 can be used as a valid class label."""
        labels = [[-1, 0], [1], [-1, -1], []]
        index2label = {-1: "background", 0: "cat", 1: "dog"}
        stats = label_stats(labels, index2label)

        # -1 should be treated as a normal class
        assert stats["label_counts_per_class"][-1] == 3
        assert stats["image_indices_per_class"][-1] == [0, 2]
        assert stats["image_counts_per_class"][-1] == 2
        assert stats["index2label"][-1] == "background"

        # Empty image should be tracked separately, not as -1 or None
        assert stats["empty_image_indices"] == [3]
        assert stats["empty_image_count"] == 1
        assert None not in stats["image_indices_per_class"]

    def test_label_stats_flat_list_classification(self):
        """Test with flat 1D list of ints for image classification."""
        labels = [0, 1, 2, 0, 1, 0]
        index2label = {0: "cat", 1: "dog", 2: "bird"}
        stats = label_stats(labels, index2label)

        # Each int should be treated as a single label for one image
        assert stats["label_counts_per_class"] == {0: 3, 1: 2, 2: 1}
        assert stats["label_counts_per_image"] == [1, 1, 1, 1, 1, 1]
        assert stats["image_counts_per_class"] == {0: 3, 1: 2, 2: 1}
        assert stats["image_indices_per_class"] == {0: [0, 3, 5], 1: [1, 4], 2: [2]}
        assert stats["classes_per_image"] == [[0], [1], [2], [0], [1], [0]]
        assert stats["image_count"] == 6
        assert stats["class_count"] == 3
        assert stats["label_count"] == 6
        assert stats["index2label"] == {0: "cat", 1: "dog", 2: "bird"}
        assert stats["empty_image_count"] == 0
        assert stats["empty_image_indices"] == []

    def test_label_stats_flat_list_single_class(self):
        """Test with flat 1D list containing only one class."""
        labels = [0, 0, 0, 0]
        index2label = {0: "cat"}
        stats = label_stats(labels, index2label)

        assert stats["label_counts_per_class"] == {0: 4}
        assert stats["label_counts_per_image"] == [1, 1, 1, 1]
        assert stats["image_counts_per_class"] == {0: 4}
        assert stats["image_indices_per_class"] == {0: [0, 1, 2, 3]}
        assert stats["classes_per_image"] == [[0], [0], [0], [0]]
        assert stats["image_count"] == 4
        assert stats["class_count"] == 1
        assert stats["label_count"] == 4
        assert stats["empty_image_count"] == 0

    def test_label_stats_flat_list_no_index2label(self):
        """Test with flat 1D list and auto-generated index2label."""
        labels = [0, 1, 2, 1, 0]
        stats = label_stats(labels, index2label=None)

        assert stats["index2label"] == {0: "0", 1: "1", 2: "2"}
        assert stats["label_counts_per_class"] == {0: 2, 1: 2, 2: 1}
        assert stats["image_count"] == 5
        assert stats["class_count"] == 3
        assert stats["label_count"] == 5
