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
        assert stats["image_count"] == 4
        assert stats["class_names"] == ["class_0", "class_1", "class_2", "class_3"]

    def test_label_stats_empty_targets(self):
        """Test handling of images with empty targets."""
        labels = [[0, 0, 0, 0, 0], [], [0, 1, 2, 3], []]
        index2label = {0: "class_0", 1: "class_1", 2: "class_2", 3: "class_3"}
        stats = label_stats(labels, index2label)

        assert stats["label_counts_per_class"] == {0: 6, 1: 1, 2: 1, 3: 1}
        assert stats["class_count"] == 4
        assert stats["label_count"] == 9
        assert stats["image_indices_per_class"] == {0: [0, 2], 1: [2], 2: [2], 3: [2]}
        assert stats["image_counts_per_class"] == {0: 2, 1: 1, 2: 1, 3: 1}
        assert stats["label_counts_per_image"] == [5, 0, 4, 0]
        assert stats["image_count"] == 4

    def test_label_stats_no_index2label(self):
        """Test that class names are auto-generated when index2label is None."""
        labels = [[0, 1], [2]]
        stats = label_stats(labels, index2label=None)

        assert stats["class_names"] == ["0", "1", "2"]
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
        assert stats["class_names"] == ["cat"]

    def test_label_stats_all_empty(self):
        """Test with all empty labels."""
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
        assert stats["class_names"] == []

    def test_label_stats_object_detection(self):
        """Test with object detection-style data (multiple labels per image)."""
        labels = [[0, 0, 1], [1, 2], [], [0, 1, 2, 3]]
        index2label = {0: "horse", 1: "cow", 2: "sheep", 3: "pig"}
        stats = label_stats(labels, index2label)

        assert stats["label_counts_per_class"] == {0: 3, 1: 3, 2: 2, 3: 1}
        assert stats["label_counts_per_image"] == [3, 2, 0, 4]
        assert stats["image_counts_per_class"] == {0: 2, 1: 3, 2: 2, 3: 1}
        assert stats["class_names"] == ["horse", "cow", "sheep", "pig"]

    def test_label_stats_image_classification(self):
        """Test with image classification-style data (one label per image)."""
        labels = [[0], [1], [2], [0]]
        index2label = {0: "cat", 1: "dog", 2: "bird"}
        stats = label_stats(labels, index2label)

        assert stats["label_counts_per_class"] == {0: 2, 1: 1, 2: 1}
        assert stats["label_counts_per_image"] == [1, 1, 1, 1]
        assert stats["image_counts_per_class"] == {0: 2, 1: 1, 2: 1}
        assert stats["class_names"] == ["cat", "dog", "bird"]
