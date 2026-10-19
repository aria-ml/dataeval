"""Tests for the core label_stats function."""

import numpy as np
import pytest

from dataeval.core import label_stats
from dataeval.core._label_stats import (
    ShapeMismatchError,
    _as_int_list,
    _count_labels,
)


@pytest.mark.required
class TestLabelStatsHelpers:
    def test_detection_example(self):
        labels = [0, 0, 1, 1, 2, 0, 1, 2, 4, 4]
        indices = [0, 0, 0, 1, 1, 3, 3, 3, 4, 4]

        counts, per_image, classes_per_image, image_indices = _count_labels(labels, indices, 5)

        assert counts == {0: 3, 1: 3, 2: 2, 4: 2}
        assert per_image == [3, 2, 0, 3, 2]
        assert classes_per_image == [[0, 1], [1, 2], [], [0, 1, 2], [4]]
        assert image_indices == {0: [0, 3], 1: [0, 1, 3], 2: [1, 3], 4: [4]}

    def test_repeated_class_in_one_image_is_counted_but_not_duplicated(self):
        counts, per_image, classes_per_image, image_indices = _count_labels([1, 1, 1], [0, 0, 0], 1)

        assert counts == {1: 3}
        assert per_image == [3]
        assert classes_per_image == [[1]]
        assert image_indices == {1: [0]}

    def test_outputs_are_sorted_regardless_of_input_order(self):
        # Annotations arrive out of image order and out of class order
        labels = [2, 0, 1, 0, 2]
        indices = [3, 3, 0, 3, 0]

        _, _, classes_per_image, image_indices = _count_labels(labels, indices, 4)

        assert classes_per_image == [[1, 2], [], [], [0, 2]]
        assert image_indices == {0: [3], 1: [0], 2: [0, 3]}

    def test_empty_input_keeps_requested_image_count(self):
        counts, per_image, classes_per_image, image_indices = _count_labels([], [], 3)

        assert counts == {}
        assert per_image == [0, 0, 0]
        assert classes_per_image == [[], [], []]
        assert image_indices == {}

    def test_returns_plain_dicts_without_defaultdict_side_effects(self):
        counts, _, _, image_indices = _count_labels([1], [0], 1)

        assert type(counts) is dict
        assert type(image_indices) is dict
        assert counts.get(99, 0) == 0
        assert 99 not in counts  # reading must not insert keys

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="is shorter than"):
            _count_labels([0, 1], [0], 2)

    def test_ndarray_and_generator_inputs_match_list_inputs(self):
        labels = [2, 0, 1, 0]

        from_arrays = _as_int_list(np.array(labels), "test_arrays")
        from_generators = _as_int_list((x for x in labels), "test_generators")
        from_narrow = _as_int_list(np.array(labels, dtype=np.uint8), "test_narrow")
        from_tuples = _as_int_list(tuple(labels), "test_tuples")

        assert from_arrays == labels
        assert from_generators == labels
        assert from_narrow == labels
        assert from_tuples == labels

        assert all(type(lbl) is int for lbl in from_arrays)

    @pytest.mark.parametrize(
        "data",
        [
            [True, False],  # Boolean
            [0.5, 1.0],  # Fractional float
            [0.0, float("nan")],  # NaN
            ["0", "1"],  # String
        ],
    )
    def test_non_integer_labels_raise_type_error(self, data):
        with pytest.raises(TypeError, match="integers"):
            _as_int_list(data, "wrong_type")

    def test_whole_valued_floats_are_accepted_as_integers(self):
        result = _as_int_list(np.array([0.0, 2.0, 1.0]), "floats")

        assert result == [0, 2, 1]
        assert all(type(lbl) is int for lbl in result)

    @pytest.mark.parametrize(
        "data",
        [
            np.zeros((2, 2), dtype=int),  # >1D numpy array
            [[0, 1, 2]],  # Nested List
        ],
    )
    def test_two_dimensional_input_raises(self, data):
        with pytest.raises(ShapeMismatchError, match="one-dimensional"):
            _as_int_list(data, "wrong_shape")


@pytest.mark.required
class TestLabelStats:
    """Test the core label_stats function with simplified inputs."""

    def test_label_stats_basic(self):
        """Test basic label counting with multiple images and classes."""
        # Flat labels: image 0 has [0,0,0,0,0], image 1 has [0,1], image 2 has [0,1,2], image 3 has [0,1,2,3]
        class_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 0, 1, 2, 3]
        item_indices = [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
        index2label = {0: "class_0", 1: "class_1", 2: "class_2", 3: "class_3"}
        stats = label_stats(class_labels, item_indices, index2label, image_count=4)

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
        assert stats["class_proportions"] == {0: 8 / 14, 1: 3 / 14, 2: 2 / 14, 3: 1 / 14}
        assert stats["imbalance_ratio_per_class"] == {0: 1.0, 1: 8 / 3, 2: 4.0, 3: 8.0}

    def test_label_stats_empty_targets(self):
        """Test handling of images with empty targets tracked separately."""
        # image 0 has [0,0,0,0,0], image 1 is empty, image 2 has [0,1,2,3], image 3 is empty
        class_labels = [0, 0, 0, 0, 0, 0, 1, 2, 3]
        item_indices = [0, 0, 0, 0, 0, 2, 2, 2, 2]
        index2label = {0: "class_0", 1: "class_1", 2: "class_2", 3: "class_3"}
        stats = label_stats(class_labels, item_indices, index2label, image_count=4)

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
        assert stats["class_proportions"] == {0: 6 / 9, 1: 1 / 9, 2: 1 / 9, 3: 1 / 9}
        assert stats["imbalance_ratio_per_class"] == {0: 1.0, 1: 6.0, 2: 6.0, 3: 6.0}

    def test_label_stats_no_index2label(self):
        """Test that class names are auto-generated when index2label is None."""
        # image 0 has [0, 1], image 1 has [2]
        class_labels = [0, 1, 2]
        item_indices = [0, 0, 1]
        stats = label_stats(class_labels, item_indices, index2label=None)

        assert stats["index2label"] == {0: "0", 1: "1", 2: "2"}
        assert stats["label_counts_per_class"] == {0: 1, 1: 1, 2: 1}
        assert stats["class_proportions"] == {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
        assert stats["imbalance_ratio_per_class"] == {0: 1.0, 1: 1.0, 2: 1.0}

    def test_label_stats_single_class(self):
        """Test with only a single class."""
        class_labels = [0, 0, 0]
        # 1:1 mapping (no item_indices)
        index2label = {0: "cat"}
        stats = label_stats(class_labels, index2label=index2label)

        assert stats["label_counts_per_class"] == {0: 3}
        assert stats["class_count"] == 1
        assert stats["label_count"] == 3
        assert stats["image_indices_per_class"] == {0: [0, 1, 2]}
        assert stats["image_counts_per_class"] == {0: 3}
        assert stats["label_counts_per_image"] == [1, 1, 1]
        assert stats["image_count"] == 3
        assert stats["index2label"] == {0: "cat"}
        assert stats["empty_image_count"] == 0
        assert stats["class_proportions"] == {0: 1.0}
        assert stats["imbalance_ratio_per_class"] == {0: 1.0}

    def test_label_stats_all_empty(self):
        """Test with all empty labels - tracked separately."""
        class_labels = []  # No labels
        item_indices = []
        index2label = {}
        stats = label_stats(class_labels, item_indices, index2label, image_count=3)

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
        assert stats["class_proportions"] == {}
        assert stats["imbalance_ratio_per_class"] == {}

    def test_label_stats_object_detection(self):
        """Test with object detection-style data (multiple labels per image)."""
        # image 0 has [0,0,1], image 1 has [1,2], image 2 is empty, image 3 has [0,1,2,3]
        class_labels = [0, 0, 1, 1, 2, 0, 1, 2, 3]
        item_indices = [0, 0, 0, 1, 1, 3, 3, 3, 3]
        index2label = {0: "horse", 1: "cow", 2: "sheep", 3: "pig"}
        stats = label_stats(class_labels, item_indices, index2label, image_count=4)

        assert stats["label_counts_per_class"] == {0: 3, 1: 3, 2: 2, 3: 1}
        assert stats["label_counts_per_image"] == [3, 2, 0, 4]
        assert stats["image_counts_per_class"] == {0: 2, 1: 3, 2: 2, 3: 1}
        assert stats["image_indices_per_class"] == {0: [0, 3], 1: [0, 1, 3], 2: [1, 3], 3: [3]}
        assert stats["empty_image_indices"] == [2]
        assert stats["empty_image_count"] == 1
        assert None not in stats["image_indices_per_class"]
        # Check that actual class names are in the mapping
        assert stats["index2label"] == {0: "horse", 1: "cow", 2: "sheep", 3: "pig"}
        assert stats["class_proportions"] == {0: 3 / 9, 1: 3 / 9, 2: 2 / 9, 3: 1 / 9}
        assert stats["imbalance_ratio_per_class"] == {0: 1.0, 1: 1.0, 2: 1.5, 3: 3.0}

    def test_label_stats_image_classification(self):
        """Test with image classification-style data (one label per image, 1:1 mapping)."""
        class_labels = [0, 1, 2, 0]
        # No item_indices means 1:1 mapping
        index2label = {0: "cat", 1: "dog", 2: "bird"}
        stats = label_stats(class_labels, index2label=index2label)

        assert stats["label_counts_per_class"] == {0: 2, 1: 1, 2: 1}
        assert stats["label_counts_per_image"] == [1, 1, 1, 1]
        assert stats["image_counts_per_class"] == {0: 2, 1: 1, 2: 1}
        assert stats["index2label"] == {0: "cat", 1: "dog", 2: "bird"}
        assert stats["empty_image_count"] == 0
        assert stats["class_proportions"] == {0: 0.5, 1: 0.25, 2: 0.25}
        assert stats["imbalance_ratio_per_class"] == {0: 1.0, 1: 2.0, 2: 2.0}

    def test_label_stats_none_sentinel_mixed(self):
        """Test separate tracking with mixed empty and non-empty images."""
        # image 0 has [0,1], image 1 is empty, image 2 has [2,3], image 3 is empty, image 4 has [0]
        class_labels = [0, 1, 2, 3, 0]
        item_indices = [0, 0, 2, 2, 4]
        index2label = {0: "a", 1: "b", 2: "c", 3: "d"}
        stats = label_stats(class_labels, item_indices, index2label, image_count=5)

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
        # image 0 has [-1,0], image 1 has [1], image 2 has [-1,-1], image 3 is empty
        class_labels = [-1, 0, 1, -1, -1]
        item_indices = [0, 0, 1, 2, 2]
        index2label = {-1: "background", 0: "cat", 1: "dog"}
        stats = label_stats(class_labels, item_indices, index2label, image_count=4)

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
        """Test with flat 1D list of ints for image classification (1:1 mapping)."""
        class_labels = [0, 1, 2, 0, 1, 0]
        index2label = {0: "cat", 1: "dog", 2: "bird"}
        stats = label_stats(class_labels, index2label=index2label)

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
        class_labels = [0, 0, 0, 0]
        index2label = {0: "cat"}
        stats = label_stats(class_labels, index2label=index2label)

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
        class_labels = [0, 1, 2, 1, 0]
        stats = label_stats(class_labels, index2label=None)

        assert stats["index2label"] == {0: "0", 1: "1", 2: "2"}
        assert stats["label_counts_per_class"] == {0: 2, 1: 2, 2: 1}
        assert stats["image_count"] == 5
        assert stats["class_count"] == 3
        assert stats["label_count"] == 5

    def test_label_stats_item_indices_length_mismatch(self):
        """Test that mismatched lengths raise an error."""
        class_labels = [0, 1, 2]
        item_indices = [0, 0]  # Wrong length
        with pytest.raises(ValueError, match="item_indices length"):
            label_stats(class_labels, item_indices)

    def test_label_stats_infer_image_count(self):
        """Test that image_count is inferred from max item_indices when not provided."""
        class_labels = [0, 1, 2]
        item_indices = [0, 2, 5]  # Max is 5, so 6 images inferred
        stats = label_stats(class_labels, item_indices)

        assert stats["image_count"] == 6
        assert stats["empty_image_indices"] == [1, 3, 4]
        assert stats["empty_image_count"] == 3

    def test_label_stats_index2label_missing_class(self):
        """Test that classes not in index2label fall back to str(cls)."""
        # Data contains class 3, but index2label only maps 0, 1, 2
        class_labels = [0, 1, 2, 3]
        index2label = {0: "cat", 1: "dog", 2: "bird"}
        stats = label_stats(class_labels, index2label=index2label)

        # Known classes should use their mapped names
        assert stats["index2label"][0] == "cat"
        assert stats["index2label"][1] == "dog"
        assert stats["index2label"][2] == "bird"
        # Unknown class should fall back to str(cls)
        assert stats["index2label"][3] == "3"
        assert stats["class_count"] == 4
        assert stats["label_count"] == 4

    def test_additional_classes(self):
        """Test with object detection-style data with extra classes (multiple labels per image)."""
        # image 0 has [0,0,1], image 1 has [1,2], image 2 is empty, image 3 has [0,1,2], image 4 has [4]
        class_labels = [0, 0, 1, 1, 2, 0, 1, 2, 4, 4]
        item_indices = [0, 0, 0, 1, 1, 3, 3, 3, 4, 4]
        index2label = {0: "horse", 1: "cow", 2: "sheep", 3: "pig", 4: "dog", 5: "cat"}
        stats = label_stats(class_labels, item_indices, index2label, image_count=5)

        assert stats["label_counts_per_class"] == {0: 3, 1: 3, 2: 2, 3: 0, 4: 2, 5: 0}
        assert stats["label_counts_per_image"] == [3, 2, 0, 3, 2]
        assert stats["image_counts_per_class"] == {0: 2, 1: 3, 2: 2, 3: 0, 4: 1, 5: 0}
        assert stats["image_indices_per_class"] == {0: [0, 3], 1: [0, 1, 3], 2: [1, 3], 3: [], 4: [4], 5: []}
        assert stats["empty_image_indices"] == [2]
        assert stats["empty_image_count"] == 1
        # Check that actual class names are in the mapping
        assert stats["index2label"] == {0: "horse", 1: "cow", 2: "sheep", 3: "pig", 4: "dog", 5: "cat"}
        # Check class proportions
        assert stats["class_proportions"] == {0: 0.3, 1: 0.3, 2: 0.2, 3: 0.0, 4: 0.2, 5: 0.0}
        # Check class imbalance
        assert stats["imbalance_ratio_per_class"] == {0: 1.0, 1: 1.0, 2: 1.5, 3: float("inf"), 4: 1.5, 5: float("inf")}
        # Check per-image class imbalance
        assert stats["image_imbalance_ratio_per_class"] == {
            0: 1.5,
            1: 1.0,
            2: 1.5,
            3: float("inf"),
            4: 3.0,
            5: float("inf"),
        }
        # Check multi-label metrics
        assert stats["label_cardinality"] == 2.0  # (2 + 2 + 3 + 1) / 4 non-empty images
        assert stats["label_density"] == pytest.approx(2.0 / 6)
        assert stats["distinct_label_sets"] == 4
        assert stats["label_sets"] == [(0, 1), (0, 1, 2), (1, 2), (4,)]
        assert stats["label_set_counts"] == [1, 1, 1, 1]

        # Check imbalance mixing (SCUMBLE)
        s0 = 1.0 - np.sqrt(1.5) / 1.25
        s1 = s0
        s3 = 1.0 - (2.25 ** (1 / 3)) / (4 / 3)
        s4 = 0.0
        expected_mixing = float(np.mean([s0, s1, s3, s4]))
        assert stats["imbalance_mixing"] == pytest.approx(expected_mixing)
        assert stats["imbalance_mixing_per_class"][0] == pytest.approx(float(np.mean([s0, s3])))
        assert stats["imbalance_mixing_per_class"][1] == pytest.approx(float(np.mean([s0, s1, s3])))
        assert stats["imbalance_mixing_per_class"][2] == pytest.approx(float(np.mean([s1, s3])))
        assert np.isnan(stats["imbalance_mixing_per_class"][3])
        assert stats["imbalance_mixing_per_class"][4] == 0.0
        assert np.isnan(stats["imbalance_mixing_per_class"][5])

    def test_multi_label_metrics_all_empty(self):
        """Test multi-label metrics when all images are empty."""
        stats = label_stats([], [], index2label={0: "a", 1: "b"}, image_count=3)
        assert stats["label_cardinality"] == 0.0
        assert stats["label_density"] == 0.0
        assert stats["distinct_label_sets"] == 0
        assert stats["label_sets"] == []
        assert stats["label_set_counts"] == []
        assert stats["image_imbalance_ratio_per_class"] == {0: 1.0, 1: 1.0}
        assert stats["imbalance_mixing"] == 0.0
        assert all(np.isnan(v) for v in stats["imbalance_mixing_per_class"].values())

    def test_multi_label_metrics_single_label_classification(self):
        """Test multi-label metrics for standard 1:1 image classification."""
        stats = label_stats([0, 1, 2, 0], index2label={0: "a", 1: "b", 2: "c"})
        assert stats["label_cardinality"] == 1.0
        assert stats["label_density"] == pytest.approx(1.0 / 3)
        assert stats["distinct_label_sets"] == 3
        assert stats["label_sets"] == [(0,), (1,), (2,)]
        assert stats["label_set_counts"] == [2, 1, 1]
        assert stats["image_imbalance_ratio_per_class"] == {0: 1.0, 1: 2.0, 2: 2.0}
        assert stats["imbalance_mixing"] == 0.0
        assert stats["imbalance_mixing_per_class"] == {0: 0.0, 1: 0.0, 2: 0.0}
