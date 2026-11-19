"""A detector's predictions for Feasibility are checked before any pass over the data, each failure naming the fix."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from dataeval.performance._feasibility_detection import ImageDetections, check_detections


@dataclass
class _Det:
    boxes: Any
    labels: Any
    scores: Any


def _one(boxes=((0.0, 0.0, 10.0, 10.0),), labels=(1,), scores=(0.9,)) -> _Det:
    return _Det(np.asarray(boxes, dtype=np.float64), np.asarray(labels), np.asarray(scores))


@pytest.mark.required
class TestCheckDetections:
    def test_returns_float64_boxes_integer_labels_and_float64_scores(self):
        out = check_detections([_one(), _Det(np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0))], 2, 3)
        assert all(isinstance(image, ImageDetections) for image in out)
        assert out[0].boxes.dtype == np.float64
        assert out[0].boxes.shape == (1, 4)
        assert out[0].labels.dtype == np.intp
        assert out[0].scores.dtype == np.float64
        assert out[1].boxes.shape == (0, 4)

    def test_scores_per_class_give_each_detection_its_own_label_column(self):
        scores = np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]])
        out = check_detections([_one(boxes=[[0, 0, 1, 1], [2, 2, 3, 3]], labels=[1, 0], scores=scores)], 1, 3)
        np.testing.assert_array_equal(out[0].scores, [0.7, 0.5])

    def test_whole_float_labels_are_accepted(self):
        """Ultralytics gives classes as floats of whole values."""
        out = check_detections([_one(labels=np.array([2.0]))], 1, 3)
        np.testing.assert_array_equal(out[0].labels, [2])

    def test_an_empty_one_dimensional_box_array_is_no_boxes(self):
        out = check_detections([_Det(np.zeros(0), np.zeros(0, dtype=int), np.zeros(0))], 1, 3)
        assert out[0].boxes.shape == (0, 4)

    def test_scores_keep_float64_precision(self):
        """float32 would make these two scores one tie."""
        out = check_detections(
            [_one(boxes=[[0, 0, 1, 1], [2, 2, 3, 3]], labels=[0, 0], scores=[0.1, 0.1 + 1e-12])], 1, 3
        )
        assert out[0].scores[0] != out[0].scores[1]

    @pytest.mark.parametrize(
        ("predictions", "match"),
        [
            ([_one()], r"one detections object per image \(2\), got 1"),
            ([_one(), object()], "predictions\\[1\\] has no 'boxes'"),
            ([_one(), _one(labels=[1, 2])], r"predictions\[1\]: 2 labels for 1 boxes"),
            ([_one(), _one(scores=[0.9, 0.8])], r"predictions\[1\]: scores of shape \(2,\) for 1 boxes"),
            ([_one(), _one(scores=np.ones((1, 4)))], r"predictions\[1\]: scores of shape \(1, 4\) .*\(1, 3\)"),
            ([_one(), _one(scores=[np.nan])], r"predictions\[1\]: scores must be finite"),
            ([_one(), _one(boxes=[[0, 0, np.inf, 1]])], r"^predictions\[1\]: box 0 .*not finite"),
            ([_one(), _one(boxes=[[5, 0, 1, 1]])], r"^predictions\[1\]: box 0 .*positive width and height"),
            ([_one(), _one(labels=[1.5])], "labels"),
            ([_one(), _one(labels=[True])], r"predictions\[1\]: labels are booleans"),
            ([_one(), _one(labels=[3])], r"predictions\[1\]: label 3 is outside the declared classes \[0, 3\)"),
            ([_one(), _one(labels=[-1])], r"predictions\[1\]: label -1 is outside"),
        ],
    )
    def test_refuses(self, predictions, match):
        with pytest.raises(ValueError, match=match):
            check_detections(predictions, 2, 3)

    def test_a_single_array_is_not_a_sequence_of_detections(self):
        with pytest.raises(ValueError, match="one detections object per image"):
            check_detections(np.zeros((2, 3)), 2, 3)  # pyright: ignore[reportArgumentType]
