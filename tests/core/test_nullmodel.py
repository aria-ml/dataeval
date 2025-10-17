import numpy as np
import pytest

from dataeval.core._nullmodel import (
    _calculate_accuracy,
    _calculate_fpr,
    _calculate_multiclass_accuracy,
    _calculate_precision,
    _calculate_recall,
    _reduce_macro,
    _reduce_micro,
    _to_confusion_matrix,
)


@pytest.mark.required
class TestMetricEstimators:
    @pytest.mark.parametrize(
        "class_probs, model_probs, expected_accuracy",
        [
            ([0, 0, 0], [0, 0, 0], 0),
            ([1, 0, 0], [0, 1, 0], 0),
            ([1, 0, 0], [1, 0, 0], 1),
            ([0.25, 0.25, 0.5], [0.5, 0.25, 0.25], 0.3125),
            ([0.25, 0, 0.75], [0.25, 0, 0.75], 0.625),
        ],
    )
    def test_calculate_multiclass_accuracy(self, class_probs, model_probs, expected_accuracy):
        """Tests multiclass accuracy estimator"""
        assert _calculate_multiclass_accuracy(class_probs, model_probs) == expected_accuracy

    @pytest.mark.parametrize(
        "counts, expected_accuracy",
        [
            ([1, 0, 1, 0], 1),
            ([0, 0, 0, 20], 0),
            ([1, 1, 1, 1], 0.5),
            ([5, 0, 0, 0], 1),
            ([20, 26, 2, 40], 0.25),
        ],
    )
    def test_calculate_accuracy(self, counts, expected_accuracy):
        """Tests binary accuracy estimator"""
        assert _calculate_accuracy(counts) == expected_accuracy

    @pytest.mark.parametrize(
        "counts, expected_precision",
        [
            ([0, 0, 0, 1], 0),
            ([0, 0, 0, 0], 1),
            ([1, 0, 40, 0], 1),
            ([0, 1, 0, 0], 0),
            ([1, 3, 1, 100], 0.25),
            ([1, 3, 0, 0], 0.25),
        ],
    )
    def test_calculate_precision(self, counts, expected_precision):
        """Tests binary precision estimator"""
        assert _calculate_precision(counts) == expected_precision

    @pytest.mark.parametrize(
        "counts, expected_false_positive_rate",
        [
            ([1, 0, 0, 10], 0),
            ([0, 1, 0, 20], 1),
            ([1, 1, 1, 1], 0.5),
            ([5, 0, 0, 0], 0),
            ([20, 6, 2, 40], 0.75),
        ],
    )
    def test_calculate_false_positive_rate(self, counts, expected_false_positive_rate):
        """Tests binary false positive rate estimator"""
        assert _calculate_fpr(counts) == expected_false_positive_rate

    @pytest.mark.parametrize(
        "counts, expected_true_positive_rate",
        [
            ([0, 1, 0, 0], 0),
            ([0, 0, 0, 0], 1),
            ([0, 1, 1, 0], 0),
            ([1, 1, 1, 1], 0.5),
            ([5, 0, 0, 0], 1),
            ([20, 0, 2, 60], 0.25),
        ],
    )
    def test_calculate_true_positive_rate(self, counts, expected_true_positive_rate):
        """Tests binary true positive rate estimator"""
        assert _calculate_recall(counts) == expected_true_positive_rate

    @pytest.mark.parametrize(
        "class_probs_1vR, prediction_probs_1vR, expected_cm_results",
        [
            ([1, 0], [1, 0], np.array([1, 0, 0, 0])),
            ([0, 1], [0, 1], np.array([0, 0, 1, 0])),
            ([0.5, 0.5], [0.5, 0.5], np.array([0.25, 0.25, 0.25, 0.25])),
            ([0.5, 0], [0.5, 0], np.array([0.25, 0, 0, 0])),
            ([0, 0.5], [0.5, 0], np.array([0, 0.25, 0, 0])),
            ([0, 0.5], [0, 0.5], np.array([0, 0, 0.25, 0])),
            ([0.5, 0], [0, 0.5], np.array([0, 0, 0, 0.25])),
            ([0, 0.5], [0, 0], np.array([0, 0, 0, 0])),
        ],
    )
    def test_get_tpfptnfn(self, class_probs_1vR, prediction_probs_1vR, expected_cm_results):
        """Tests classification results estimator"""
        assert all(_to_confusion_matrix(class_probs_1vR, prediction_probs_1vR) == expected_cm_results)


@pytest.mark.required
class TestReducers:
    @pytest.mark.parametrize(
        "method, counts, expected_value",
        [
            (_calculate_accuracy, [[1, 0, 0, 0], [0, 0, 1, 0]], 1),
            (_calculate_precision, [[1, 0, 0, 0], [0, 0, 0, 0]], 1),
            (_calculate_fpr, [[0, 1, 0, 0], [0, 0, 0, 0]], 1),
            (_calculate_recall, [[1, 0, 0, 0], [0, 0, 0, 0]], 1),
        ],
    )
    def test_micro_reducer(self, method, counts, expected_value):
        """Tests micro reducer with each callable estimator"""
        assert _reduce_micro(method, counts) == expected_value

    @pytest.mark.parametrize(
        "method, counts, expected_value",
        [
            (_calculate_accuracy, [[1, 0, 1, 0], [0, 1, 0, 1]], 0.5),
            (_calculate_precision, [[1, 0, 0, 0], [0, 0, 1, 1]], 0.5),
            (_calculate_fpr, [[0, 1, 0, 0], [1, 0, 0, 1]], 0.5),
            (_calculate_recall, [[1, 0, 0, 0], [0, 1, 0, 1]], 0.5),
        ],
    )
    def test_macro_reducer(self, method, counts, expected_value):
        """Tests macro reducer with each callable estimator"""
        assert _reduce_macro(method, counts) == expected_value
