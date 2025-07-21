import numpy as np
import pytest

from dataeval.metrics.estimators._nullmodel import (
    _estimate_accuracy,
    _estimate_ber,
    _estimate_false_positive_rate,
    _estimate_multiclass_accuracy,
    _estimate_multiclass_ber,
    _estimate_precision,
    _estimate_recall,
    _estimate_true_positive_rate,
    _get_tpfptnfn,
    _reduce_macro,
    _reduce_micro,
    null_model_metrics,
)


@pytest.mark.required
class TestNullModelMetrics:
    @pytest.mark.parametrize(
        "test_labels, train_labels, expected_models, expected_metrics",
        [
            # binary, no train set
            (np.random.randint(1, 3, size=100), None, 1, 12),
            # binary, train set
            (np.random.randint(4, 6, size=100), np.random.randint(0, 2, size=100), 3, 12),
            # multiclass, no train set
            (np.random.randint(0, 5, size=100), None, 1, 10),
            # multiclass, train set
            (np.random.randint(20, 25, size=100), np.random.randint(0, 5, size=100), 3, 10),
            # multiclass test set, binary train set
            (
                np.random.randint(0, 5, size=100),
                np.random.randint(0, 2, size=100),
                3,
                10,
            ),
            # multiclass train set, binary test set
            (
                np.random.randint(0, 2, size=100),
                np.random.randint(0, 5, size=100),
                3,
                10,
            ),
        ],
    )
    def test_null_model_inputs(self, test_labels, train_labels, expected_models, expected_metrics):
        """Tests expected models and expected metrics for binary and multiclass label sets"""
        output = null_model_metrics(test_labels, train_labels)
        assert len(output.results_by_metric) == expected_metrics
        assert len(output.results_by_model) == expected_models
        for model in output.results_by_model.values():
            assert len(model) == expected_metrics

        for metric in output.results_by_metric.values():
            assert len(metric) == expected_models

    def test_invalid_inputs(self):
        """Tests that invalid data properly raises exceptions"""
        test_labels = []
        with pytest.raises(ValueError, match="Empty or null test labels provided"):
            output = null_model_metrics(test_labels)
        test_labels = None
        with pytest.raises(ValueError, match="Empty or null test labels provided"):
            output = null_model_metrics(test_labels)  # type: ignore
        test_labels = np.random.randint(0, 5, size=100)
        train_labels = []
        output = null_model_metrics(test_labels, train_labels)
        assert output is not None


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
    def test_estimate_multiclass_accuracy(self, class_probs, model_probs, expected_accuracy):
        """Tests multiclass accuracy estimator"""
        assert _estimate_multiclass_accuracy(class_probs, model_probs) == expected_accuracy

    @pytest.mark.parametrize(
        "class_probs, model_probs, expected_ber",
        [
            ([0, 0, 0], [0, 0, 0], 1),
            ([1, 0, 0], [0, 1, 0], 1),
            ([1, 0, 0], [1, 0, 0], 0),
            ([0.25, 0.25, 0.5], [0.5, 0.25, 0.25], 0.6875),
            ([0.25, 0, 0.75], [0.25, 0, 0.75], 0.375),
        ],
    )
    def test_estimate_multiclass_ber(self, class_probs, model_probs, expected_ber):
        """Tests multiclass Bayes error rate estimator"""
        assert _estimate_multiclass_ber(class_probs, model_probs) == expected_ber

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
    def test_estimate_accuracy(self, counts, expected_accuracy):
        """Tests binary accuracy estimator"""
        assert _estimate_accuracy(counts) == expected_accuracy

    @pytest.mark.parametrize(
        "counts, expected_ber",
        [
            ([1, 0, 1, 0], 0),
            ([0, 0, 0, 20], 1),
            ([1, 1, 1, 1], 0.5),
            ([5, 0, 0, 0], 0),
            ([20, 26, 2, 40], 0.75),
        ],
    )
    def test_estimate_ber(self, counts, expected_ber):
        """Tests binary Bayes error rate estimator"""
        assert _estimate_ber(counts) == expected_ber

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
    def test_estimate_precision(self, counts, expected_precision):
        """Tests binary precision estimator"""
        assert _estimate_precision(counts) == expected_precision

    @pytest.mark.parametrize(
        "counts, expected_recall",
        [
            ([0, 1, 0, 0], 0),
            ([0, 0, 0, 0], 1),
            ([0, 1, 1, 0], 0),
            ([1, 1, 1, 1], 0.5),
            ([5, 0, 0, 0], 1),
            ([20, 0, 2, 60], 0.25),
        ],
    )
    def test_estimate_recall(self, counts, expected_recall):
        """Tests binary recall estimator"""
        assert _estimate_recall(counts) == expected_recall

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
    def test_estimate_false_positive_rate(self, counts, expected_false_positive_rate):
        """Tests binary false positive rate estimator"""
        assert _estimate_false_positive_rate(counts) == expected_false_positive_rate

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
    def test_estimate_true_positive_rate(self, counts, expected_true_positive_rate):
        """Tests binary true positive rate estimator"""
        assert _estimate_true_positive_rate(counts) == expected_true_positive_rate

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
        assert all(_get_tpfptnfn(class_probs_1vR, prediction_probs_1vR) == expected_cm_results)


@pytest.mark.required
class TestReducers:
    @pytest.mark.parametrize(
        "method, counts, expected_value",
        [
            (_estimate_accuracy, [[1, 0, 0, 0], [0, 0, 1, 0]], 1),
            (_estimate_ber, [[0, 0, 1, 0], [1, 0, 0, 0]], 0),
            (_estimate_precision, [[1, 0, 0, 0], [0, 0, 0, 0]], 1),
            (_estimate_recall, [[1, 0, 0, 0], [0, 0, 0, 0]], 1),
            (_estimate_false_positive_rate, [[0, 1, 0, 0], [0, 0, 0, 0]], 1),
            (_estimate_true_positive_rate, [[1, 0, 0, 0], [0, 0, 0, 0]], 1),
        ],
    )
    def test_micro_reducer(self, method, counts, expected_value):
        """Tests micro reducer with each callable estimator"""
        assert _reduce_micro(method, counts) == expected_value

    @pytest.mark.parametrize(
        "method, counts, expected_value",
        [
            (_estimate_accuracy, [[1, 0, 1, 0], [0, 1, 0, 1]], 0.5),
            (_estimate_ber, [[1, 0, 1, 0], [0, 1, 0, 1]], 0.5),
            (_estimate_precision, [[1, 0, 0, 0], [0, 0, 1, 1]], 0.5),
            (_estimate_recall, [[1, 0, 0, 0], [0, 1, 0, 1]], 0.5),
            (_estimate_false_positive_rate, [[0, 1, 0, 0], [1, 0, 0, 1]], 0.5),
            (_estimate_true_positive_rate, [[1, 0, 0, 0], [0, 1, 0, 1]], 0.5),
        ],
    )
    def test_macro_reducer(self, method, counts, expected_value):
        """Tests macro reducer with each callable estimator"""
        assert _reduce_macro(method, counts) == expected_value
