from dataclasses import asdict

import numpy as np
import pytest

from dataeval.metrics.estimators._nullmodel import null_model_metrics


@pytest.mark.required
class TestNullModelMetrics:
    @pytest.mark.parametrize(
        "test_labels, train_labels, expected_models, expected_metrics",
        [
            # binary, no train set
            (np.random.randint(1, 3, size=100), None, 1, 8),
            # binary, train set
            (np.random.randint(4, 6, size=100), np.random.randint(0, 2, size=100), 3, 8),
            # multiclass, no train set
            (np.random.randint(0, 5, size=100), None, 1, 7),
            # multiclass, train set
            (np.random.randint(20, 25, size=100), np.random.randint(0, 5, size=100), 3, 7),
            # multiclass test set, binary train set
            (
                np.random.randint(0, 5, size=100),
                np.random.randint(0, 2, size=100),
                3,
                7,
            ),
            # multiclass train set, binary test set
            (
                np.random.randint(0, 2, size=100),
                np.random.randint(0, 5, size=100),
                3,
                7,
            ),
        ],
    )
    def test_null_model_inputs(self, test_labels, train_labels, expected_models, expected_metrics):
        """Tests expected models and expected metrics for binary and multiclass label sets"""
        output = null_model_metrics(test_labels, train_labels)
        output_dict = {k: v for k, v in output.data().items() if v is not None}
        assert len(output_dict) == expected_models
        for metrics in output_dict.values():
            metrics_list = [k for k, v in asdict(metrics).items() if v is not None]
            assert len(metrics_list) == expected_metrics

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
