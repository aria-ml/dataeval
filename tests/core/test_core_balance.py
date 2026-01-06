import logging

import numpy as np
import pytest

from dataeval.core._mutual_info import (
    _merge_labels_and_factors,
    _validate_num_neighbors,
    mutual_info,
)

CLASS_LABELS = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
FACTOR_DATA = np.array(
    [
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 2],
        [1, 0, 3],
        [2, 0, 4],
        [2, 1, 5],
        [3, 1, 6],
        [3, 1, 7],
        [4, 1, 8],
        [4, 1, 9],
    ]
)


@pytest.mark.required
class TestBalanceValidateNumNeighbors:
    @pytest.mark.parametrize(
        "test_param, expected_exception, err_msg",
        [
            ("7", pytest.raises(TypeError), "Variable 7 is not real-valued numeric type."),
            (0, pytest.raises(ValueError), "Invalid value for 0."),
        ],
    )
    def test_validate_num_neighbors_type_errors(self, test_param, expected_exception, err_msg):
        with expected_exception as e:
            _validate_num_neighbors(test_param)
        assert err_msg in str(e.value)

    def test_validate_num_neighbors_warning(self, caplog):
        err_msg = "Variable 4 is currently type float and will be truncated to type int."
        with caplog.at_level(logging.WARNING):
            _validate_num_neighbors(4.0)  # type: ignore
        assert err_msg in caplog.text

    def test_validate_num_neighbors_pass(self):
        _validate_num_neighbors(10)
        pass


@pytest.mark.required
class TestBalanceMergeLabelsAndFactors:
    def test_without_discrete_features(self):
        data, discrete_features = _merge_labels_and_factors(CLASS_LABELS, FACTOR_DATA, None)
        assert data.shape == (FACTOR_DATA.shape[0], FACTOR_DATA.shape[1] + 1)
        assert discrete_features == [False, True, True, False]

    def test_provided_discrete_features(self):
        provided_discrete_features = [False, True, False]
        expected_discrete_features = [False] + provided_discrete_features

        data, discrete_features = _merge_labels_and_factors(CLASS_LABELS, FACTOR_DATA, provided_discrete_features)
        assert data.shape == (FACTOR_DATA.shape[0], FACTOR_DATA.shape[1] + 1)
        assert discrete_features == expected_discrete_features

    def test_provided_discrete_features_override_unique(self):
        provided_discrete_features = [False, True, True]
        expected_discrete_features = [False, False, True, False]

        data, discrete_features = _merge_labels_and_factors(CLASS_LABELS, FACTOR_DATA, provided_discrete_features)
        assert data.shape == (FACTOR_DATA.shape[0], FACTOR_DATA.shape[1] + 1)
        assert discrete_features == expected_discrete_features


@pytest.mark.required
class TestBalanceFunctional:
    def test_balance(self):
        """Test the balance function with TypedDict return."""
        result = mutual_info(CLASS_LABELS, FACTOR_DATA)

        # Test that result is a dict with the expected keys
        assert "class_to_factor" in result
        assert "interfactor" in result

        # Test factors array
        assert result["class_to_factor"].ndim == 1
        assert len(result["class_to_factor"]) == FACTOR_DATA.shape[1] + 1
        np.testing.assert_allclose(
            result["class_to_factor"],
            np.array([0.255898, 0.032484, 0.0, 0.036158]),
            atol=1e-6,
        )

        # Test interfactor matrix
        assert result["interfactor"].ndim == 2
        assert result["interfactor"].shape == (FACTOR_DATA.shape[1], FACTOR_DATA.shape[1])
        np.testing.assert_allclose(result["interfactor"], result["interfactor"].T, atol=1e-6)
        np.testing.assert_allclose(
            result["interfactor"],
            np.array([[1.0, 0.8, 0.55596], [0.8, 1.0, 0.785557], [0.55596, 0.785557, 0.705458]]),
            atol=1e-6,
        )
