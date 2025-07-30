import numpy as np
import pytest

from dataeval.core._balance import _merge_labels_and_factors, _validate_num_neighbors, balance, balance_classwise

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

    def test_validate_num_neighbors_warning(self):
        err_msg = "[ UserWarning('Variable 4 is currently type float and will be truncated to type int.')]"
        with pytest.warns(UserWarning, match=err_msg):
            _validate_num_neighbors(4.0)  # type: ignore

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
        output = balance(CLASS_LABELS, FACTOR_DATA)
        np.testing.assert_allclose(
            output,
            np.array(
                [
                    [0.115358, 0.045411, 0.0, 0.006385],
                    [0.045411, 1.0, 0.481648, 0.538835],
                    [0.0, 0.481648, 1.0, 0.331053],
                    [0.006385, 0.538835, 0.331053, 0.205118],
                ]
            ),
            atol=1e-6,
        )

    def test_balance_classwise(self):
        output = balance_classwise(CLASS_LABELS, FACTOR_DATA)
        np.testing.assert_allclose(
            output,
            np.array(
                [
                    [2.32606, 0.240824, 0.029049, 0.0],
                    [2.32606, 0.240824, 0.029049, 0.0],
                ]
            ),
            atol=1e-6,
        )
