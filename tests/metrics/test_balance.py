from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from dataeval._internal.metrics.metadata import Balance, BalanceClasswise


@pytest.fixture(params=[Balance, BalanceClasswise])
def balance(request):
    # use a decent number of samples because is_categorical is inferred.  small
    # samples could break that test
    num_samples = 20
    vals = ["a", "b"]
    metadata = [
        {"var_cat": vals[np.random.randint(low=0, high=len(vals))], "var_cnt": np.random.randn()} for _ in range(20)
    ]
    class_labels = [np.random.randint(low=0, high=2) for _ in range(num_samples)]
    balance = request.param()
    # this is typically in a loop, provided by a dataloader
    balance.update(class_labels, metadata)
    balance._collect_data()
    return balance


class TestBalanceUnit:
    @pytest.mark.parametrize(
        "test_param,expected_exception",
        [
            ("7", pytest.raises(TypeError)),
            (0, pytest.raises(ValueError)),
            (10, does_not_raise()),
        ],
    )
    @pytest.mark.parametrize("test_class", [Balance, BalanceClasswise])
    def test_validates_balance_inputs(self, test_param, expected_exception, test_class):
        with expected_exception:
            test_class(test_param)

    def test_passes_with_correct_num_factors(self, balance):
        assert balance.num_factors == 2 + 1  # var1, var2, class_label

    def test_passes_with_correct_num_samples(self, balance):
        assert balance.num_samples == 20

    def test_passes_with_correct_is_categorical(self, balance):
        idx = balance.names.index("var_cnt")
        assert not balance.is_categorical[idx]
        assert all(balance.is_categorical[:idx] + balance.is_categorical[idx + 1 :])

    def test_passes_with_data_conversion_to_numeric(self, balance):
        # after adding continuous variables the entire 'data' array gets cast up
        # to float instead of int
        assert balance.data.dtype == float

    def test_passes_with_correct_mi_shape_and_dtype(self, balance):
        mi = balance.compute()
        if isinstance(balance, Balance):
            assert mi.shape[0] == 3 & mi.shape[1] == 3
        elif isinstance(balance, BalanceClasswise):
            assert mi.shape[0] == 2 & mi.shape[1] == balance.num_factors - 1
        assert mi.dtype == float
