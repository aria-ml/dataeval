import numpy as np
import pytest

from dataeval._internal.metrics.metadata import Balance, BalanceClasswise


@pytest.fixture
def balance_obj(num_samples=20, num_factors=2):
    # use a decent number of samples because is_categorical is inferred.  small
    # samples could break that test
    vals = ["a", "b"]
    metadata = [
        {f"var{idx}": vals[np.random.randint(low=0, high=len(vals))] for idx in range(num_factors)}
        for _ in range(num_samples)
    ]
    class_labels = [np.random.randint(low=0, high=1) for _ in range(num_samples)]
    balance = Balance()
    # this is typically in a loop, provided by a dataloader
    balance.update(class_labels, metadata)
    balance._collect_data()
    return balance


@pytest.fixture
def balance_cw_obj(num_samples=20, num_factors=2):
    # use a decent number of samples because is_categorical is inferred.  small
    # samples could break that test
    vals = ["a", "b"]
    metadata = [
        {f"var{idx}": vals[np.random.randint(low=0, high=len(vals))] for idx in range(num_factors)}
        for _ in range(num_samples)
    ]
    class_labels = [np.random.randint(low=0, high=2) for _ in range(num_samples)]
    balance = BalanceClasswise()
    # this is typically in a loop, provided by a dataloader
    balance.update(class_labels, metadata)
    balance._collect_data()
    return balance


class TestBalanceUnit:
    def test_fails_with_invalid_num_neighbors_type(self):
        with pytest.raises(TypeError):
            metric = Balance(num_neighbors="10")  # type: ignore

    def test_fails_with_invalid_num_neighbors_value(self):
        with pytest.raises(ValueError):
            metric = Balance(num_neighbors=0)

    def test_passes_with_valid_num_neighbors_value(self):
        metric = Balance(10)

    def test_passes_with_correct_num_factors(self, balance_obj):
        assert balance_obj.num_factors == 2 + 1  # var1, var2, class_label

    def test_passes_with_correct_num_samples(self, balance_obj):
        assert balance_obj.num_samples == 20

    def test_passes_with_correct_is_categorical(self, balance_obj):
        assert all(balance_obj.is_categorical)  # var1, var2, class_label

    def test_passes_with_data_conversion_to_int(self, balance_obj):
        assert balance_obj.data.dtype == int

    def test_passes_with_correct_mi_shape_and_dtype(self, balance_obj):
        mi = balance_obj.compute()
        assert mi.shape[0] == 3 & mi.shape[1] == 3
        assert mi.dtype == float
        assert all(0 <= val <= 1 for val in mi.ravel())


class TestBalanceClasswiseUnit:
    def test_fails_with_invalid_num_neighbors_type(self):
        with pytest.raises(TypeError):
            metric = BalanceClasswise(num_neighbors="10")  # type: ignore

    def test_fails_with_invalid_num_neighbors_value(self):
        with pytest.raises(ValueError):
            metric = BalanceClasswise(num_neighbors=0)

    def test_passes_with_valid_num_neighbors_value(self):
        metric = BalanceClasswise(10)

    def test_passes_with_correct_num_factors(self, balance_cw_obj):
        assert balance_cw_obj.num_factors == 2 + 1  # var1, var2, class_label

    def test_passes_with_correct_num_samples(self, balance_cw_obj):
        assert balance_cw_obj.num_samples == 20

    def test_passes_with_correct_is_categorical(self, balance_cw_obj):
        assert all(balance_cw_obj.is_categorical)

    def test_passes_with_data_conversion_to_int(self, balance_cw_obj):
        assert balance_cw_obj.data.dtype == int

    def test_passes_with_correct_mi_shape_and_dtype(self, balance_cw_obj):
        mi = balance_cw_obj.compute()
        assert mi.shape[0] == 2 & mi.shape[1] == balance_cw_obj.num_factors - 1
        assert mi.dtype == float
