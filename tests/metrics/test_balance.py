from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from dataeval._internal.metrics.balance import Balance, BalanceClasswise
from dataeval._internal.metrics.functional import preprocess_metadata

num_samples = 20
vals = ["a", "b"]
class_labels = [np.random.randint(low=0, high=2) for _ in range(num_samples)]
metadata = [
    {"var_cat": vals[np.random.randint(low=0, high=len(vals))], "var_cnt": np.random.randn()} for _ in range(20)
]


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

    def test_preprocess(self):
        data, names, is_categorical = preprocess_metadata(class_labels, metadata)
        assert len(names) == 2 + 1  # var1, var2, class_label
        idx = names.index("var_cnt")
        assert not is_categorical[idx]
        assert all(is_categorical[:idx] + is_categorical[idx + 1 :])
        assert data.dtype == float

    @pytest.mark.parametrize(
        "balance_class, expected_shape",
        [(Balance, (3, 3)), (BalanceClasswise, (2, 2))],
    )
    def test_correct_mi_shape_and_dtype(self, balance_class, expected_shape):
        mi = balance_class().evaluate(class_labels, metadata)
        assert mi.shape == expected_shape
        assert mi.dtype == float
