from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from dataeval._internal.metrics.utils import infer_categorical, preprocess_metadata
from dataeval.metrics import balance, balance_classwise


@pytest.fixture
def class_labels():
    return [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]


@pytest.fixture
def metadata():
    str_vals = ["b", "b", "b", "b", "b", "a", "a", "b", "a", "b", "b", "a"]
    cnt_vals = np.array(
        [
            -0.54425898,
            -0.31630016,
            0.41163054,
            1.04251337,
            -0.12853466,
            1.36646347,
            -0.66519467,
            0.35151007,
            0.90347018,
            0.0940123,
            -0.74349925,
            -0.92172538,
        ]
    )
    cat_vals = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0])
    # for unit testing
    md = [
        {
            "var_cat": strv,
            "var_cnt": cntv,
            "var_float_cat": catv + 0.1,
        }
        for strv, cntv, catv in zip(str_vals, cnt_vals, cat_vals)
    ]
    return md


class TestBalanceUnit:
    @pytest.mark.parametrize(
        "test_param,expected_exception",
        [
            ("7", pytest.raises(TypeError)),
            (0, pytest.raises(ValueError)),
            (10, does_not_raise()),
            (4.0, pytest.warns(UserWarning)),
        ],
    )
    @pytest.mark.parametrize("balance_fn", [balance, balance_classwise])
    def test_validates_balance_inputs(self, test_param, expected_exception, balance_fn, class_labels, metadata):
        with expected_exception:
            balance_fn(class_labels, metadata, test_param)

    def test_preprocess(self, metadata, class_labels):
        data, names, is_categorical = preprocess_metadata(class_labels, metadata)
        assert len(names) == len(metadata[0].keys()) + 1  # 3 variables, class_label
        idx = names.index("var_cnt")
        assert not is_categorical[idx]
        assert all(is_categorical[:idx] + is_categorical[idx + 1 :])
        assert data.dtype == float

    def test_preprocess_no_float(self, class_labels, metadata):
        # test case with no float data
        md = [{"var_cat": md["var_cat"]} for md in metadata]
        balance(class_labels=class_labels, metadata=md)

    def test_infer_categorical_2D_data(self):
        x = np.ones((10, 2))
        _ = infer_categorical(x)

    @pytest.mark.parametrize("balance_fcn", [balance, balance_classwise])
    def test_correct_mi_shape_and_dtype(self, balance_fcn, class_labels, metadata):
        num_vars = len(metadata[0].keys())
        expected_shape = (2, num_vars) if "classwise" in balance_fcn.__name__ else (num_vars + 1, num_vars + 1)
        mi = balance_fcn(class_labels, metadata).mutual_information
        assert mi.shape == expected_shape
        assert mi.dtype == float
