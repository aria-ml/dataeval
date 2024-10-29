from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from dataeval.metrics.bias import balance
from dataeval.metrics.bias.metadata import infer_categorical, preprocess_metadata


@pytest.fixture
def simple_class_labels():
    return [1] * 100 + [2] * 100


@pytest.fixture
def homog_class_labels():
    return [1] * 10


@pytest.fixture
def inhomog_metadata():
    return {
        "factor1": list(range(10)),
        "factor2": list(range(10)),
        "factor3": list(range(10)),
    }


@pytest.fixture
def simple_metadata():
    return {"factor1": [1] * 100 + [2] * 100, "factor2": [1] * 100 + [2] * 100}


@pytest.fixture
def simple_class_labels():
    return [1] * 100 + [2] * 100


@pytest.fixture
def homog_class_labels():
    return [1] * 10


@pytest.fixture
def inhomog_metadata():
    return {
        "factor1": list(range(10)),
        "factor2": list(range(10)),
        "factor3": list(range(10)),
    }


@pytest.fixture
def simple_metadata():
    return {"factor1": [1] * 100 + [2] * 100, "factor2": [1] * 100 + [2] * 100}


@pytest.fixture
def class_labels():
    return np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0], dtype=int)


@pytest.fixture
def metadata():
    str_vals = np.array(["b", "b", "b", "b", "b", "a", "a", "b", "a", "b", "b", "a"])
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
    cat_vals = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0], dtype=int)
    # for unit testing
    md = {"var_cat": str_vals, "var_cnt": cnt_vals, "var_float_cat": cat_vals + 0.1}
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
    def test_validates_balance_inputs(self, test_param, expected_exception, class_labels, metadata):
        with expected_exception:
            balance(class_labels, metadata, test_param)

    def test_preprocess(self, metadata, class_labels):
        data, names, is_categorical = preprocess_metadata(class_labels, metadata)
        assert len(names) == len(metadata.keys()) + 1  # 3 variables, class_label
        idx = names.index("var_cnt")
        assert not is_categorical[idx]
        assert all(is_categorical[:idx] + is_categorical[idx + 1 :])
        assert data.dtype == float

    def test_infer_categorical_2D_data(self):
        x = np.ones((10, 2))
        _ = infer_categorical(x)

    def test_correct_mi_shape_and_dtype(self, class_labels, metadata):
        num_vars = len(metadata.keys())
        expected_shape = {
            "balance": (num_vars + 1,),
            "factors": (num_vars, num_vars),
            "classwise": (2, num_vars + 1),
            "class_list": (len(class_labels),),
            "metadata_names": (len(metadata.keys())),
        }
        expected_type = {
            "balance": float,
            "factors": float,
            "classwise": float,
            "class_list": int,
        }
        mi = balance(class_labels, metadata)
        for k, v in mi.dict().items():
            if type(v) is list:
                assert len(v) == expected_shape[k]
            else:
                assert v.shape == expected_shape[k]
                if k in expected_type:
                    assert v.dtype == expected_type[k]


class TestBalanceFunctional:
    def test_unity_balance(self, simple_class_labels, simple_metadata):
        output = balance(simple_class_labels, simple_metadata)
        assert np.all(output.balance > 0.999)
        assert np.all(output.factors > 0.999)

    def test_zero_balance(self, homog_class_labels, inhomog_metadata):
        output = balance(homog_class_labels, inhomog_metadata)
        assert np.all(np.isclose(output.balance, 0))
