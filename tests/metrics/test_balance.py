from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

try:
    from matplotlib.figure import Figure
except ImportError:
    Figure = type(None)

from dataeval.metrics.bias._balance import _validate_num_neighbors, balance
from tests.conftest import preprocess


@pytest.fixture(scope="module")
def metadata_results():
    str_vals = ["b", "b", "b", "b", "b", "a", "a", "b", "a", "b", "b", "a"]
    cnt_vals = [
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
    cat_vals = [1.1, 1.1, 0.1, 0.1, 1.1, 0.1, 1.1, 0.1, 0.1, 1.1, 1.1, 0.1]
    class_labels = ["dog", "dog", "dog", "cat", "dog", "cat", "dog", "dog", "dog", "cat", "cat", "cat"]
    md = {"var_cat": str_vals, "var_cnt": cnt_vals, "var_float_cat": cat_vals}
    return preprocess(md, class_labels, {"var_cnt": 3, "var_float_cat": 2})


@pytest.fixture(scope="module")
def mismatch_metadata():
    raw_metadata = {"factor1": list(range(10)), "factor2": list(range(10)), "factor3": list(range(10))}
    class_labels = [1] * 10
    continuous_bins = {"factor1": 5, "factor2": 5, "factor3": 5}
    return preprocess(raw_metadata, class_labels, continuous_bins)


@pytest.fixture(scope="module")
def simple_metadata():
    raw_metadata = {"factor1": [1] * 100 + [2] * 100, "factor2": [1] * 100 + [2] * 100}
    class_labels = [1] * 100 + [2] * 100
    return preprocess(raw_metadata, class_labels)


@pytest.mark.required
class TestBalanceUnit:
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
        with does_not_raise():
            _validate_num_neighbors(10)

    def test_correct_mi_shape_and_dtype(self, metadata_results):
        num_vars = len(metadata_results.discrete_factor_names + metadata_results.continuous_factor_names) + 1
        expected_shape = {
            "balance": (num_vars,),
            "factors": (num_vars - 1, num_vars - 1),
            "classwise": (2, num_vars),
            "factor_names": (num_vars),
            "class_names": (np.unique(metadata_results.class_labels).size),
        }
        expected_type = {
            "balance": float,
            "factors": float,
            "classwise": float,
            "factor_names": list,
            "class_names": list,
        }
        mi = balance(metadata_results)
        for k, v in mi.data().items():
            if type(v) is list:
                assert len(v) == expected_shape[k]
            else:
                assert v.shape == expected_shape[k]
                if k in expected_type:
                    assert v.dtype == expected_type[k]


@pytest.mark.requires_all
@pytest.mark.required
class TestBalancePlot:
    def test_base_plotting(self, metadata_results):
        mi = balance(metadata_results)
        output = mi.plot()
        assert isinstance(output, Figure)
        classwise_output = mi.plot(plot_classwise=True)
        assert isinstance(classwise_output, Figure)

    @pytest.mark.parametrize("factor_type", ("discrete", "continuous", "both"))
    def test_plotting_vars(self, metadata_results, factor_type):
        mi = balance(metadata_results)
        factor_names = mi._by_factor_type("factor_names", factor_type)
        heat_labels = np.arange(len(factor_names))
        output = mi.plot(heat_labels[:-1], heat_labels[1:], plot_classwise=False, factor_type=factor_type)
        assert isinstance(output, Figure)
        _, row_labels = np.unique(mi.class_names, return_inverse=True)
        col_labels = np.arange(len(factor_names))
        classwise_output = mi.plot(row_labels, col_labels, plot_classwise=True, factor_type=factor_type)
        assert isinstance(classwise_output, Figure)


@pytest.mark.optional
class TestBalanceFunctional:
    def test_unity_balance(self, simple_metadata):
        output = balance(simple_metadata)
        assert np.all(output.balance > 0.999)
        assert np.all(output.factors > 0.999)

    def test_zero_balance(self, mismatch_metadata):
        output = balance(mismatch_metadata)
        assert np.all(np.isclose(output.balance, 0))
