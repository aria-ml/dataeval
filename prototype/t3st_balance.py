from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from dataeval._internal.metrics.utils import infer_categorical, preprocess_metadata
from dataeval.metrics import balance, balance_classwise


def get_index(class_label, corr_val, one_class=True):
    cond = class_label == 0 and rng_bal.random() < corr_val if one_class else rng_bal.random() < corr_val
    return class_label if cond else rng_bal.integers(low=0, high=2)


def conditionals_all_classes(corr):
    v1 = corr + (1 - corr) / 2
    v2 = 1 - v1
    pr_cond = np.array([[v1, v2], [v2, v1]])
    return pr_cond


def conditionals_one_class(corr):
    v1 = corr + (1 - corr) / 2
    v2 = 1 - v1
    pr_cond = np.array([[v1, 0.5], [v2, 0.5]])
    return pr_cond


def analytic_mi(corr, one_class=True):
    # analytical mutual information for factor correlated with class label
    # one_class=True -> classwise_balance
    # one_class=False-> balance
    cond = conditionals_one_class(corr) if one_class else conditionals_all_classes(corr)
    pr_class = np.array([0.5, 0.5])
    ent_class = -sum(pr_class * np.log(pr_class))
    joint = cond * pr_class.T
    pr_factor = cond.dot(pr_class)
    ent_factor = -sum(pr_factor * np.log(pr_factor))
    mi = np.sum(joint * np.log(joint / np.outer(pr_factor, pr_class)))
    return mi * 2 / (ent_class + ent_factor)


# fix seed so that pipelines do not stochastically fail
rng_bal = np.random.default_rng(1234)


@pytest.fixture
def class_labels():
    return [rng_bal.integers(low=0, high=2) for _ in range(20)]


@pytest.fixture
def metadata(class_labels):
    vals = ["a", "b"]
    # for unit testing
    md = [
        {
            "var_cat": vals[get_index(class_labels[idx], 0.75, one_class=True)],
            "var_cnt": rng_bal.standard_normal(),
            "var_float_cat": rng_bal.integers(low=0, high=len(vals)) + 0.1,
        }
        for idx in range(20)
    ]
    return md


@pytest.fixture
def class_labels_func():
    return [rng_bal.integers(low=0, high=2) for _ in range(500)]


@pytest.fixture
def metadata_one(class_labels_func):
    vals = ["a", "b"]
    md = [
        {
            "var_cat": vals[get_index(class_labels_func[idx], 0.75, one_class=True)],
            "var_cnt": rng_bal.standard_normal(),
        }
        for idx in range(500)
    ]
    return md


@pytest.fixture
def metadata_all(class_labels_func):
    vals = ["a", "b"]
    md = [
        {
            "var_cat": vals[get_index(class_labels_func[idx], 0.75, one_class=False)],
            "var_cnt": rng_bal.standard_normal(),
        }
        for idx in range(500)
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


class TestBalanceFunctional:
    @pytest.mark.parametrize(
        "md, mi_type",
        [("metadata_all", "mi_all"), ("metadata_one", "mi_one")],
    )
    def test_correct_balance(self, md, mi_type, class_labels_func, request):
        one_class = "one" in mi_type
        md = request.getfixturevalue(md)
        mi_val = analytic_mi(0.75, one_class=one_class)
        # mi_val = request.getfixturevalue(mi_val)
        mi = balance(class_labels_func, md).mutual_information
        # first row (with class)
        mi_0 = np.array([1, mi_val, 0])
        assert np.max(np.abs(mi[0] - mi_0)) < 0.054

    @pytest.mark.parametrize(
        "md, mi_type",
        [("metadata_all", "mi_all"), ("metadata_one", "mi_one")],
    )
    def test_correct_balance_classwise(self, md, mi_type, class_labels_func, request):
        one_class = "one" in mi_type
        md = request.getfixturevalue(md)
        mi_val = analytic_mi(0.75, one_class=one_class)
        mi = balance_classwise(class_labels_func, md).mutual_information
        assert np.max(np.abs(mi[:, 0] - mi_val)) < 0.05

