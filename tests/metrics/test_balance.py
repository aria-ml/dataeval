from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from dataeval._internal.metrics.utils import preprocess_metadata
from dataeval.metrics import balance, balance_classwise


def get_index_one_class(class_label, corr_val):
    # correlate this metadata with class label
    if class_label == 0 and np.random.rand() < corr_val:
        return class_label
    else:
        return np.random.randint(low=0, high=2)


def get_index_all_class(class_label, corr_val):
    # correlate this metadata with class label
    if np.random.rand() < corr_val:
        return class_label
    else:
        return np.random.randint(low=0, high=2)


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
np.random.seed(2)
num_samples = 20
corr_val = 0.75
vals = ["a", "b"]
class_labels = [np.random.randint(low=0, high=2) for _ in range(num_samples)]
# for unit testing
metadata = [
    {
        "var_cat": vals[get_index_one_class(class_labels[idx], corr_val)],
        "var_cnt": np.random.randn(),
        "var_float_cat": np.random.randint(low=0, high=len(vals)) + 0.1,
    }
    for idx in range(20)
]

# for functional testing
N = 1000
class_labels_n = [np.random.randint(low=0, high=2) for _ in range(N)]
metadata_one = [
    {
        "var_cat": vals[get_index_one_class(class_labels_n[idx], corr_val)],
        "var_cnt": np.random.randn(),
    }
    for idx in range(N)
]
metadata_all = [
    {
        "var_cat": vals[get_index_all_class(class_labels_n[idx], corr_val)],
        "var_cnt": np.random.randn(),
    }
    for idx in range(N)
]
num_vars = len(metadata[0].keys())
mi_one = analytic_mi(corr_val, one_class=True)
mi_all = analytic_mi(corr_val, one_class=False)


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
    def test_validates_balance_inputs(self, test_param, expected_exception, balance_fn):
        with expected_exception:
            balance_fn(class_labels, metadata, test_param)

    def test_preprocess(self):
        data, names, is_categorical = preprocess_metadata(class_labels, metadata)
        assert len(names) == num_vars + 1  # 3 variables, class_label
        idx = names.index("var_cnt")
        assert not is_categorical[idx]
        assert all(is_categorical[:idx] + is_categorical[idx + 1 :])
        assert data.dtype == float

    @pytest.mark.parametrize(
        "balance_fn, expected_shape",
        [(balance, (num_vars + 1, num_vars + 1)), (balance_classwise, (2, num_vars))],
    )
    def test_correct_mi_shape_and_dtype(self, balance_fn, expected_shape):
        mi = balance_fn(class_labels, metadata).mutual_information
        assert mi.shape == expected_shape
        assert mi.dtype == float


class TestBalanceFunctional:
    @pytest.mark.parametrize(
        "md, mi_val",
        [(metadata_all, mi_all), (metadata_one, mi_one)],
    )
    def test_correct_balance(self, md, mi_val):
        mi = balance(class_labels_n, md).mutual_information
        # first row (with class)
        mi_0 = np.array([1, mi_val, 0])
        assert np.max(np.abs(mi[0] - mi_0)) < 0.02

    @pytest.mark.parametrize(
        "md, mi_val",
        [(metadata_all, mi_all), (metadata_one, mi_one)],
    )
    def test_correct_balance_classwise(self, md, mi_val):
        mi = balance_classwise(class_labels_n, md).mutual_information
        assert np.max(np.abs(mi[:, 0] - mi_val)) < 0.02
