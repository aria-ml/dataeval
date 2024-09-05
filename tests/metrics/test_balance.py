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
np.random.seed(7)


@pytest.fixture
def corr_val():
    return 0.75


@pytest.fixture
def num_samples():
    return 20


@pytest.fixture
def class_labels(num_samples):
    return [np.random.randint(low=0, high=2) for _ in range(num_samples)]


@pytest.fixture
def metadata(corr_val, num_samples, class_labels):
    vals = ["a", "b"]

    # for unit testing
    md = [
        {
            "var_cat": vals[get_index_one_class(class_labels[idx], corr_val)],
            "var_cnt": np.random.randn(),
            "var_float_cat": np.random.randint(low=0, high=len(vals)) + 0.1,
        }
        for idx in range(num_samples)
    ]
    return md


@pytest.fixture
def metadata_no_cnt(num_samples, metadata):
    md = [
        {
            "var_cat": metadata[idx]["var_cat"],
        }
        for idx in range(num_samples)
    ]
    return md


@pytest.fixture
def num_samp_func():
    return 500


@pytest.fixture
def class_labels_func(num_samp_func):
    return [np.random.randint(low=0, high=2) for _ in range(num_samp_func)]


@pytest.fixture
def metadata_one(num_samp_func, class_labels_func, corr_val):
    vals = ["a", "b"]
    md = [
        {
            "var_cat": vals[get_index_one_class(class_labels_func[idx], corr_val)],
            "var_cnt": np.random.randn(),
        }
        for idx in range(num_samp_func)
    ]
    return md


@pytest.fixture
def metadata_all(num_samp_func, class_labels_func, corr_val):
    vals = ["a", "b"]
    md = [
        {
            "var_cat": vals[get_index_all_class(class_labels_func[idx], corr_val)],
            "var_cnt": np.random.randn(),
        }
        for idx in range(num_samp_func)
    ]
    return md


@pytest.fixture
def num_vars(metadata):
    return len(metadata[0].keys())


@pytest.fixture
def balance_expected_shape(num_vars):
    return (num_vars + 1, num_vars + 1)


@pytest.fixture
def balance_cw_expected_shape(num_vars):
    return (2, num_vars)


@pytest.fixture
def mi_one(corr_val):
    return analytic_mi(corr_val, one_class=True)


@pytest.fixture
def mi_all(corr_val):
    return analytic_mi(corr_val, one_class=False)


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

    def test_preprocess(self, metadata, class_labels, num_vars):
        data, names, is_categorical = preprocess_metadata(class_labels, metadata)
        assert len(names) == num_vars + 1  # 3 variables, class_label
        idx = names.index("var_cnt")
        assert not is_categorical[idx]
        assert all(is_categorical[:idx] + is_categorical[idx + 1 :])
        assert data.dtype == float

    def test_preprocess_no_float(self, class_labels, metadata_no_cnt):
        # test case with no float data
        balance(class_labels=class_labels, metadata=metadata_no_cnt)

    @pytest.mark.parametrize(
        "balance_fn, expected_shape",
        [(balance, "balance_expected_shape"), (balance_classwise, "balance_cw_expected_shape")],
    )
    def test_correct_mi_shape_and_dtype(self, balance_fn, expected_shape, class_labels, metadata, request):
        mi = balance_fn(class_labels, metadata).mutual_information
        expected_shape = request.getfixturevalue(expected_shape)
        assert mi.shape == expected_shape
        assert mi.dtype == float


class TestBalanceFunctional:
    @pytest.mark.parametrize(
        "md, mi_val",
        [("metadata_all", "mi_all"), ("metadata_one", "mi_one")],
    )
    def test_correct_balance(self, md, mi_val, class_labels_func, request):
        md = request.getfixturevalue(md)
        mi_val = request.getfixturevalue(mi_val)
        mi = balance(class_labels_func, md).mutual_information
        # first row (with class)
        mi_0 = np.array([1, mi_val, 0])
        print(np.max(np.abs(mi[0] - mi_0)))
        assert np.max(np.abs(mi[0] - mi_0)) < 0.067

    @pytest.mark.parametrize(
        "md, mi_val",
        [("metadata_all", "mi_all"), ("metadata_one", "mi_one")],
    )
    def test_correct_balance_classwise(self, md, mi_val, class_labels_func, request):
        md = request.getfixturevalue(md)
        mi_val = request.getfixturevalue(mi_val)
        mi = balance_classwise(class_labels_func, md).mutual_information
        print(np.max(np.abs(mi[:, 0] - mi_val)))
        assert np.max(np.abs(mi[:, 0] - mi_val)) < 0.067
