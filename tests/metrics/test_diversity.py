import numpy as np
import pytest

from dataeval._internal.metrics.diversity import diversity, diversity_classwise
from dataeval._internal.metrics.utils import entropy


@pytest.fixture
def metadata():
    # vals = ["a", "b"]
    str_vals = ["a", "a", "a", "a", "b", "a", "a", "a", "b", "b"]
    cnt_vals = np.array(
        [0.63784, -0.86422, -0.1017, -1.95131, -0.08494, -1.02940, 0.07908, -0.31724, -1.45562, 1.03368]
    )
    md = [{"var_cat": sv, "var_cnt": cv} for sv, cv in zip(str_vals, cnt_vals)]
    return md


@pytest.fixture
def class_labels():
    return np.array([0, 0, 0, 1, 0, 1, 0, 0, 0, 1])


@pytest.fixture
def entropy_test_vars():
    ent = {}
    ent["is_categorical"] = [True, False]
    ent["data"] = np.stack([np.ones(10, dtype=int), np.array([1, 5, 3, 5, 8, 9, 0, 2, 4, 7])]).T
    ent["names"] = ["a", "b"]
    return ent


@pytest.mark.parametrize("norm", [True, False])
def test_entropy_normalization(norm, entropy_test_vars):
    ent = entropy(
        entropy_test_vars["data"], entropy_test_vars["names"], entropy_test_vars["is_categorical"], normalized=norm
    )
    assert ent[0] == 0


@pytest.mark.parametrize("div_fn", [diversity, diversity_classwise])
class TestDiversityUnit:
    @pytest.mark.parametrize("met", ["Simpson", "ShANnOn"])
    def test_invalid_method(self, div_fn, met):
        with pytest.raises(ValueError):
            div_fn([], [], method=met)  # type: ignore

    @pytest.mark.parametrize("met", ["simpson", "shannon"])
    def test_range_of_values(self, div_fn, met, metadata, class_labels):
        div = div_fn(class_labels, metadata, method=met).diversity_index
        assert div.dtype == float
        assert np.logical_and(div >= 0, div <= 1).all()
