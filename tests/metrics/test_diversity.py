import numpy as np
import pytest

from dataeval._internal.metrics.utils import entropy
from dataeval.metrics import diversity, diversity_classwise

num_samples = 20
vals = ["a", "b"]
metadata = [
    {"var_cat": vals[np.random.randint(low=0, high=len(vals))], "var_cnt": np.random.randn()} for _ in range(20)
]
class_labels = [np.random.randint(low=0, high=2) for _ in range(num_samples)]


@pytest.fixture
def entropy_test_vars():
    ent = {}
    ent["is_categorical"] = [True, False]
    ent["data"] = np.stack((np.ones(10, dtype=int), np.random.randint(low=0, high=8, size=10))).T
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
    def test_range_of_values(self, div_fn, met):
        div = div_fn(class_labels, metadata, method=met).diversity_index
        assert div.dtype == float
        assert np.logical_and(div >= 0, div <= 1).all()
