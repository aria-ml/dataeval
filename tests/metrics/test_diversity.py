import numpy as np
import pytest

from dataeval._internal.metrics.diversity import Diversity, DiversityClasswise

num_samples = 20
vals = ["a", "b"]
metadata = [
    {"var_cat": vals[np.random.randint(low=0, high=len(vals))], "var_cnt": np.random.randn()} for _ in range(20)
]
class_labels = [np.random.randint(low=0, high=2) for _ in range(num_samples)]


@pytest.mark.parametrize("div_class", [Diversity, DiversityClasswise])
class TestDiversityUnit:
    @pytest.mark.parametrize("met", ["Simpson", "ShANnOn"])
    def test_invalid_method(self, div_class, met):
        with pytest.raises(KeyError):
            div_class(method=met)  # type: ignore

    @pytest.mark.parametrize("met", ["simpson", "shannon"])
    def test_valid_method(self, div_class, met):
        div_class(method=met)

    @pytest.mark.parametrize("met", ["simpson", "shannon"])
    def test_range_of_values(self, div_class, met):
        div = div_class(method=met).evaluate(class_labels, metadata)
        assert div.dtype == float
        assert np.logical_and(div >= 0, div <= 1).all()
