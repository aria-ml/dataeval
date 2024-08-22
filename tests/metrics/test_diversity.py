import numpy as np
import pytest

from dataeval._internal.metrics.metadata import Diversity, DiversityClasswise


@pytest.fixture(params=[(div, met) for met in ["simpson", "shannon"] for div in [Diversity, DiversityClasswise]])
def diversity(request):
    class_name, metric = request.param
    num_samples = 20
    vals = ["a", "b"]
    metadata = [
        {"var_cat": vals[np.random.randint(low=0, high=len(vals))], "var_cnt": np.random.randn()} for _ in range(20)
    ]
    class_labels = [np.random.randint(low=0, high=2) for _ in range(num_samples)]
    diversity = class_name(metric=metric)
    # this is typically in a loop, provided by a dataloader
    diversity.update(class_labels, metadata)
    diversity._collect_data()
    return diversity


class TestDiversityUnit:
    def test_fails_with_invalid_metric(self):
        with pytest.raises(ValueError):
            Diversity(metric="sampson")  # type: ignore

    @pytest.mark.parametrize("met", ["simpson", "shannon", "Simpson", "ShANnOn"])
    def test_passes_with_valid_metric(self, met):
        Diversity(metric=met)

    def test_passes_with_correct_num_factors(self, diversity):
        assert diversity.num_factors == 2 + 1  # var_cat, var_cnt, class_label

    def test_passes_with_correct_num_samples(self, diversity):
        assert diversity.num_samples == 20

    def test_passes_with_correct_is_categorical(self, diversity):
        idx = diversity.names.index("var_cnt")
        assert not diversity.is_categorical[idx]
        assert all(diversity.is_categorical[:idx] + diversity.is_categorical[idx + 1 :])

    def test_passes_with_correct_range_of_values(self, diversity):
        div = diversity.compute()
        assert div.dtype == float
        assert np.logical_and(div >= 0, div <= 1).all()
