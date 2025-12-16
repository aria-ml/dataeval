import logging

import numpy as np
import pytest

from tests.conftest import to_metadata


@pytest.mark.required
class TestMDPreprocessingUnit:
    def test_bad_factor_ref(self, caplog):
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["a"] * 5 + ["b"] * 5}
        continuous_bincounts = {"something_else": 2}
        err_msg = "The keys - {'something_else'} - are present in the `continuous_factor_bins` dictionary"
        with caplog.at_level(logging.WARNING):
            to_metadata(factors, labels, continuous_bincounts)._bin()
        assert err_msg in caplog.text

    def test_doesnt_modify_input(self):
        factors = {"data1": [0.1, 0.2, 0.3]}
        labels = [0, 0, 0]
        bincounts = {"data1": 1}
        output = to_metadata(factors, labels, bincounts)
        if output.factor_data is not None:
            cont_factors = output.factor_data.T[0]
            assert np.all(cont_factors == [0.1, 0.2, 0.3])

    @pytest.mark.parametrize(
        "data_values",
        [
            list(np.random.rand(100)),
            list(np.random.randint(0, 1000, 100)),
        ],
    )
    def test_discrete_without_bins(self, data_values, caplog):
        factors = {"data": data_values}
        labels = list(np.random.randint(5, size=len(data_values)))
        err_msg = "A user defined binning was not provided for data."
        with caplog.at_level(logging.WARNING):
            to_metadata(factors, labels)._bin()
        assert err_msg in caplog.text

    @pytest.mark.parametrize("factors", ({"a": [1, 2, 3], "b": [1, 2, 3]}, {"a": [1, 2, 3]}))
    @pytest.mark.parametrize("bincounts", ({"a": 1, "b": 1}, {"a": 1}, None))
    def test_exclude_raw_metadata_only(self, factors, bincounts):
        labels = [0, 0, 0]
        output = to_metadata(factors, labels, bincounts, exclude=["b"])
        assert "b" not in output.index2label.values()


@pytest.mark.optional
class TestMDPreprocessingFunctional:
    def test_nbins(self):
        factors = {"data1": [0.1, 0.2, 0.3, 1.1, 1.2]}
        labels = [0, 0, 0, 0, 0]
        bincounts = {"data1": 2}
        output = to_metadata(factors, labels, bincounts)
        disc_factors = output.binned_data
        assert len(np.unique(disc_factors)) == 2

    def test_bin_edges(self):
        factors = {"data1": [0.1, 0.2, 0.3, 1.1, 1.2]}
        bin_edges = {"data1": [-np.inf, 1, np.inf]}
        labels = [0, 0, 0, 0, 0]
        output = to_metadata(factors, labels, bin_edges)
        disc_factors = output.binned_data
        assert len(np.unique(disc_factors)) == 2

    def test_mix_match(self):
        factors = {"data1": [-1.1, 0.2, 0.3, 1.1, 1.2], "data2": [-1.1, 0.2, 0.3, 1.1, 1.2]}
        labels = [0, 1, 2, 0, 1]
        bincounts = {"data1": 3, "data2": [-np.inf, 1, np.inf]}
        output = to_metadata(factors, labels, bincounts)
        disc_factors = output.binned_data.T
        assert len(np.unique(disc_factors[0])) == 3
        assert len(np.unique(disc_factors[1])) == 2

    def test_one_bin(self):
        factors = {"data1": [0.1, 0.2, 0.3, 1.1, 1.2]}
        labels = [0, 0, 0, 0, 0]
        bincounts = {"data1": 1}
        output = to_metadata(factors, labels, bincounts)
        disc_factors = output.binned_data
        assert len(np.unique(disc_factors)) == 1

    def test_over_specified(self):
        factors = {"data1": [0.1, 0.2, 0.3, 1.1, 1.2]}
        labels = [0, 0, 0, 0, 0]
        bincounts = {"data1": 100}
        output = to_metadata(factors, labels, bincounts)
        disc_factors = output.binned_data
        assert len(np.unique(disc_factors)) == 5
