import numpy as np
import pytest

from dataeval.metrics.bias.metadata_binning import metadata_binning, user_defined_bin


class TestMDBinningUnit:
    def test_nbins_returns_array(self):
        factors = [0.1, 1.1, 1.2]
        bincounts = 2
        hist = user_defined_bin(factors, bincounts)
        assert type(hist) is np.ndarray

    def test_bin_edges_returns_array(self):
        factors = [0.1, 1.1, 1.2]
        bin_edges = [-np.inf, 1, np.inf]
        hist = user_defined_bin(factors, bin_edges)
        assert type(hist) is np.ndarray

    def test_crashes_with_negative_nbins(self):
        factors = [0.1, 1.1, 1.2]
        bincounts = -10
        with pytest.raises(ValueError):
            user_defined_bin(factors, bincounts)

    def test_crashes_with_wrong_order(self):
        factors = [0.1, 1.1, 1.2]
        bin_edges = [np.inf, 1, 2]
        with pytest.raises(ValueError):
            user_defined_bin(factors, bin_edges)

    def test_doesnt_modify_input(self):
        factors = [{"data1": [0.1, 0.2, 0.3]}]
        bincounts = {"data1": 1}
        output = metadata_binning(factors, bincounts)
        cont_factors = output.continuous["data1"]
        assert np.all(cont_factors == [0.1, 0.2, 0.3])


class TestMDBinningFunctional:
    def test_nbins(self):
        factors = [{"data1": [0.1, 0.2, 0.3, 1.1, 1.2]}]
        bincounts = {"data1": 2}
        output = metadata_binning(factors, bincounts)
        disc_factors = output.discrete
        assert len(np.unique(disc_factors["data1"])) == 2

    def test_bin_edges(self):
        factors = [{"data1": [0.1, 0.2, 0.3, 1.1, 1.2]}]
        bin_edges = {"data1": [-np.inf, 1, np.inf]}
        output = metadata_binning(factors, bin_edges)
        disc_factors = output.discrete
        assert len(np.unique(disc_factors["data1"])) == 2

    def test_mix_match(self):
        factors = [{"data1": [-1.1, 0.2, 0.3, 1.1, 1.2], "data2": [-1.1, 0.2, 0.3, 1.1, 1.2]}]
        bincounts = {"data1": 3, "data2": [-np.inf, 1, np.inf]}
        output = metadata_binning(factors, bincounts)
        disc_factors = output.discrete
        assert len(np.unique(disc_factors["data1"])) == 3
        assert len(np.unique(disc_factors["data2"])) == 2

    def test_udb_regression_nbins(self):
        factors = [0.1, 1.1, 1.2]
        bincounts = 2
        hist = user_defined_bin(factors, bincounts)
        assert np.all(hist == [1, 2, 2])

    def test_udb_regression_bin_edges(self):
        factors = [0.1, 1.1, 1.2]
        bin_edges = [-np.inf, 1, np.inf]
        hist = user_defined_bin(factors, bin_edges)
        assert np.all(hist == [1, 2, 2])

    def test_udb_regression_flipped_bin_edges(self):
        factors = [0.1, 1.1, 1.2]
        bin_edges = [np.inf, 1, -np.inf]
        hist = user_defined_bin(factors, bin_edges)
        assert np.all(hist == [2, 1, 1])

    def test_one_bin(self):
        factors = [{"data1": [0.1, 0.2, 0.3, 1.1, 1.2]}]
        bincounts = {"data1": 1}
        output = metadata_binning(factors, bincounts)
        disc_factors = output.discrete
        assert len(np.unique(disc_factors["data1"])) == 1

    def test_over_specified(self):
        factors = [{"data1": [0.1, 0.2, 0.3, 1.1, 1.2]}]
        bincounts = {"data1": 100}
        output = metadata_binning(factors, bincounts)
        disc_factors = output.discrete
        assert len(np.unique(disc_factors["data1"])) == 5
