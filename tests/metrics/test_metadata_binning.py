import numpy as np

from dataeval.metrics.bias.metadata_binning import metadata_binning, user_defined_bin


class TestMDBinningUnit:
    def test_returns_array(self):
        factors = [0.1, 1.1, 1.2]
        bincounts = 2
        hist = user_defined_bin(factors, bincounts)
        assert type(hist) is np.ndarray


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

    def test_udb_regression(self):
        factors = [0.1, 1.1, 1.2]
        bincounts = 2
        hist = user_defined_bin(factors, bincounts)
        assert np.all(hist == [1, 2, 2])
