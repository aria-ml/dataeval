import numpy as np

from dataeval.metrics.bias.metadata_binning import metadata_binning


class TestMDBinningFunctional:
    def test_nbins(self):
        factors = [{"data1": [0.1, 1.1, 1.2]}]
        bincounts = {"data1": 2}
        output = metadata_binning(factors, bincounts)
        disc_factors = output.discrete
        assert len(np.unique(disc_factors["data1"])) == 2

    def test_bin_edges(self):
        factors = [{"data1": [0.1, 1.1, 1.2]}]
        bin_edges = {"data1": [-np.inf, 1, np.inf]}
        output = metadata_binning(factors, bin_edges)
        disc_factors = output.discrete
        assert len(np.unique(disc_factors["data1"])) == 2
