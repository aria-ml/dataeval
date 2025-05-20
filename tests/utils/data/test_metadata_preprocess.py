import numpy as np
import pytest

from dataeval.utils.data.metadata import merge
from tests.conftest import to_metadata


@pytest.mark.required
class TestMDPreprocessingUnit:
    def test_uneven_factor_lengths(self):
        labels = [0] * 10 + [1] * 10
        factors = {"factor1": ["a"] * 10, "factor2": ["b"] * 11}
        err_msg = "Number of metadata (2) does not match number of images (20)."
        with pytest.raises(Exception) as e:
            to_metadata(factors, labels)._bin()
        assert err_msg in str(e.value)

    def test_bad_factor_ref(self):
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["a"] * 5 + ["b"] * 5}
        continuous_bincounts = {"something_else": 2}
        err_msg = "The keys - {'something_else'} - are present in the `continuous_factor_bins` dictionary"
        with pytest.warns(UserWarning, match=err_msg):
            to_metadata(factors, labels, continuous_bincounts)._bin()

    def test_wrong_shape(self):
        labels = [[0], [1]]
        factors = {"factor1": [10, 20]}
        err_msg = "Labels must be a sequence of integers for image classification."
        with pytest.raises(TypeError) as e:
            to_metadata(factors, labels)._bin()
        assert err_msg in str(e.value)

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
            list(np.random.choice(2000, size=120000) / 1000),
            list(np.random.rand(100) * 100),
        ],
    )
    def test_discrete_without_bins(self, data_values):
        factors = {"data": data_values}
        labels = list(np.random.randint(5, size=len(data_values)))
        err_msg = "A user defined binning was not provided for data."
        with pytest.warns(UserWarning, match=err_msg):
            to_metadata(factors, labels)._bin()

    @pytest.mark.parametrize("factors", ({"a": [1, 2, 3], "b": [1, 2, 3]}, {"a": [1, 2, 3]}))
    @pytest.mark.parametrize("bincounts", ({"a": 1, "b": 1}, {"a": 1}, None))
    def test_exclude_raw_metadata_only(self, factors, bincounts):
        labels = [0, 0, 0]
        output = to_metadata(factors, labels, bincounts, exclude=["b"])
        assert "b" not in output.class_names

    @pytest.mark.parametrize(
        "factors, labels",
        [
            [[{"data1": [0, 1, 2, 3, 4], "id": 0}], np.repeat(np.arange(5), 3)],
            [[{"data1": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4], "id": 0}], np.arange(5)],
        ],
    )
    def test_label_length_mismatch(self, factors, labels):
        flat_factors = merge(factors)
        err_msg = f"Number of metadata (3) does not match number of images ({len(labels)})."
        with pytest.raises(ValueError) as e:
            to_metadata(flat_factors, labels)._bin()
        assert err_msg in str(e.value)


@pytest.mark.optional
class TestMDPreprocessingFunctional:
    def test_nbins(self):
        factors = {"data1": [0.1, 0.2, 0.3, 1.1, 1.2]}
        labels = [0, 0, 0, 0, 0]
        bincounts = {"data1": 2}
        output = to_metadata(factors, labels, bincounts)
        disc_factors = output.discretized_data
        assert len(np.unique(disc_factors)) == 2

    def test_bin_edges(self):
        factors = {"data1": [0.1, 0.2, 0.3, 1.1, 1.2]}
        bin_edges = {"data1": [-np.inf, 1, np.inf]}
        labels = [0, 0, 0, 0, 0]
        output = to_metadata(factors, labels, bin_edges)
        disc_factors = output.discretized_data
        assert len(np.unique(disc_factors)) == 2

    def test_mix_match(self):
        factors = {"data1": [-1.1, 0.2, 0.3, 1.1, 1.2], "data2": [-1.1, 0.2, 0.3, 1.1, 1.2]}
        labels = [0, 1, 2, 0, 1]
        bincounts = {"data1": 3, "data2": [-np.inf, 1, np.inf]}
        output = to_metadata(factors, labels, bincounts)
        disc_factors = output.discretized_data.T
        assert len(np.unique(disc_factors[0])) == 3
        assert len(np.unique(disc_factors[1])) == 2

    def test_one_bin(self):
        factors = {"data1": [0.1, 0.2, 0.3, 1.1, 1.2]}
        labels = [0, 0, 0, 0, 0]
        bincounts = {"data1": 1}
        output = to_metadata(factors, labels, bincounts)
        disc_factors = output.discretized_data
        assert len(np.unique(disc_factors)) == 1

    def test_over_specified(self):
        factors = {"data1": [0.1, 0.2, 0.3, 1.1, 1.2]}
        labels = [0, 0, 0, 0, 0]
        bincounts = {"data1": 100}
        output = to_metadata(factors, labels, bincounts)
        disc_factors = output.discretized_data
        assert len(np.unique(disc_factors)) == 5
