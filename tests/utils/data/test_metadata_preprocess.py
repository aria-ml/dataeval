import numpy as np
import pytest

from dataeval.utils.data.metadata import _is_metadata_dict_of_dicts, merge
from tests.conftest import preprocess


@pytest.mark.required
class TestMDPreprocessingUnit:
    def test_uneven_factor_lengths(self):
        labels = [0] * 10 + [1] * 10
        factors = {"factor1": ["a"] * 10, "factor2": ["b"] * 11}
        err_msg = "The lists/arrays in the metadata dict have varying lengths."
        with pytest.raises(ValueError) as e:
            preprocess(factors, labels)._process()
        assert err_msg in str(e.value)

    def test_bad_factor_ref(self):
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["a"] * 5 + ["b"] * 5}
        continuous_bincounts = {"something_else": 2}
        err_msg = "The keys - {'something_else'} - are present in the `continuous_factor_bins` dictionary "
        with pytest.raises(KeyError) as e:
            preprocess(factors, labels, continuous_bincounts)._process()
        assert err_msg in str(e.value)

    def test_wrong_shape(self):
        labels = [[0], [1]]
        factors = {"factor1": [10, 20]}
        err_msg = "Got class labels with 2-dimensional shape (2, 1), but expected a 1-dimensional array."
        with pytest.raises(ValueError) as e:
            preprocess(factors, labels)._process()
        assert err_msg in str(e.value)

    def test_doesnt_modify_input(self):
        factors = {"data1": [0.1, 0.2, 0.3]}
        labels = [0, 0, 0]
        bincounts = {"data1": 1}
        output = preprocess(factors, labels, bincounts)
        if output.continuous_data is not None:
            cont_factors = output.continuous_data.T[0]
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
            preprocess(factors, labels)._process()

    @pytest.mark.parametrize("factors", ({"a": [1, 2, 3], "b": [1, 2, 3]}, {"a": [1, 2, 3]}))
    @pytest.mark.parametrize("bincounts", ({"a": 1, "b": 1}, {"a": 1}, None))
    def test_exclude_raw_metadata_only(self, factors, bincounts):
        labels = [0, 0, 0]
        output = preprocess(factors, labels, bincounts, exclude=["b"])
        assert "b" not in output.class_names

    def test_is_metadata_dict_of_dicts(self):
        assert not _is_metadata_dict_of_dicts({"a": 1})
        assert not _is_metadata_dict_of_dicts({"a": [1], "b": 1})

    @pytest.mark.parametrize(
        "factors, labels",
        [
            [[{"data1": [0, 1, 2, 3, 4], "id": 0}], np.repeat(np.arange(5), 3)],
            [[{"data1": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4], "id": 0}], np.arange(5)],
        ],
    )
    def test_label_length_mismatch(self, factors, labels):
        flat_factors = merge(factors)
        err_msg = f"The length of the label array {len(labels)} is not the same as"
        with pytest.raises(ValueError) as e:
            preprocess(flat_factors, labels)._process()
        assert err_msg in str(e.value)


@pytest.mark.optional
class TestMDPreprocessingFunctional:
    def test_nbins(self):
        factors = {"data1": [0.1, 0.2, 0.3, 1.1, 1.2]}
        labels = [0, 0, 0, 0, 0]
        bincounts = {"data1": 2}
        output = preprocess(factors, labels, bincounts)
        disc_factors = output.discrete_data
        assert len(np.unique(disc_factors)) == 2

    def test_bin_edges(self):
        factors = {"data1": [0.1, 0.2, 0.3, 1.1, 1.2]}
        bin_edges = {"data1": [-np.inf, 1, np.inf]}
        labels = [0, 0, 0, 0, 0]
        output = preprocess(factors, labels, bin_edges)
        disc_factors = output.discrete_data
        assert len(np.unique(disc_factors)) == 2

    def test_mix_match(self):
        factors = {"data1": [-1.1, 0.2, 0.3, 1.1, 1.2], "data2": [-1.1, 0.2, 0.3, 1.1, 1.2]}
        labels = [0, 1, 2, 0, 1]
        bincounts = {"data1": 3, "data2": [-np.inf, 1, np.inf]}
        output = preprocess(factors, labels, bincounts)
        disc_factors = output.discrete_data.T
        assert len(np.unique(disc_factors[0])) == 3
        assert len(np.unique(disc_factors[1])) == 2

    def test_one_bin(self):
        factors = {"data1": [0.1, 0.2, 0.3, 1.1, 1.2]}
        labels = [0, 0, 0, 0, 0]
        bincounts = {"data1": 1}
        output = preprocess(factors, labels, bincounts)
        disc_factors = output.discrete_data
        assert len(np.unique(disc_factors)) == 1

    def test_over_specified(self):
        factors = {"data1": [0.1, 0.2, 0.3, 1.1, 1.2]}
        labels = [0, 0, 0, 0, 0]
        bincounts = {"data1": 100}
        output = preprocess(factors, labels, bincounts)
        disc_factors = output.discrete_data
        assert len(np.unique(disc_factors)) == 5
