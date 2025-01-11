import numpy as np
import pytest

from dataeval.utils.metadata import (
    CONTINUOUS_MIN_SAMPLE_SIZE,
    _bin_data,
    _digitize_data,
    _is_continuous,
    _is_metadata_dict_of_dicts,
    preprocess,
)


class TestMDPreprocessingUnit:
    def test_uneven_factor_lengths(self):
        labels = [0] * 5 + [1] * 5
        factors = [{"factor1": ["a"] * 10, "factor2": ["b"] * 11}]
        err_msg = """[UserWarning("Dropping nested list found in '('factor2',)'.")]"""
        with pytest.warns(UserWarning, match=err_msg):
            preprocess(factors, labels)

    def test_bad_factor_ref(self):
        labels = [0] * 5 + [1] * 5
        factors = [{"factor1": ["a"] * 5 + ["b"] * 5}]
        continuous_bincounts = {"something_else": 2}
        err_msg = "The keys - {'something_else'} - are present in the `continuous_factor_bins` dictionary "
        with pytest.raises(KeyError) as e:
            preprocess(factors, labels, continuous_bincounts)
        assert err_msg in str(e.value)

    def test_wrong_shape(self):
        labels = [[0], [1]]
        factors = [{"factor1": [10, 20]}]
        err_msg = "Got class labels with 2-dimensional shape (2, 1), but expected a 1-dimensional array."
        with pytest.raises(ValueError) as e:
            preprocess(factors, labels)
        assert err_msg in str(e.value)

    def test_doesnt_modify_input(self):
        factors = [{"data1": [0.1, 0.2, 0.3]}]
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
        factors = [{"data": data_values}]
        labels = list(np.random.randint(5, size=len(data_values)))
        err_msg = "A user defined binning was not provided for data."
        with pytest.warns(UserWarning, match=err_msg):
            preprocess(factors, labels)

    def test_exclude_raw_metadata_only(self):
        factors = [{"data1": [0.1, 0.2, 0.3], "data2": [1, 2, 3]}]
        labels = [0, 0, 0]
        bincounts = {"data1": 1}
        output = preprocess(factors, labels, bincounts, exclude=["data2"])
        assert "data2" not in output.class_names

    def test_exclude_raw_metadata_and_bincounts(self):
        factors = [{"data1": [0.1, 0.2, 0.3], "data2": [1, 2, 3]}]
        labels = [0, 0, 0]
        bincounts = {"data1": 1, "data2": 1}
        output = preprocess(factors, labels, bincounts, exclude=["data2"])
        assert "data2" not in output.class_names

    def test_is_metadata_dict_of_dicts(self):
        assert not _is_metadata_dict_of_dicts({"a": 1})
        assert not _is_metadata_dict_of_dicts({"a": [1], "b": 1})


class TestMDPreprocessingFunctional:
    def test_nbins(self):
        factors = [{"data1": [0.1, 0.2, 0.3, 1.1, 1.2]}]
        labels = [0, 0, 0, 0, 0]
        bincounts = {"data1": 2}
        output = preprocess(factors, labels, bincounts)
        disc_factors = output.discrete_data
        assert len(np.unique(disc_factors)) == 2

    def test_bin_edges(self):
        factors = [{"data1": [0.1, 0.2, 0.3, 1.1, 1.2]}]
        bin_edges = {"data1": [-np.inf, 1, np.inf]}
        labels = [0, 0, 0, 0, 0]
        output = preprocess(factors, labels, bin_edges)
        disc_factors = output.discrete_data
        assert len(np.unique(disc_factors)) == 2

    def test_mix_match(self):
        factors = [{"data1": [-1.1, 0.2, 0.3, 1.1, 1.2], "data2": [-1.1, 0.2, 0.3, 1.1, 1.2]}]
        labels = [0, 0, 0, 0, 0]
        bincounts = {"data1": 3, "data2": [-np.inf, 1, np.inf]}
        output = preprocess(factors, labels, bincounts)
        disc_factors = output.discrete_data.T
        assert len(np.unique(disc_factors[0])) == 3
        assert len(np.unique(disc_factors[1])) == 2

    def test_one_bin(self):
        factors = [{"data1": [0.1, 0.2, 0.3, 1.1, 1.2]}]
        labels = [0, 0, 0, 0, 0]
        bincounts = {"data1": 1}
        output = preprocess(factors, labels, bincounts)
        disc_factors = output.discrete_data
        assert len(np.unique(disc_factors)) == 1

    def test_over_specified(self):
        factors = [{"data1": [0.1, 0.2, 0.3, 1.1, 1.2]}]
        labels = [0, 0, 0, 0, 0]
        bincounts = {"data1": 100}
        output = preprocess(factors, labels, bincounts)
        disc_factors = output.discrete_data
        assert len(np.unique(disc_factors)) == 5


class TestUserDefinedBinUnit:
    def test_nbins_returns_array(self):
        factors = [0.1, 1.1, 1.2]
        bincounts = 2
        hist = _digitize_data(factors, bincounts)
        assert type(hist) is np.ndarray

    def test_bin_edges_returns_array(self):
        factors = [0.1, 1.1, 1.2]
        bin_edges = [-np.inf, 1, np.inf]
        hist = _digitize_data(factors, bin_edges)
        assert type(hist) is np.ndarray

    def test_crashes_with_negative_nbins(self):
        factors = [0.1, 1.1, 1.2]
        bincounts = -10
        with pytest.raises(ValueError):
            _digitize_data(factors, bincounts)

    def test_crashes_with_wrong_order(self):
        factors = [0.1, 1.1, 1.2]
        bin_edges = [np.inf, 1, 2]
        with pytest.raises(ValueError):
            _digitize_data(factors, bin_edges)

    def test_mixed_type(self):
        factors = [1, "a", 4.0]
        bins = 3
        err_msg = "Encountered a data value with non-numeric type when digitizing a factor."
        with pytest.raises(TypeError) as e:
            _digitize_data(factors, bins)
        assert err_msg in str(e.value)


class TestUserDefinedBinFunctional:
    def test_udb_regression_nbins(self):
        factors = [0.1, 1.1, 1.2]
        bincounts = 2
        hist = _digitize_data(factors, bincounts)
        assert np.all(hist == [1, 2, 2])

    def test_udb_regression_bin_edges(self):
        factors = [0.1, 1.1, 1.2]
        bin_edges = [-np.inf, 1, np.inf]
        hist = _digitize_data(factors, bin_edges)
        assert np.all(hist == [1, 2, 2])

    def test_udb_regression_flipped_bin_edges(self):
        factors = [0.1, 1.1, 1.2]
        bin_edges = [np.inf, 1, -np.inf]
        hist = _digitize_data(factors, bin_edges)
        assert np.all(hist == [2, 1, 1])

    def test_narrow_bin_edges(self):
        factors = [0.1, 1.1, 1.5]
        bin_edges = [-10, 1, 1.2]
        hist = _digitize_data(factors, bin_edges)
        assert np.all(hist == [1, 2, 3])


class TestBinningUnit:
    @pytest.mark.parametrize(
        "method, data, expected_result",
        [
            ("uniform_width", np.array([0, 4, 3, 5, 6, 8] * 300), 6),
            ("uniform_width", np.concatenate([np.arange(2), np.arange(140, 1500)]), 10),
            # ("uniform_count", np.array([0, 4, 3, 5, 6, 8] * 10 + [5] * 30), 6), # BROKEN IN NUMPY 2.1+
            ("uniform_count", np.array([0, 4, 3, 5, 6, 9] * 10 + [5] * 30), 6),
        ],
    )
    def test_binning_method(self, method, data, expected_result):
        output = _bin_data(data, method)
        assert np.unique(output).size == expected_result

    def test_clusters_warn(self):
        data = np.array([0, 4, 3, 5, 6, 8] * 15)
        err_msg = "Binning by clusters is currently unavailable until changes to the clustering function go through."
        with pytest.warns(UserWarning, match=err_msg):
            output = _bin_data(data, "clusters")
        assert np.unique(output).size == 6


class TestIsContinuousUnit:
    @pytest.mark.parametrize(
        "data, repeats",
        [
            (np.array([0, 4, 3, 5, 6, 8] * 15), np.arange(15 * 6)),
            (np.array([0, 1, 9, 4, 3, 5, 2, 7, 8] * 10), np.array([0, 4, 3, 5, 6, 8] * 15)),
            (
                np.concatenate([np.repeat(val, 3) for val in range(20)]),
                np.concatenate([np.repeat(val, 2) for val in range(20)]),
            ),
            (
                np.concatenate(
                    [
                        np.repeat(val, 3)
                        for val in [0, 5, 13, 18, 2, 14, 1, 19, 16, 7, 15, 17, 4, 9, 10, 8, 12, 6, 11, 3]
                    ]
                ),
                np.concatenate(
                    [
                        np.repeat(val, 3)
                        for val in [0, 5, 13, 18, 2, 14, 1, 19, 16, 7, 15, 17, 4, 9, 10, 8, 12, 6, 11, 3]
                    ]
                ),
            ),
        ],
    )
    def test_is_continuous_repeats(self, data, repeats):
        _, image_unsorted = np.unique(repeats, return_index=True)
        image_indicies = np.sort(image_unsorted)
        output = _is_continuous(data, image_indicies)
        assert output is not True

    def test_is_coninuous_warning(self):
        data = np.array([0, 4, 3, 5, 6, 8] * 15)
        repeats = np.array([0, 4, 3, 5, 6, 8] * 15)
        _, image_unsorted = np.unique(repeats, return_index=True)
        image_indicies = np.sort(image_unsorted)
        warn_msg = (
            f"[UserWarning('All samples look discrete with so few data points (< {CONTINUOUS_MIN_SAMPLE_SIZE})')]"
        )
        with pytest.warns(UserWarning, match=warn_msg):
            output = _is_continuous(data, image_indicies)
        assert output is not True
