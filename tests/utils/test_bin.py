import numpy as np
import pytest

from dataeval.utils._bin import CONTINUOUS_MIN_SAMPLE_SIZE, bin_data, digitize_data, is_continuous


@pytest.mark.required
class TestUserDefinedBinUnit:
    def test_nbins_returns_array(self):
        factors = [0.1, 1.1, 1.2]
        bincounts = 2
        hist = digitize_data(factors, bincounts)
        assert type(hist) is np.ndarray

    def test_bin_edges_returns_array(self):
        factors = [0.1, 1.1, 1.2]
        bin_edges = [-np.inf, 1, np.inf]
        hist = digitize_data(factors, bin_edges)
        assert type(hist) is np.ndarray

    def test_crashes_with_negative_nbins(self):
        factors = [0.1, 1.1, 1.2]
        bincounts = -10
        with pytest.raises(ValueError):
            digitize_data(factors, bincounts)

    def test_crashes_with_wrong_order(self):
        factors = [0.1, 1.1, 1.2]
        bin_edges = [np.inf, 1, 2]
        with pytest.raises(ValueError):
            digitize_data(factors, bin_edges)

    def test_mixed_type(self):
        factors = [1, "a", 4.0]
        bins = 3
        err_msg = "Encountered a data value with non-numeric type when digitizing a factor."
        with pytest.raises(TypeError) as e:
            digitize_data(factors, bins)
        assert err_msg in str(e.value)


@pytest.mark.optional
class TestUserDefinedBinFunctional:
    def test_udb_regression_nbins(self):
        factors = [0.1, 1.1, 1.2]
        bincounts = 2
        hist = digitize_data(factors, bincounts)
        assert np.all(hist == [1, 2, 2])

    def test_udb_regression_bin_edges(self):
        factors = [0.1, 1.1, 1.2]
        bin_edges = [-np.inf, 1, np.inf]
        hist = digitize_data(factors, bin_edges)
        assert np.all(hist == [1, 2, 2])

    def test_udb_regression_flipped_bin_edges(self):
        factors = [0.1, 1.1, 1.2]
        bin_edges = [np.inf, 1, -np.inf]
        hist = digitize_data(factors, bin_edges)
        assert np.all(hist == [2, 1, 1])

    def test_narrow_bin_edges(self):
        factors = [0.1, 1.1, 1.5]
        bin_edges = [-10, 1, 1.2]
        hist = digitize_data(factors, bin_edges)
        assert np.all(hist == [1, 2, 3])


@pytest.mark.required
class TestBinningUnit:
    @pytest.mark.parametrize(
        "method, data, expected_result",
        [
            ("uniform_width", np.array([0, 4, 8, 5, 6, 15] * 300), 6),
            ("uniform_width", np.concatenate([np.arange(2), np.arange(140, 1500)]), 10),
            # ("uniform_count", np.array([0, 4, 3, 5, 6, 8] * 10 + [5] * 30), 6), # BROKEN IN NUMPY 2.1+
            ("uniform_count", np.array([0, 4, 8, 5, 6, 15] * 10 + [5] * 30), 6),
            ("clusters", np.array([0, 4, 8, 5, 6, 15] * 300), 5),
        ],
    )
    def test_binning_method(self, method, data, expected_result):
        output = bin_data(data, method)
        unq, vals = np.unique(output, return_inverse=True)
        print(unq)
        print(data[:20])
        print(vals[:20])
        assert np.unique(output).size == expected_result


@pytest.mark.required
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
        image_indices = np.sort(image_unsorted)
        output = is_continuous(data, image_indices)
        assert output is not True

    def test_is_coninuous_warning(self):
        data = np.array([0, 4, 3, 5, 6, 8] * 15)
        repeats = np.array([0, 4, 3, 5, 6, 8] * 15)
        _, image_unsorted = np.unique(repeats, return_index=True)
        image_indices = np.sort(image_unsorted)
        warn_msg = (
            f"[UserWarning('All samples look discrete with so few data points (< {CONTINUOUS_MIN_SAMPLE_SIZE})')]"
        )
        with pytest.warns(UserWarning, match=warn_msg):
            output = is_continuous(data, image_indices)
        assert output is not True
