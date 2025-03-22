import re
from itertools import compress
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.random import Generator

from dataeval.metadata._ood import (
    _calc_median_deviations,
    _combine_metadata,
    _compare_keys,
    _validate_factors_and_data,
    most_deviated_factors,
)


def mock_metadata(factor_names, discrete_data=None, cont_factor_names=None, continuous_data=None) -> MagicMock:
    """
    Creates a magic mock method that contains discrete and continuous data and factors
    but has no hard dependency on Metadata.
    """

    m = MagicMock()

    m.discrete_factor_names = factor_names
    count = len(factor_names)

    if cont_factor_names is not None:
        m.continuous_factor_names = cont_factor_names
        count += len(cont_factor_names)

    if discrete_data is not None:
        m.discrete_data = discrete_data
    if continuous_data is not None:
        m.continuous_data = continuous_data

    m.total_num_factors = count

    return m


def mock_ood_output(is_ood) -> MagicMock:
    """
    Creates a mock object with only is_ood as scores are not relevant to the calculations
    """

    m = MagicMock()
    m.is_ood = is_ood
    return m


@pytest.mark.required
class TestMetadataValidation:
    """
    Group functions used for validation of Metadata properties
    """

    @pytest.mark.parametrize("factors", (1, 10))
    @pytest.mark.parametrize("samples", (0, 1, 2))
    @patch("dataeval.metadata._ood._combine_metadata")
    def test_warn_on_insufficient_samples(self, mock, RNG, samples, factors):
        """
        Tests that a warning is raised when the reference set has less than 3 samples
        """

        ood = mock_ood_output(np.array([True]))
        # Samples are "shared" over discrete and continuous as they are considered different factors
        mock.return_value = ([], [RNG.random((samples, factors)), RNG.random((samples, factors))], [])

        # Neither have enough samples
        warn_msg = f"At least 3 reference metadata samples are needed, got {samples}"
        with pytest.warns(UserWarning, match=warn_msg):
            res = most_deviated_factors(
                metadata_1=MagicMock(),
                metadata_2=MagicMock(),
                ood=ood,
            )
            assert res == []

    @pytest.mark.parametrize(
        "lst",
        (
            ["a", "b", "c"],
            ["a", "b", "c", "e"],
            ["a", "b", "d", "c"],
            [],
        ),
    )
    def test_compare_invalid_keys(self, lst):
        """
        Test lists that differ from the reference list are found to be not identical
        """
        reference: list[str] = ["a", "b", "c", "d"]

        error_msg = f"Metadata keys must be identical, got {reference} and {lst}"

        with pytest.raises(ValueError) as exec_info:
            _compare_keys(reference, lst)

        assert str(exec_info.value) == error_msg

    @pytest.mark.parametrize("lst", (["a", "b"], ["a", "a"], []))
    def test_compare_valid_keys(self, lst):
        """Tests identical keys are not flagged"""

        _compare_keys(lst, lst)

    @pytest.mark.parametrize("length", (2, 4))
    @patch("dataeval.metadata._ood._combine_metadata")
    def test_error_ood_testmd_lengths(self, mock, length):
        """Tests the case where the ood mask differs from the length of the data lengths"""

        mock.return_value = ([], [np.ones((3, 3))], [np.ones((3, 1))])

        ood = mock_ood_output(np.array([True] * length))
        error_msg = f"ood and test metadata must have the same length, got {length} and 3 respectively."

        with pytest.raises(ValueError, match=error_msg):
            most_deviated_factors(MagicMock(), MagicMock(), ood)

    def test_error_disc_cont_samples(self, RNG: Generator):
        """
        Tests that discrete and continuous data with differing number of samples
        is not allowed as all samples should have every factor
        """

        ood = mock_ood_output(np.array([True, True, True]))

        # 3 and 4 samples conflict
        md = mock_metadata(
            factor_names=["a", "b"],
            discrete_data=RNG.random((3, 3)),
            cont_factor_names=["a", "b", "c"],
            continuous_data=RNG.random((4, 3)),
        )

        with pytest.raises(ValueError):
            most_deviated_factors(md, md, ood)


@pytest.mark.required
class TestDeviatedFactors:
    """
    Unit tests for most_deviated_factors function to check output shape and selection logic
    """

    @pytest.mark.parametrize("is_ood_2", (False, True))
    @pytest.mark.parametrize("is_ood_1", (False, True))
    @pytest.mark.parametrize("is_ood_0", (False, True))
    @patch("dataeval.metadata._ood._calc_median_deviations")
    def test_mask_selection(
        self,
        mock_calc_median_dev,
        is_ood_0,
        is_ood_1,
        is_ood_2,
        RNG: Generator,
    ):
        """Tests ood flags correctly select most deviated metadata factor"""

        ood_mask = np.array([is_ood_0, is_ood_1, is_ood_2], dtype=np.bool)
        ood = mock_ood_output(ood_mask)

        # 3 samples to reduce unnecessary warnings in pytest logs
        shape = (3, 3)

        devs = np.arange(9).reshape(shape)

        mock_calc_median_dev.return_value = devs

        factors = ["a", "b", "c"]
        result = most_deviated_factors(
            mock_metadata(factors, RNG.random(shape), cont_factor_names=[]),
            mock_metadata(factors, RNG.random(shape), cont_factor_names=[]),
            ood,
        )

        truths = np.sum(ood_mask)
        # Built-in list masking function
        expected_factors = compress(factors, ood_mask)
        # Get largest value for "true" ood factors
        expected_value = devs[ood_mask][-1] if truths else []

        assert len(result) == truths
        assert len(result) == len(list(expected_factors))

        for factor, value, res in zip(expected_factors, expected_value, result):
            assert factor == res[0]
            assert value == res[1]

    def test_valid_sample_factor_shapes(self):
        """Tests single test value runs with correct output shape and value"""

        ood = mock_ood_output(np.array([True, True, True]))
        m1 = mock_metadata(["a", "b", "c"], np.ones((3, 3)), [], [])
        result = most_deviated_factors(m1, m1, ood=ood)

        assert len(result) == 3
        assert result[0] == ("a", 0.0)  # zero deviation in all ones


@pytest.mark.required
class TestInternalFuncs:
    @pytest.mark.parametrize("samples_ref", (1, 5, 10, 100))
    @pytest.mark.parametrize("samples_tst", (1, 5, 10, 100))
    @pytest.mark.parametrize("factors", (1, 5, 10))
    def test_calc_median_deviations(self, samples_ref, samples_tst, factors):
        """Tests all combinations of 1-D and 2-D inputs including (1, F), (S, 1), and (1, 1)"""

        r = np.arange(samples_ref * factors).reshape(samples_ref, factors)
        t = np.arange(samples_tst * factors).reshape(samples_tst, factors)

        res = _calc_median_deviations(r, t)

        assert res.shape == (samples_tst, factors)
        assert not np.any(np.isnan(res))

    def test_error_combine_diff_num_factors(self):
        """Tests the quick check of total num factors equality"""

        error_msg = re.escape("Number of factors differs between metadata_1 (1) and metadata_2 (2)")

        m1 = mock_metadata(["a"])
        m2 = mock_metadata(["a", "b"])

        with pytest.raises(ValueError, match=error_msg):
            _combine_metadata(m1, m2)

    def test_no_factors(self):
        m1 = mock_metadata([], cont_factor_names=[])
        m2 = mock_metadata([], cont_factor_names=[])

        result = _combine_metadata(m1, m2)

        assert len(result) == 3

        for res in result:
            assert res == []

    @pytest.mark.parametrize(
        "dnames, cnames, expected_shape",
        [
            (["discrete"], [], (3, 1)),
            ([], ["cont"], (5, 1)),
        ],
    )
    def test_one_factor_type(self, dnames, cnames, expected_shape):
        """Tests that only data with factor names is returned"""

        m1 = mock_metadata(
            dnames, discrete_data=np.ones((3, 1)), cont_factor_names=cnames, continuous_data=np.ones((5, 1))
        )
        m2 = mock_metadata(
            dnames, discrete_data=np.ones((3, 1)), cont_factor_names=cnames, continuous_data=np.ones((5, 1))
        )

        result = _combine_metadata(m1, m2)

        assert len(result) == 3
        f, md1, md2 = result  # factor names, m1 data, m2 data

        assert f == dnames if dnames else cnames

        for md in [md1, md2]:
            assert len(md) == 1
            assert isinstance(md[0], np.ndarray)
            assert md[0].shape == expected_shape

    def test_both_factors(self):
        m1 = mock_metadata(
            ["disc"], discrete_data=np.ones((3, 1)), cont_factor_names=["cont"], continuous_data=np.ones((5, 1))
        )
        m2 = mock_metadata(
            ["disc"], discrete_data=np.ones((3, 1)), cont_factor_names=["cont"], continuous_data=np.ones((5, 1))
        )

        result = _combine_metadata(m1, m2)

        assert len(result) == 3
        f, md1, md2 = result  # factor names, m1 data, m2 data

        assert f == ["disc", "cont"]

        for md in [md1, md2]:
            assert len(md) == 2
            assert isinstance(md1[0], np.ndarray)
            assert md[0].shape == (3, 1)
            assert md[1].shape == (5, 1)


@pytest.mark.required
class TestUtils:
    """
    Tests the validation functions used by metadata utility functions to ensure Metadata data is valid.

    TODO: Move to separate file
    """

    @pytest.mark.parametrize("samples", (1, 3, 5))
    @pytest.mark.parametrize("factors", (0, 1, 3))
    def test_valid_factors_and_data(self, samples, factors):
        """Tests that only number of factors determines validity"""

        f = ["a"] * factors
        d = np.ones(shape=(samples, factors))

        _validate_factors_and_data(f, d)

    def test_invalid_factors_and_data(self):
        """Tests inequality of factors and data columns"""
        f = ["a", "b", "c"]
        d = np.ones((3, 1))  # (S, F)

        error_msg = "Factors and data have mismatched lengths. Got 3 and 1"

        with pytest.raises(ValueError, match=error_msg):
            _validate_factors_and_data(f, d)


@pytest.mark.optional
class TestFunctional:
    """
    With a more realistic number of samples, make sure that the values and shapes make logical sense
    """

    def test_big_data_with_noise(self, metadata_ref_big, metadata_tst_big):
        """Tests larger matrix with multiple samples and factors"""

        samples = len(metadata_ref_big.discrete_data)
        feature_names = metadata_ref_big.discrete_factor_names

        is_ood = np.array([True] * samples)
        ood = mock_ood_output(is_ood=is_ood)
        output = most_deviated_factors(metadata_ref_big, metadata_tst_big, ood)

        half = samples // 2

        assert {out[0] for out in output[:half]} == {feature_names[0]}
        assert {out[0] for out in output[half:]} == {feature_names[1]}
