from unittest.mock import MagicMock

import numpy as np
import pytest

from dataeval.data._metadata import Metadata
from dataeval.evaluators.bias._parity import parity
from tests.conftest import to_metadata


@pytest.mark.required
class TestMDParityUnit:
    def test_warns_with_not_enough_frequency(self):
        labels = [0, 1]
        factors = {"factor1": [10, 20]}
        metadata = to_metadata(factors, labels)
        with pytest.warns(UserWarning):
            parity(metadata)

    def test_passes_with_enough_frequency(self):
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["foo"] * 10}
        metadata = to_metadata(factors, labels)
        parity(metadata)

    def test_to_dataframe(self):
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["foo"] * 10}
        metadata = to_metadata(factors, labels)
        df = parity(metadata).to_dataframe()
        assert df is not None

    def test_empty_metadata(self):
        mock_metadata = MagicMock(spec=Metadata)
        mock_metadata.factor_names = []
        with pytest.raises(ValueError):
            parity(mock_metadata)


class TestMDParityFunctional:
    def test_correlated_factors(self):
        """
        In this dataset, class and factor1 are perfectly correlated.
        This tests that the p-value<P-Value>` is less than 0.05, which
        corresponds to class and factor1 being highly correlated.
        """
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["a"] * 5 + ["b"] * 5}
        metadata = to_metadata(factors, labels)
        result = parity(metadata)

        # Checks that factor1 is highly correlated with class
        assert result.p_value[0] < 0.05

    def test_uncorrelated_factors(self):
        """
        This verifies that if the factor is homogeneous for the whole dataset,
        that chi2 and p correspond to factor1 being uncorrelated with class.
        """
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["foo"] * 10}
        metadata = to_metadata(factors, labels)
        result = parity(metadata)

        # Checks that factor1 is uncorrelated with class
        assert np.isclose(result.score[0], 0)
        assert np.isclose(result.p_value[0], 1)

    def test_quantized_factors(self):
        """
        This discretizes 'factor1' into having two values.
        This verifies that the '11' and '10' values get grouped together.
        """
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": [10] * 2 + [11] * 3 + [20] * 5}
        continuous_bincounts = {"factor1": 2}
        metadata = to_metadata(factors, labels, continuous_bincounts)
        result1 = parity(metadata)

        discrete_dataset = {"factor2": [10] * 5 + [20] * 5}
        metadata = to_metadata(discrete_dataset, labels)
        result2 = parity(metadata)

        # Checks that the test on the quantization continuous_dataset is
        # equivalent to the test on the discrete dataset discrete_dataset
        assert result1.score[0] == result2.score[0]
        assert result1.p_value[0] == result2.p_value[0]

    def test_overquantized_factors(self):
        """
        This quantizes factor1 to have only one value, so that the discretized
        factor1 is the same over the entire dataset.
        """
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": [10] * 2 + [11] * 3 + [20] * 5}
        continuous_bincounts = {"factor1": 1}
        metadata = to_metadata(factors, labels, continuous_bincounts)
        result = parity(metadata)

        # Checks if factor1 and class are perfectly uncorrelated
        assert np.isclose(result.score[0], 0)
        assert np.isclose(result.p_value[0], 1)

    def test_underquantized_has_low_freqs(self):
        """
        This quantizes factor1 such that there are large regions with bins
        that contain a small number of points.
        """
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": list(np.arange(10))}
        continuous_bincounts = {"factor1": 10}
        metadata = to_metadata(factors, labels, continuous_bincounts)

        # Looks for a warning that there are (class,factor1) pairs with too low frequency
        with pytest.warns(UserWarning):
            parity(metadata)

    def test_underquantized_has_repeated_low_freqs(self):
        """
        This quantizes factor1 such that there are large regions with bins
        that contain a small number of points.
        """
        labels = [0] * 5 + [1] * 5 + [0] * 5 + [1] * 5
        factors = {"factor1": list(np.arange(10)) + list(np.arange(10))}
        continuous_bincounts = {"factor1": 10}
        metadata = to_metadata(factors, labels, continuous_bincounts)

        # Looks for a warning that there are (class,factor1) pairs with too low frequency
        with pytest.warns(UserWarning):
            parity(metadata)
