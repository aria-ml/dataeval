import logging

import numpy as np
import polars as pl
import pytest

from dataeval._metadata import Metadata
from dataeval.bias import Parity
from tests.conftest import MockMetadata, to_metadata


@pytest.mark.required
class TestParityUnit:
    """Test the Parity class interface"""

    def test_initialization_defaults(self):
        parity_obj = Parity()
        assert parity_obj.score_threshold == 0.3
        assert parity_obj.p_value_threshold == 0.05

    def test_initialization_custom(self):
        parity_obj = Parity(score_threshold=0.4, p_value_threshold=0.01)
        assert parity_obj.score_threshold == 0.4
        assert parity_obj.p_value_threshold == 0.01

    def test_warns_with_not_enough_frequency(self, caplog):
        labels = [0, 1]
        factors = {"factor1": [10, 20]}
        metadata = to_metadata(factors, labels)
        parity_obj = Parity()
        with caplog.at_level(logging.WARNING):
            parity_obj.evaluate(metadata)
        assert len(caplog.text) > 0

    def test_passes_with_enough_frequency(self):
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["foo"] * 10}
        metadata = to_metadata(factors, labels)
        parity_obj = Parity()
        result = parity_obj.evaluate(metadata)
        assert isinstance(result.factors, pl.DataFrame)

    def test_output_is_dataframe(self):
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["foo"] * 10}
        metadata = to_metadata(factors, labels)
        parity_obj = Parity()
        result = parity_obj.evaluate(metadata)

        # Check that output is a DataFrame
        assert isinstance(result.factors, pl.DataFrame)

        # Check schema
        assert set(result.factors.schema.keys()) == {
            "factor_name",
            "score",
            "p_value",
            "is_correlated",
            "has_insufficient_data",
        }
        assert result.factors.schema["factor_name"].base_type() == pl.Categorical
        assert result.factors.schema["score"] == pl.Float64
        assert result.factors.schema["p_value"] == pl.Float64
        assert result.factors.schema["is_correlated"] == pl.Boolean
        assert result.factors.schema["has_insufficient_data"] == pl.Boolean

    def test_empty_metadata(self):
        mock_metadata = MockMetadata(
            class_labels=np.array([], dtype=np.intp),
            factor_data=np.array([], dtype=np.int64),
            factor_names=[],
            is_discrete=[],
            index2label={},
        )
        parity_obj = Parity()
        with pytest.raises(ValueError):
            parity_obj.evaluate(mock_metadata)

    def test_metadata_stored(self):
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["foo"] * 10}
        metadata = to_metadata(factors, labels)
        parity_obj = Parity()
        parity_obj.evaluate(metadata)
        assert parity_obj.metadata is not None
        assert isinstance(parity_obj.metadata, Metadata)
        assert parity_obj.metadata.factor_names == metadata.factor_names

    def test_threshold_parameters(self):
        """Test that custom thresholds affect correlation detection"""
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["a"] * 5 + ["b"] * 5}
        metadata = to_metadata(factors, labels)

        parity_obj1 = Parity(score_threshold=0.1, p_value_threshold=0.5)
        result1 = parity_obj1.evaluate(metadata)

        parity_obj2 = Parity(score_threshold=0.9, p_value_threshold=0.01)
        result2 = parity_obj2.evaluate(metadata)

        # Lower thresholds should detect more correlated factors (or equal)
        correlated_1 = result1.factors.filter(pl.col("is_correlated")).height
        correlated_2 = result2.factors.filter(pl.col("is_correlated")).height
        assert correlated_1 >= correlated_2


class TestParityFunctional:
    """Test functional behavior of Parity class"""

    def test_correlated_factors(self):
        """
        In this dataset, class and factor1 are perfectly correlated.
        This tests that the p-value is less than 0.05, which
        corresponds to class and factor1 being highly correlated.
        """
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["a"] * 5 + ["b"] * 5}
        metadata = to_metadata(factors, labels)
        parity_obj = Parity()
        result = parity_obj.evaluate(metadata)

        # Checks that factor1 is highly correlated with class
        p_value = result.factors.filter(pl.col("factor_name") == "factor1")["p_value"][0]
        assert p_value < 0.05

    def test_uncorrelated_factors(self):
        """
        This verifies that if the factor is homogeneous for the whole dataset,
        that chi2 and p correspond to factor1 being uncorrelated with class.
        """
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["foo"] * 10}
        metadata = to_metadata(factors, labels)
        parity_obj = Parity()
        result = parity_obj.evaluate(metadata)

        # Checks that factor1 is uncorrelated with class
        factor_row = result.factors.filter(pl.col("factor_name") == "factor1")
        score = factor_row["score"][0]
        p_value = factor_row["p_value"][0]

        assert np.isclose(score, 0)
        assert np.isclose(p_value, 1)

    def test_quantized_factors(self):
        """
        This discretizes 'factor1' into having two values.
        This verifies that the '11' and '10' values get grouped together.
        """
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": [10] * 2 + [11] * 3 + [20] * 5}
        continuous_bincounts = {"factor1": 2}
        metadata = to_metadata(factors, labels, continuous_bincounts)
        parity_obj = Parity()
        result1 = parity_obj.evaluate(metadata)

        discrete_dataset = {"factor2": [10] * 5 + [20] * 5}
        metadata = to_metadata(discrete_dataset, labels)
        result2 = parity_obj.evaluate(metadata)

        # Checks that the test on the quantization continuous_dataset is
        # equivalent to the test on the discrete dataset discrete_dataset
        score1 = result1.factors["score"][0]
        p_value1 = result1.factors["p_value"][0]
        score2 = result2.factors["score"][0]
        p_value2 = result2.factors["p_value"][0]

        assert score1 == score2
        assert p_value1 == p_value2

    def test_overquantized_factors(self):
        """
        This quantizes factor1 to have only one value, so that the discretized
        factor1 is the same over the entire dataset.
        """
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": [10] * 2 + [11] * 3 + [20] * 5}
        continuous_bincounts = {"factor1": 1}
        metadata = to_metadata(factors, labels, continuous_bincounts)
        parity_obj = Parity()
        result = parity_obj.evaluate(metadata)

        # Checks if factor1 and class are perfectly uncorrelated
        score = result.factors["score"][0]
        p_value = result.factors["p_value"][0]

        assert np.isclose(score, 0)
        assert np.isclose(p_value, 1)

    def test_underquantized_has_low_freqs(self, caplog):
        """
        This quantizes factor1 such that there are large regions with bins
        that contain a small number of points.
        """
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": list(np.arange(10))}
        continuous_bincounts = {"factor1": 10}
        metadata = to_metadata(factors, labels, continuous_bincounts)
        parity_obj = Parity()

        # Looks for a warning that there are (class,factor1) pairs with too low frequency
        with caplog.at_level(logging.WARNING):
            result = parity_obj.evaluate(metadata)
        assert len(caplog.text) > 0

        # Check that has_insufficient_data flag is set
        has_insuff = result.factors["has_insufficient_data"][0]
        assert has_insuff is True

    def test_underquantized_has_repeated_low_freqs(self, caplog):
        """
        This quantizes factor1 such that there are large regions with bins
        that contain a small number of points.
        """
        labels = [0] * 5 + [1] * 5 + [0] * 5 + [1] * 5
        factors = {"factor1": list(np.arange(10)) + list(np.arange(10))}
        continuous_bincounts = {"factor1": 10}
        metadata = to_metadata(factors, labels, continuous_bincounts)
        parity_obj = Parity()

        # Looks for a warning that there are (class,factor1) pairs with too low frequency
        with caplog.at_level(logging.WARNING):
            result = parity_obj.evaluate(metadata)
        assert len(caplog.text) > 0

        # Check that has_insufficient_data flag is set
        has_insuff = result.factors["has_insufficient_data"][0]
        assert has_insuff is True
