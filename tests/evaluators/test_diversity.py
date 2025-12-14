from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from dataeval.data._metadata import Metadata
from dataeval.types import ExecutionMetadata

try:
    from matplotlib.figure import Figure
except ImportError:
    Figure = type(None)

from dataeval.evaluators.bias import Diversity
from tests.conftest import to_metadata


@pytest.fixture(scope="module")
def metadata_results():
    str_vals = ["a", "a", "a", "a", "b", "a", "a", "a", "b", "b"]
    cnt_vals = [0.63784, -0.86422, -0.1017, -1.95131, -0.08494, -1.02940, 0.07908, -0.31724, -1.45562, 1.03368]
    class_labels = ["dog", "dog", "dog", "cat", "dog", "cat", "dog", "dog", "dog", "cat"]
    md = {"var_cat": str_vals, "var_cnt": cnt_vals}
    return to_metadata(md, class_labels, {"var_cnt": 3})


@pytest.mark.required
class TestDiversityUnit:
    """Test the Diversity class interface"""

    def test_initialization_defaults(self):
        diversity_obj = Diversity()
        assert diversity_obj.method == "simpson"
        assert diversity_obj.threshold == 0.5

    def test_initialization_custom(self):
        diversity_obj = Diversity(method="shannon", threshold=0.6)
        assert diversity_obj.method == "shannon"
        assert diversity_obj.threshold == 0.6

    @pytest.mark.parametrize("met", ["Simpson", "ShANnOn"])
    def test_invalid_method(self, metadata_results, met):
        diversity_obj = Diversity(method=met)
        with pytest.raises(ValueError):
            diversity_obj.evaluate(metadata_results)

    @pytest.mark.parametrize("met", ["simpson", "shannon"])
    def test_range_of_values(self, metadata_results, met):
        diversity_obj = Diversity(method=met)
        result = diversity_obj.evaluate(metadata_results)

        # Check factors DataFrame
        assert isinstance(result.factors, pl.DataFrame)
        assert all(0 <= v <= 1 for v in result.factors["diversity_value"])

        # Check classwise DataFrame
        assert isinstance(result.classwise, pl.DataFrame)
        assert all(0 <= v <= 1 for v in result.classwise["diversity_value"])

    @pytest.mark.parametrize("met", ["simpson", "shannon"])
    def test_output_dtypes(self, metadata_results, met):
        diversity_obj = Diversity(method=met)
        result = diversity_obj.evaluate(metadata_results)

        # Check that outputs are DataFrames
        assert isinstance(result.factors, pl.DataFrame)
        assert isinstance(result.classwise, pl.DataFrame)

        # Check factors DataFrame schema
        assert set(result.factors.schema.keys()) == {
            "factor_name",
            "diversity_value",
            "is_low_diversity",
        }
        assert result.factors.schema["factor_name"].base_type() == pl.Categorical
        assert result.factors.schema["diversity_value"] == pl.Float64
        assert result.factors.schema["is_low_diversity"] == pl.Boolean

        # Check classwise DataFrame schema
        assert set(result.classwise.schema.keys()) == {
            "class_name",
            "factor_name",
            "diversity_value",
            "is_low_diversity",
        }
        assert result.classwise.schema["class_name"].base_type() == pl.Categorical
        assert result.classwise.schema["factor_name"].base_type() == pl.Categorical
        assert result.classwise.schema["diversity_value"] == pl.Float64
        assert result.classwise.schema["is_low_diversity"] == pl.Boolean

        assert isinstance(result.meta(), ExecutionMetadata)

    def test_empty_metadata(self):
        mock_metadata = MagicMock(spec=Metadata)
        mock_metadata.factor_names = []
        diversity_obj = Diversity()
        with pytest.raises(ValueError):
            diversity_obj.evaluate(mock_metadata)

    def test_metadata_stored(self, metadata_results):
        diversity_obj = Diversity()
        diversity_obj.evaluate(metadata_results)
        assert diversity_obj.metadata is not None
        assert isinstance(diversity_obj.metadata, Metadata)
        assert diversity_obj.metadata.factor_names == metadata_results.factor_names

    def test_threshold_parameters(self):
        """Test that custom thresholds affect bias detection"""
        metadata = to_metadata({"factor1": [1, 1, 1, 2, 2, 3]}, [0, 0, 0, 1, 1, 1], {})

        diversity_obj1 = Diversity(threshold=1.0)
        result1 = diversity_obj1.evaluate(metadata)

        diversity_obj2 = Diversity(threshold=0.0)
        result2 = diversity_obj2.evaluate(metadata)

        # Higher threshold should flag fewer items as low diversity
        low_div_1 = result1.factors.filter(pl.col("is_low_diversity")).height
        low_div_2 = result2.factors.filter(pl.col("is_low_diversity")).height
        assert low_div_1 >= low_div_2


@pytest.mark.optional
class TestDiversityFunctional:
    """Test functional behavior of Diversity class"""

    @pytest.mark.parametrize(
        "metadata, expected_diversity, expected_classwise",
        [
            (
                to_metadata({"factor1": [5, 5, 5, 6, 6, 6]}, [0, 0, 0, 1, 1, 1], {}),
                1.0,
                0.0,
            ),
            (
                to_metadata({"factor1": [5.1, 4.9, 4.9, 6.1, 6.2, 6.1]}, [0, 0, 0, 1, 1, 1], {"factor1": 2}),
                1.0,
                0.0,
            ),
            (
                to_metadata({"factor1": [5, 5, 5, 5, 5, 5]}, [0, 0, 0, 1, 1, 1], {}),
                0.0,
                0.0,
            ),
        ],
    )
    def test_simpson(self, metadata, expected_diversity, expected_classwise):
        diversity_obj = Diversity(method="simpson")
        result = diversity_obj.evaluate(metadata)

        # Check factors DataFrame
        diversity_values = result.factors["diversity_value"].to_numpy()
        np.testing.assert_array_almost_equal(diversity_values, [expected_diversity])

        # Check classwise DataFrame
        classwise_values = result.classwise["diversity_value"].to_numpy()
        for val in classwise_values:
            np.testing.assert_almost_equal(val, expected_classwise)

    @pytest.mark.parametrize(
        "metadata, expected_diversity, expected_classwise",
        [
            (
                to_metadata({"factor1": [5, 5, 5, 6, 6, 6]}, [0, 0, 0, 1, 1, 1], {}),
                1.0,
                0.0,
            ),
            (
                to_metadata({"factor1": [5.1, 4.9, 4.9, 6.1, 6.2, 6.1]}, [0, 0, 0, 1, 1, 1], {"factor1": 2}),
                1.0,
                0.0,
            ),
            (
                to_metadata({"factor1": [5, 5, 5, 5, 5, 5]}, [0, 0, 0, 1, 1, 1], {}),
                0.0,
                0.0,
            ),
        ],
    )
    def test_shannon(self, metadata, expected_diversity, expected_classwise):
        diversity_obj = Diversity(method="shannon")
        result = diversity_obj.evaluate(metadata)

        # Check factors DataFrame
        diversity_values = result.factors["diversity_value"].to_numpy()
        np.testing.assert_array_almost_equal(diversity_values, [expected_diversity])

        # Check classwise DataFrame
        classwise_values = result.classwise["diversity_value"].to_numpy()
        for val in classwise_values:
            np.testing.assert_almost_equal(val, expected_classwise)
