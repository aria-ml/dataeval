import copy

import numpy as np
import polars as pl
import pytest

from dataeval._metadata import Metadata
from dataeval.bias._balance import Balance
from tests.conftest import MockMetadata, to_metadata


@pytest.fixture(scope="module")
def metadata_results():
    str_vals = ["b", "b", "b", "b", "b", "a", "a", "b", "a", "b", "b", "a"]
    cnt_vals = [
        -0.54425898,
        -0.31630016,
        0.41163054,
        1.04251337,
        -0.12853466,
        1.36646347,
        -0.66519467,
        0.35151007,
        0.90347018,
        0.0940123,
        -0.74349925,
        -0.92172538,
    ]
    cat_vals = [1.1, 1.1, 0.1, 0.1, 1.1, 0.1, 1.1, 0.1, 0.1, 1.1, 1.1, 0.1]
    class_labels = ["dog", "dog", "dog", "cat", "dog", "cat", "dog", "dog", "dog", "cat", "cat", "cat"]
    md = {"var_cat": str_vals, "var_cnt": cnt_vals, "var_float_cat": cat_vals}
    return to_metadata(md, class_labels, {"var_cnt": 3, "var_float_cat": 2})


@pytest.fixture(scope="module")
def mismatch_metadata():
    raw_metadata = {"factor1": list(range(10)), "factor2": list(range(10)), "factor3": list(range(10))}
    class_labels = [1] * 10
    continuous_bins = {"factor1": 5, "factor2": 5, "factor3": 5}
    return to_metadata(raw_metadata, class_labels, continuous_bins)


@pytest.fixture(scope="module")
def simple_metadata():
    raw_metadata = {"factor1": [1] * 100 + [2] * 100, "factor2": [1] * 100 + [2] * 100}
    class_labels = [1] * 100 + [2] * 100
    return to_metadata(raw_metadata, class_labels)


@pytest.mark.required
class TestBalanceUnit:
    """Test the Balance class interface"""

    def test_initialization_defaults(self):
        balance_obj = Balance()
        assert balance_obj.num_neighbors == 5
        assert balance_obj.class_imbalance_threshold == 0.3
        assert balance_obj.factor_correlation_threshold == 0.5

    def test_initialization_custom(self):
        balance_obj = Balance(num_neighbors=10, class_imbalance_threshold=0.4, factor_correlation_threshold=0.6)
        assert balance_obj.num_neighbors == 10
        assert balance_obj.class_imbalance_threshold == 0.4
        assert balance_obj.factor_correlation_threshold == 0.6

    def test_empty_metadata(self):
        mock_metadata = MockMetadata(
            class_labels=np.array([], dtype=np.intp),
            factor_data=np.array([], dtype=np.int64),
            factor_names=[],
            is_discrete=[],
            index2label={},
        )
        balance_obj = Balance()
        with pytest.raises(ValueError):
            balance_obj.evaluate(mock_metadata)

    def test_metadata_stored(self, metadata_results):
        balance_obj = Balance()
        balance_obj.evaluate(metadata_results)
        assert balance_obj.metadata is not None
        assert isinstance(balance_obj.metadata, Metadata)
        assert balance_obj.metadata.factor_names == metadata_results.factor_names

    def test_threshold_parameters(self, simple_metadata):
        """Test that custom thresholds affect the output"""
        balance_obj1 = Balance(class_imbalance_threshold=0.1, factor_correlation_threshold=0.1)
        result1 = balance_obj1.evaluate(simple_metadata)

        balance_obj2 = Balance(class_imbalance_threshold=0.9, factor_correlation_threshold=0.9)
        result2 = balance_obj2.evaluate(simple_metadata)

        # Lower thresholds should detect more issues (or equal)
        imbalanced_1 = result1.classwise.filter(pl.col("is_imbalanced")).height
        imbalanced_2 = result2.classwise.filter(pl.col("is_imbalanced")).height
        assert imbalanced_1 >= imbalanced_2

        correlated_1 = result1.factors.filter(pl.col("is_correlated")).height
        correlated_2 = result2.factors.filter(pl.col("is_correlated")).height
        assert correlated_1 >= correlated_2

    def test_correct_dataframe_shapes(self, metadata_results):
        metadata = copy.deepcopy(metadata_results)
        metadata.exclude = []
        num_factors = len(metadata.factor_names)
        num_classes = len(np.unique(metadata.class_labels))

        balance_obj = Balance()
        result = balance_obj.evaluate(metadata)

        # Check balance DataFrame
        assert isinstance(result.balance, pl.DataFrame)
        # balance includes class_label + metadata factors
        assert result.balance.height == num_factors + 1

        # Check balance DataFrame schema
        assert set(result.balance.schema.keys()) == {
            "factor_name",
            "mi_value",
        }
        assert result.balance.schema["factor_name"].base_type() == pl.Categorical
        assert result.balance.schema["mi_value"] == pl.Float64

        # First entry should be class_label
        assert result.balance["factor_name"][0] == "class_label"

        # Check classwise DataFrame
        assert isinstance(result.classwise, pl.DataFrame)
        assert result.classwise.height == num_classes * (num_factors + 1)

        # Check classwise DataFrame schema
        assert set(result.classwise.schema.keys()) == {
            "class_name",
            "factor_name",
            "mi_value",
            "is_imbalanced",
        }
        assert result.classwise.schema["class_name"].base_type() == pl.Categorical
        assert result.classwise.schema["factor_name"].base_type() == pl.Categorical
        assert result.classwise.schema["mi_value"] == pl.Float64
        assert result.classwise.schema["is_imbalanced"] == pl.Boolean

        # Check factors DataFrame
        assert isinstance(result.factors, pl.DataFrame)
        # Number of ordered pairs = n*(n-1) (includes both A->B and B->A)
        expected_pairs = num_factors * (num_factors - 1)
        assert result.factors.height == expected_pairs

        # Check factors DataFrame schema
        assert set(result.factors.schema.keys()) == {
            "factor1",
            "factor2",
            "mi_value",
            "is_correlated",
        }
        assert result.factors.schema["factor1"].base_type() == pl.Categorical
        assert result.factors.schema["factor2"].base_type() == pl.Categorical
        assert result.factors.schema["mi_value"] == pl.Float64
        assert result.factors.schema["is_correlated"] == pl.Boolean
