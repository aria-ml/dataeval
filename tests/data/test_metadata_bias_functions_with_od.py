"""
Tests for bias/metadata analysis functions with OD datasets and dual-key indexing.

This module ensures that bias functions (balance, diversity, parity) and metadata
analysis functions (find_most_deviated_factors, find_ood_predictors) correctly
handle both Image Classification (IC) and Object Detection (OD) datasets with
dual-key indexing.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataeval.data import Metadata
from dataeval.evaluators.bias import Balance, Diversity, Parity
from tests.data.test_data_embeddings import MockDataset


@pytest.fixture
def od_dataset_for_bias():
    """Create an OD dataset with both image-level and target-level metadata."""
    from dataclasses import dataclass

    @dataclass
    class ODTarget:
        boxes: np.ndarray
        labels: np.ndarray
        scores: np.ndarray

    # 5 images with varying detections
    data = np.ones((5, 3, 32, 32))
    targets = [
        ODTarget(
            boxes=np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
            labels=np.array([0, 1]),
            scores=np.array([0.9, 0.8]),
        ),
        ODTarget(
            boxes=np.array([[5, 5, 15, 15]]),
            labels=np.array([1]),
            scores=np.array([0.95]),
        ),
        ODTarget(
            boxes=np.array([[1, 1, 5, 5], [10, 10, 20, 20]]),
            labels=np.array([0, 0]),
            scores=np.array([0.85, 0.75]),
        ),
        ODTarget(
            boxes=np.array([[15, 15, 25, 25], [30, 30, 40, 40], [5, 5, 10, 10]]),
            labels=np.array([2, 1, 0]),
            scores=np.array([0.92, 0.88, 0.79]),
        ),
        ODTarget(
            boxes=np.array([[2, 2, 8, 8]]),
            labels=np.array([2]),
            scores=np.array([0.91]),
        ),
    ]

    # Image-level metadata (5 values)
    # Target-level metadata (2+1+2+3+1 = 9 values)
    metadata = [
        {"weather": "sunny", "time": "morning", "bbox_metadata": [10.5, 12.3]},
        {"weather": "rainy", "time": "afternoon", "bbox_metadata": [8.7]},
        {"weather": "cloudy", "time": "evening", "bbox_metadata": [15.2, 9.8]},
        {"weather": "sunny", "time": "night", "bbox_metadata": [11.1, 13.5, 7.9]},
        {"weather": "rainy", "time": "morning", "bbox_metadata": [14.3]},
    ]

    return MockDataset(data, targets, metadata)


@pytest.fixture
def ic_dataset_for_bias():
    """Create an IC dataset for comparison."""
    # 5 images with single labels (using one-hot encoding)
    data = np.ones((5, 3, 32, 32))

    # One-hot encoded targets for 3 classes
    targets = [
        np.array([1.0, 0.0, 0.0]),  # class 0
        np.array([0.0, 1.0, 0.0]),  # class 1
        np.array([1.0, 0.0, 0.0]),  # class 0
        np.array([0.0, 0.0, 1.0]),  # class 2
        np.array([0.0, 0.0, 1.0]),  # class 2
    ]

    # Image-level metadata only (5 values)
    # Mix of categorical (for bias functions) and numeric (for OOD functions)
    metadata = [
        {"weather": "sunny", "time": "morning", "altitude": 100, "brightness": 0.8},
        {"weather": "rainy", "time": "afternoon", "altitude": 200, "brightness": 0.4},
        {"weather": "cloudy", "time": "evening", "altitude": 150, "brightness": 0.6},
        {"weather": "sunny", "time": "night", "altitude": 180, "brightness": 0.3},
        {"weather": "rainy", "time": "morning", "altitude": 120, "brightness": 0.5},
    ]

    return MockDataset(data, targets, metadata)


@pytest.fixture
def ic_dataset_numeric_only():
    """Create an IC dataset with only numeric metadata for OOD functions."""
    # 5 images with single labels (using one-hot encoding)
    data = np.ones((5, 3, 32, 32))

    # One-hot encoded targets for 3 classes
    targets = [
        np.array([1.0, 0.0, 0.0]),  # class 0
        np.array([0.0, 1.0, 0.0]),  # class 1
        np.array([1.0, 0.0, 0.0]),  # class 0
        np.array([0.0, 0.0, 1.0]),  # class 2
        np.array([0.0, 0.0, 1.0]),  # class 2
    ]

    # Only numeric metadata for OOD functions
    metadata = [
        {"altitude": 100, "brightness": 0.8, "temperature": 25.0},
        {"altitude": 200, "brightness": 0.4, "temperature": 18.5},
        {"altitude": 150, "brightness": 0.6, "temperature": 22.0},
        {"altitude": 180, "brightness": 0.3, "temperature": 15.0},
        {"altitude": 120, "brightness": 0.5, "temperature": 20.5},
    ]

    return MockDataset(data, targets, metadata)


class TestMetadataStructureWithBiasFunctions:
    """Test that Metadata correctly structures data for bias function consumption."""

    def test_od_metadata_has_image_and_target_rows(self, od_dataset_for_bias):
        """Verify OD datasets create separate image and target rows."""
        md = Metadata(od_dataset_for_bias)

        # Should have 5 image rows + 9 target rows = 14 total
        assert len(md.dataframe) == 14

        # Image rows
        image_rows = md.image_data
        assert len(image_rows) == 5
        assert all(image_rows["target_index"].is_null())
        assert image_rows["weather"].to_list() == ["sunny", "rainy", "cloudy", "sunny", "rainy"]

        # Target rows
        target_rows = md.target_data
        assert len(target_rows) == 9
        assert all(target_rows["target_index"].is_not_null())
        assert all(target_rows["weather"].is_not_null())

    def test_ic_metadata_has_only_target_rows(self, ic_dataset_for_bias):
        """Verify IC datasets only create target rows (no separate image rows)."""
        md = Metadata(ic_dataset_for_bias)

        # Should have 5 rows only (one per image)
        assert len(md.dataframe) == 5

        # All rows are image and target rows for IC
        target_rows = md.target_data
        assert len(target_rows) == 5

        # All rows are image and target rows for IC
        image_rows = md.image_data
        assert len(image_rows) == 5

    def test_od_factor_data_uses_target_rows(self, od_dataset_for_bias):
        """Verify factor_data for OD uses target rows (9 rows)."""
        md = Metadata(od_dataset_for_bias)

        # factor_data should have 9 rows (one per detection)
        assert md.factor_data.shape[0] == 9

        # class_labels should also have 9 values
        assert len(md.class_labels) == 9

    def test_ic_factor_data_uses_all_rows(self, ic_dataset_for_bias):
        """Verify factor_data for IC uses all rows (5 rows)."""
        md = Metadata(ic_dataset_for_bias)

        # factor_data should have 5 rows (one per image)
        assert md.factor_data.shape[0] == 5

        # class_labels should also have 5 values
        assert len(md.class_labels) == 5


class TestBiasFunctionsWithOD:
    """Test bias functions (balance, diversity, parity) with OD datasets."""

    def test_balance_with_od_dataset(self, od_dataset_for_bias):
        """Test balance() works with OD dataset (uses target-level data)."""
        md = Metadata(od_dataset_for_bias)

        # balance() uses binned_data and class_labels
        # For OD, this should be target-level (9 samples, 1 factor: bbox_metadata)
        result = Balance().evaluate(md)

        # Should return results for target-level analysis
        # result.balance is a DataFrame with factor_name and mi_value columns
        # It includes class_label + all metadata factors
        assert result.balance.height == len(md.factor_names) + 1  # class_label + bbox_metadata
        assert "factor_name" in result.balance.columns
        assert "mi_value" in result.balance.columns

        # result.factors is a DataFrame with factor1, factor2, mi_value, is_correlated columns
        # For n factors, we get n*(n-1) factor pairs (excluding diagonal)
        n_factors = len(md.factor_names)
        expected_rows = n_factors * (n_factors - 1) if n_factors > 1 else 0
        assert result.factors.height == expected_rows
        if expected_rows > 0:
            assert "factor1" in result.factors.columns
            assert "factor2" in result.factors.columns
            assert "mi_value" in result.factors.columns
            assert "is_correlated" in result.factors.columns

        # result.classwise is a DataFrame with class_name, factor_name, mi_value, is_imbalanced columns
        # It contains one row per (class, factor) combination (excluding class_label itself)
        n_classes = len(md.index2label)
        assert result.classwise.height == n_classes * len(md.factor_names)
        assert "class_name" in result.classwise.columns
        assert "factor_name" in result.classwise.columns
        assert "mi_value" in result.classwise.columns
        assert "is_imbalanced" in result.classwise.columns

    def test_diversity_with_od_dataset(self, od_dataset_for_bias):
        """Test diversity() works with OD dataset (uses target-level data)."""
        md = Metadata(od_dataset_for_bias)

        # diversity() uses binned_data and class_labels
        result = Diversity().evaluate(md)

        # result.factors is a DataFrame with factor_name, diversity_value, is_low_diversity columns
        # It includes only metadata factors (not class_label)
        assert result.factors.height == len(md.factor_names)
        assert "factor_name" in result.factors.columns
        assert "diversity_value" in result.factors.columns
        assert "is_low_diversity" in result.factors.columns

        # result.classwise is a DataFrame with class_name, factor_name, diversity_value, is_low_diversity
        # One row per (class, factor) combination
        n_classes = len(md.index2label)
        assert result.classwise.height == n_classes * len(md.factor_names)
        assert "class_name" in result.classwise.columns
        assert "factor_name" in result.classwise.columns
        assert "diversity_value" in result.classwise.columns
        assert "is_low_diversity" in result.classwise.columns

    def test_parity_with_od_dataset(self, od_dataset_for_bias):
        """Test parity() works with OD dataset (uses target-level data)."""
        md = Metadata(od_dataset_for_bias)

        # parity() uses binned_data and class_labels
        result = Parity().evaluate(md)

        # result.factors is a DataFrame with factor_name, score, p_value, is_correlated, has_insufficient_data
        assert result.factors.height == len(md.factor_names)
        assert "factor_name" in result.factors.columns
        assert "score" in result.factors.columns
        assert "p_value" in result.factors.columns
        assert "is_correlated" in result.factors.columns
        assert "has_insufficient_data" in result.factors.columns

        # Verify factor names match
        factor_names_in_result = result.factors["factor_name"].to_list()
        assert factor_names_in_result == list(md.factor_names)

    def test_bias_functions_with_ic_dataset(self, ic_dataset_for_bias):
        """Test bias functions work with IC dataset (baseline)."""
        md = Metadata(ic_dataset_for_bias)

        # All bias functions should work
        # Use num_neighbors=3 for Balance since IC dataset has only 5 samples
        # (sklearn requires n_neighbors < n_samples_fit for self-query)
        balance_result = Balance(num_neighbors=3).evaluate(md)
        diversity_result = Diversity().evaluate(md)
        parity_result = Parity().evaluate(md)

        # Verify basic structure - all now return DataFrames
        # Balance: balance includes class_label + metadata factors
        assert balance_result.balance.height == len(md.factor_names) + 1
        # Diversity: factors includes only metadata factors (not class_label)
        assert diversity_result.factors.height == len(md.factor_names)
        # Parity: factors includes one row per metadata factor
        assert parity_result.factors.height == len(md.factor_names)


class TestFactorDataConsistency:
    """Test that factor_data is consistent with what bias functions expect."""

    def test_od_factor_data_length_matches_class_labels(self, od_dataset_for_bias):
        """Verify factor_data and class_labels have same length for OD."""
        md = Metadata(od_dataset_for_bias)

        assert md.factor_data.shape[0] == len(md.class_labels)
        # Should be 9 (number of detections)
        assert md.factor_data.shape[0] == 9

    def test_ic_factor_data_length_matches_class_labels(self, ic_dataset_for_bias):
        """Verify factor_data and class_labels have same length for IC."""
        md = Metadata(ic_dataset_for_bias)

        assert md.factor_data.shape[0] == len(md.class_labels)
        # Should be 5 (number of images)
        assert md.factor_data.shape[0] == 5

    def test_od_binned_data_length_matches_class_labels(self, od_dataset_for_bias):
        """Verify binned_data and class_labels have same length for OD."""
        md = Metadata(od_dataset_for_bias)

        assert md.binned_data.shape[0] == len(md.class_labels)
        # Should be 9 (number of detections)
        assert md.binned_data.shape[0] == 9

    def test_ic_binned_data_length_matches_class_labels(self, ic_dataset_for_bias):
        """Verify binned_data and class_labels have same length for IC."""
        md = Metadata(ic_dataset_for_bias)

        assert md.binned_data.shape[0] == len(md.class_labels)
        # Should be 5 (number of images)
        assert md.binned_data.shape[0] == 5
