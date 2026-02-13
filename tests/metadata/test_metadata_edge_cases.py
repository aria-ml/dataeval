"""Test edge cases and internal methods for Metadata class."""

import numpy as np
import pytest

from dataeval._metadata import Metadata
from tests.embeddings.test_embeddings import MockDataset, ObjectDetectionTarget


@pytest.fixture
def mock_ds():
    """Create a simple mock dataset."""
    return MockDataset(
        np.ones((10, 3, 3)),
        np.ones((10, 3)),
        [{str(i): float(i), "category": f"cat_{i % 3}"} for i in range(10)],
    )


@pytest.fixture
def od_dataset_with_varied_types():
    """Create OD dataset with various factor value types."""
    boxes = [
        np.array([[10, 10, 20, 20], [30, 30, 40, 40]]),
        np.array([[5, 5, 15, 15]]),
        np.array([[25, 25, 35, 35], [45, 45, 55, 55]]),
    ]
    labels = [np.array([0, 1]), np.array([1]), np.array([0, 2])]
    scores = [np.array([0.9, 0.8]), np.array([0.95]), np.array([0.85, 0.92])]
    targets = [ObjectDetectionTarget(boxes[i], labels[i], scores[i]) for i in range(3)]

    metadata = [{"image_factor": i, "tuple_factor": (i, i + 1)} for i in range(3)]

    return MockDataset(np.ones((3, 3, 16, 16)), targets, metadata)


class TestMetadataProcessTargets:
    """Test _process_targets internal method."""

    def test_process_targets_none_dataset(self):
        """Test _process_targets returns None when dataset is None."""
        metadata = Metadata()
        raw = []
        labels = []
        bboxes = []
        scores = []
        srcidx = []
        result = metadata._process_targets(raw, labels, bboxes, scores, srcidx, 0, None)
        assert result is None


class TestMetadataFactorValueTypes:
    """Test handling of different factor value types."""

    def test_factor_values_as_tuple(self, od_dataset_with_varied_types):
        """Test factors with tuple values (non-list, non-ndarray iterables)."""
        metadata = Metadata(od_dataset_with_varied_types)
        # Should process despite tuple type
        factors = metadata.factor_names
        assert len(factors) >= 0  # Should not crash

    def test_build_image_rows_with_iterables(self, od_dataset_with_varied_types):
        """Test _build_image_rows with various iterable types."""
        metadata = Metadata(od_dataset_with_varied_types)
        # Access image_data to trigger processing
        image_data = metadata.image_data
        assert len(image_data) == 3

    def test_get_target_factor_values_non_list_iterable(self):
        """Test _get_target_factor_values with non-list iterables."""
        metadata = Metadata()
        # Test with tuple (non-list iterable)
        factor_values = (1.0, 2.0, 3.0)
        srcidx = np.array([0, 1, 2])
        result = metadata._get_target_factor_values("test", factor_values, srcidx, False, None, None)
        assert result == [1.0, 2.0, 3.0]

    def test_get_target_factor_values_od_image_level_iterable(self):
        """Test _get_target_factor_values for OD image-level with non-list iterable."""
        metadata = Metadata()
        factor_values = (1.0, 2.0, 3.0)
        srcidx = np.array([0, 0, 1, 1, 2])
        result = metadata._get_target_factor_values(
            "test",
            factor_values,
            srcidx,
            True,
            {"test"},
            {"test": factor_values},
        )
        assert len(result) == 5
        assert result == [1.0, 1.0, 2.0, 2.0, 3.0]

    def test_get_target_factor_values_od_target_level_iterable(self):
        """Test _get_target_factor_values for OD target-level with non-list iterable."""
        metadata = Metadata()
        factor_values = (1.0, 2.0, 3.0, 4.0, 5.0)
        srcidx = np.array([0, 0, 1, 1, 2])
        result = metadata._get_target_factor_values("test", factor_values, srcidx, True, {"other"}, None)
        assert result == [1.0, 2.0, 3.0, 4.0, 5.0]


class TestMetadataBuildFactors:
    """Test _build_factors internal method."""

    def test_build_factors_unstructured(self):
        """Test _build_factors with unstructured metadata."""
        metadata = Metadata()
        metadata._is_structured = False
        metadata._build_factors()
        assert metadata._factors == {}


class TestMetadataFilterByFactor:
    """Test filter_by_factor method."""

    def test_filter_by_factor_empty(self):
        """Test filter_by_factor returns empty array when no factors."""
        metadata = Metadata()
        # Manually set empty factors to avoid structuring
        metadata._factors = {}
        metadata._is_structured = True
        metadata._is_binned = True
        result = metadata.filter_by_factor(lambda name, info: True)
        assert result.shape == (0,)
        assert result.dtype == np.float64


class TestMetadataStructureUnbound:
    """Test _structure method with unbound dataset."""

    def test_structure_unbound_raises(self):
        """Test _structure raises when dataset is None."""
        metadata = Metadata()
        metadata._is_structured = False
        with pytest.raises(ValueError, match="No dataset bound"):
            metadata._structure()


class TestMetadataImageRowsIterables:
    """Test _build_image_rows with different iterable types."""

    def test_build_image_rows_tuple_values(self):
        """Test _build_image_rows handles tuple factor values."""
        metadata = Metadata()
        image_factor_dict = {"factor1": (1.0, 2.0, 3.0)}
        result = metadata._build_image_rows(3, image_factor_dict)
        assert result["factor1"] == [1.0, 2.0, 3.0]

    def test_build_image_rows_ndarray_values(self):
        """Test _build_image_rows handles ndarray factor values."""
        metadata = Metadata()
        image_factor_dict = {"factor1": np.array([1.0, 2.0, 3.0])}
        result = metadata._build_image_rows(3, image_factor_dict)
        assert result["factor1"] == [1.0, 2.0, 3.0]

    def test_build_image_rows_list_values(self):
        """Test _build_image_rows handles list factor values."""
        metadata = Metadata()
        image_factor_dict = {"factor1": [1.0, 2.0, 3.0]}
        result = metadata._build_image_rows(3, image_factor_dict)
        assert result["factor1"] == [1.0, 2.0, 3.0]
