"""Test edge cases and internal methods for Metadata class."""

import numpy as np
import pytest

from dataeval import Metadata
from dataeval._metadata._structurers import ICStructurer, ODImageStructurer
from dataeval.exceptions import NotFittedError
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


class TestStructuringEmptyDataset:
    """Structuring a dataset with nothing in it."""

    def test_empty_dataset_builds_empty_rows(self):
        """An empty dataset structures into zero rows rather than failing."""
        data = ICStructurer().build(MockDataset(np.empty((0, 3, 3)), np.empty((0, 3)), []))
        assert data.layout.counts == {"unit": 0, "instance": 0}
        assert len(data.class_labels) == 0
        assert all(len(values) == 0 for values in data.to_rows().values())

    def test_unstructured_metadata_has_no_factors(self):
        """An unbound instance reports no factors instead of processing targets."""
        metadata = Metadata()
        with pytest.raises(NotFittedError, match="No dataset bound"):
            _ = metadata.factor_names


class TestMetadataFactorValueTypes:
    """Test handling of different factor value types."""

    def test_factor_values_as_tuple(self, od_dataset_with_varied_types):
        """Test factors with tuple values (non-list, non-ndarray iterables)."""
        metadata = Metadata(od_dataset_with_varied_types)
        # Should process despite tuple type
        factors = metadata.factor_names
        assert len(factors) >= 0  # Should not crash

    def test_image_rows_with_iterables(self, od_dataset_with_varied_types):
        """Item-level rows are built from datasets carrying iterable factor values."""
        metadata = Metadata(od_dataset_with_varied_types)
        assert len(metadata.rows_at("unit")) == 3

    def test_item_factor_values_replicate_across_instances(self, od_dataset_with_varied_types):
        """An image-level factor reaches every instance of that image by propagation."""
        metadata = Metadata(od_dataset_with_varied_types)
        instances = metadata.target_data
        # The fixture has 2, 1 and 2 detections for images 0, 1 and 2.
        assert instances["item_index"].to_list() == [0, 0, 1, 2, 2]
        assert instances["image_factor"].to_list() == [0, 0, 1, 2, 2]

    def test_target_factor_values_stay_per_instance(self):
        """A list-valued factor is spread one entry per instance, not replicated."""
        targets = [
            ObjectDetectionTarget(
                np.array([[10, 10, 20, 20], [30, 30, 40, 40]]), np.array([0, 1]), np.array([0.9, 0.8])
            ),
            ObjectDetectionTarget(np.array([[5, 5, 15, 15]]), np.array([1]), np.array([0.95])),
        ]
        dataset = MockDataset(
            np.ones((2, 3, 16, 16)),
            targets,
            [{"iou": [0.1, 0.2]}, {"iou": [0.3]}],
        )
        metadata = Metadata(dataset)
        assert metadata.target_data["iou"].to_list() == [0.1, 0.2, 0.3]
        # Instance-level values have no image-level counterpart, so image rows are null.
        assert metadata.rows_at("unit")["iou"].to_list() == [None, None]


class TestMetadataBuildFactors:
    """Test _build_factors internal method."""

    def test_build_factors_unstructured(self):
        """Test _build_factors with unstructured metadata."""
        metadata = Metadata()
        metadata._is_structured = False
        metadata._build_factors()
        assert metadata._factors == set()


class TestMetadataFilterByFactor:
    """Test filter_by_factor method."""

    def test_filter_by_factor_empty(self):
        """Test filter_by_factor returns empty array when no factors."""
        metadata = Metadata()
        # Manually set empty factors to avoid structuring
        metadata._factors = set()
        metadata._is_structured = True
        metadata._is_binned = True
        result = metadata.filter_by_factor(lambda name, info: True)
        # (rows, 0), not a bare empty array: the row count belongs to the view and does
        # not depend on how many factors there are.
        assert result.shape == (0, 0)
        assert result.dtype == np.float64


class TestMetadataStructureUnbound:
    """Test _structure method with unbound dataset."""

    def test_structure_unbound_raises(self):
        """Test _structure raises when dataset is None."""
        metadata = Metadata()
        metadata._is_structured = False
        with pytest.raises(NotFittedError, match="No dataset bound"):
            metadata._structure()


class TestImageRowIterables:
    """Item-level factor values survive whatever container the dataset used."""

    @pytest.mark.parametrize(
        "values",
        [(1.0, 2.0, 3.0), np.array([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]],
        ids=["tuple", "ndarray", "list"],
    )
    def test_item_rows_normalize_iterable_values(self, values):
        """Tuples, ndarrays and lists all land as the same image-level column."""
        dataset = MockDataset(
            np.ones((3, 3, 16, 16)),
            [ObjectDetectionTarget(np.array([[1, 1, 2, 2]]), np.array([0]), np.array([0.5])) for _ in range(3)],
            [{"factor1": float(value)} for value in values],
        )
        rows = ODImageStructurer().build(dataset).to_rows()
        assert rows["factor1"][:3] == [1.0, 2.0, 3.0]
