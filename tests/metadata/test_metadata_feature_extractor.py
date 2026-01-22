"""Test Metadata class as FeatureExtractor protocol (unbound usage)."""

import numpy as np
import pytest

from dataeval._metadata import Metadata
from tests.embeddings.test_embeddings import MockDataset


@pytest.fixture
def mock_ds():
    """Create a simple mock dataset."""
    return MockDataset(
        np.ones((10, 3, 3)),
        np.ones((10, 3)),
        [{str(i): float(i), "category": f"cat_{i % 3}"} for i in range(10)],
    )


@pytest.fixture
def mock_ds2():
    """Create a different mock dataset with same structure."""
    return MockDataset(
        np.ones((5, 3, 3)),
        np.ones((5, 3)),
        [{str(i): float(i + 10), "category": f"cat_{i % 3}"} for i in range(5)],
    )


class TestMetadataFeatureExtractor:
    """Test Metadata as a FeatureExtractor (unbound scenarios)."""

    def test_is_bound_false(self):
        """Test is_bound property returns False for unbound instance."""
        metadata = Metadata()
        assert not metadata.is_bound

    def test_is_bound_true(self, mock_ds):
        """Test is_bound property returns True for bound instance."""
        metadata = Metadata(mock_ds)
        assert metadata.is_bound

    def test_bind_returns_self(self, mock_ds):
        """Test bind() returns self for method chaining."""
        metadata = Metadata()
        result = metadata.bind(mock_ds)
        assert result is metadata
        assert metadata.is_bound

    def test_len_unbound_raises(self):
        """Test __len__ raises ValueError when no dataset is bound."""
        metadata = Metadata()
        with pytest.raises(ValueError, match="No dataset bound"):
            _ = len(metadata)

    def test_iter_unbound_raises(self):
        """Test __iter__ raises ValueError when no dataset is bound."""
        metadata = Metadata()
        with pytest.raises(ValueError, match="No dataset bound"):
            for _ in metadata:
                pass

    def test_getitem_unbound_raises(self):
        """Test __getitem__ raises ValueError when no dataset is bound."""
        metadata = Metadata()
        with pytest.raises(ValueError, match="No dataset bound"):
            _ = metadata[0]

    def test_shape_unbound_raises(self):
        """Test shape property raises ValueError when no dataset is bound."""
        metadata = Metadata()
        with pytest.raises(ValueError, match="No dataset bound"):
            _ = metadata.shape

    def test_call_unbound_raises(self):
        """Test __call__ raises ValueError when data is None and no dataset is bound."""
        metadata = Metadata()
        with pytest.raises(ValueError, match="No dataset bound"):
            _ = metadata()

    def test_call_with_data_unbound(self, mock_ds):
        """Test __call__ with data argument on unbound instance."""
        metadata = Metadata(continuous_factor_bins={"0": 5})
        result = metadata(mock_ds)
        assert result.shape[0] == 10
        assert len(result.shape) == 2

    def test_call_bound_no_args(self, mock_ds):
        """Test __call__ without arguments uses bound dataset."""
        metadata = Metadata(mock_ds, continuous_factor_bins={"0": 5})
        result = metadata()
        assert result.shape[0] == 10

    def test_call_same_dataset(self, mock_ds):
        """Test __call__ with same dataset (by identity) returns cached data."""
        metadata = Metadata(mock_ds, continuous_factor_bins={"0": 5})
        result1 = metadata()
        result2 = metadata(mock_ds)
        np.testing.assert_array_equal(result1, result2)

    def test_call_different_dataset(self, mock_ds, mock_ds2):
        """Test __call__ with different dataset creates new computation."""
        metadata = Metadata(mock_ds, continuous_factor_bins={"0": 5})
        result1 = metadata()
        result2 = metadata(mock_ds2)
        assert result1.shape[0] == 10
        assert result2.shape[0] == 5

    def test_bind_clears_state(self, mock_ds, mock_ds2):
        """Test bind() clears cached state."""
        metadata = Metadata(mock_ds, continuous_factor_bins={"0": 5})
        # Access to cache state
        _ = metadata()

        # Bind new dataset
        metadata.bind(mock_ds2)

        # Verify state was cleared by checking we get new data
        result = metadata()
        assert result.shape[0] == 5


class TestMetadataErrorCases:
    """Test error handling in Metadata."""

    def test_exclude_and_include_raises(self, mock_ds):
        """Test both exclude and include raises ValueError."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            Metadata(mock_ds, exclude=["a"], include=["b"])

    def test_getitem_slice_unbound_raises(self):
        """Test __getitem__ with slice raises when unbound."""
        metadata = Metadata()
        with pytest.raises(ValueError, match="No dataset bound"):
            _ = metadata[0:5]

    def test_getitem_string_unbound_raises(self):
        """Test __getitem__ with string raises when unbound."""
        metadata = Metadata()
        with pytest.raises(ValueError, match="No dataset bound"):
            _ = metadata["some_factor"]

    def test_getitem_invalid_factor_raises(self, mock_ds):
        """Test __getitem__ with invalid factor name raises KeyError."""
        metadata = Metadata(mock_ds)
        with pytest.raises(KeyError, match="not found"):
            _ = metadata["nonexistent_factor"]

    def test_getitem_invalid_type_raises(self, mock_ds):
        """Test __getitem__ with invalid type raises TypeError."""
        metadata = Metadata(mock_ds)
        with pytest.raises(TypeError, match="Index must be"):
            _ = metadata[1.5]  # type: ignore


class TestMetadataTargetFactorsOnly:
    """Test target_factors_only property."""

    def test_target_factors_only_default(self, mock_ds):
        """Test target_factors_only defaults to False."""
        metadata = Metadata(mock_ds)
        assert not metadata.target_factors_only

    def test_target_factors_only_setter_triggers_rebuild(self, get_od_dataset):
        """Test setting target_factors_only triggers factor rebuild."""
        od_ds = get_od_dataset(10, 2)

        metadata = Metadata(od_ds)
        initial_factors = set(metadata.factor_names)

        # Set target_factors_only to True should rebuild
        metadata.target_factors_only = True
        filtered_factors = set(metadata.factor_names)

        # Should have fewer or equal factors when filtering to target-only
        assert len(filtered_factors) <= len(initial_factors)


class TestMetadataItemCount:
    """Test item_count property."""

    def test_item_count_no_trigger_when_nonzero(self, mock_ds):
        """Test item_count property doesn't trigger structure when count is already set."""
        metadata = Metadata(mock_ds)
        # Count is set during init
        count = metadata.item_count
        assert count == 10
