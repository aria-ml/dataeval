"""Test Embeddings class as FeatureExtractor protocol (unbound usage)."""

import numpy as np
import pytest
import torch

from dataeval import Embeddings
from tests.embeddings.test_embeddings import MockDataset


@pytest.fixture
def mock_ds():
    """Create a simple mock dataset."""
    return MockDataset(torch.ones((10, 3, 16, 16)), torch.ones((10, 3)))


@pytest.fixture
def mock_ds2():
    """Create a different mock dataset with same structure."""
    return MockDataset(torch.ones((5, 3, 16, 16)), torch.ones((5, 3)))


@pytest.fixture
def simple_model():
    """Simple model for testing."""

    class SimpleModel(torch.nn.Module):
        def forward(self, x):
            return x.flatten(1)

    return SimpleModel()


class TestEmbeddingsFeatureExtractor:
    """Test Embeddings as a FeatureExtractor (unbound scenarios)."""

    def test_is_bound_false(self, simple_model):
        """Test is_bound property returns False for unbound instance."""
        embeddings = Embeddings(model=simple_model)
        assert not embeddings.is_bound

    def test_is_bound_true(self, mock_ds, simple_model):
        """Test is_bound property returns True for bound instance."""
        embeddings = Embeddings(mock_ds, model=simple_model)
        assert embeddings.is_bound

    def test_bind_returns_self(self, mock_ds, simple_model):
        """Test bind() returns self for method chaining."""
        embeddings = Embeddings(model=simple_model)
        result = embeddings.bind(mock_ds)
        assert result is embeddings
        assert embeddings.is_bound

    def test_bind_clears_cache(self, mock_ds, mock_ds2, simple_model):
        """Test bind() clears cached embeddings."""
        embeddings = Embeddings(mock_ds, model=simple_model, batch_size=10)
        # Compute embeddings to populate cache
        _ = embeddings[0]
        assert len(embeddings._cached_idx) > 0

        # Bind new dataset should clear cache
        embeddings.bind(mock_ds2)
        assert len(embeddings._cached_idx) == 0
        assert embeddings._embeddings.size == 0

    def test_call_unbound_raises(self, simple_model):
        """Test __call__ raises ValueError when data is None and no dataset is bound."""
        embeddings = Embeddings(model=simple_model)
        with pytest.raises(ValueError, match="No dataset bound"):
            _ = embeddings()

    def test_call_with_data_unbound(self, mock_ds, simple_model):
        """Test __call__ with data argument on unbound instance."""
        embeddings = Embeddings(model=simple_model, batch_size=10)
        result = embeddings(mock_ds)
        assert result.shape[0] == 10

    def test_call_bound_no_args(self, mock_ds, simple_model):
        """Test __call__ without arguments uses bound dataset."""
        embeddings = Embeddings(mock_ds, model=simple_model, batch_size=10)
        result = embeddings()
        assert result.shape[0] == 10

    def test_call_same_dataset(self, mock_ds, simple_model):
        """Test __call__ with same dataset (by identity) returns cached data."""
        embeddings = Embeddings(mock_ds, model=simple_model, batch_size=10)
        result1 = embeddings()
        result2 = embeddings(mock_ds)
        np.testing.assert_array_equal(result1, result2)

    def test_call_different_dataset(self, mock_ds, mock_ds2, simple_model):
        """Test __call__ with different dataset creates new computation."""
        embeddings = Embeddings(mock_ds, model=simple_model, batch_size=10)
        result1 = embeddings()
        result2 = embeddings(mock_ds2)
        assert result1.shape[0] == 10
        assert result2.shape[0] == 5

    def test_shape_unbound_raises(self, simple_model):
        """Test shape property raises ValueError when no dataset is bound."""
        embeddings = Embeddings(model=simple_model)
        with pytest.raises(ValueError, match="Cannot determine shape"):
            _ = embeddings.shape

    def test_len_unbound_raises(self, simple_model):
        """Test __len__ raises ValueError when no dataset is bound."""
        embeddings = Embeddings(model=simple_model)
        with pytest.raises(ValueError, match="Cannot determine length"):
            _ = len(embeddings)


class TestEmbeddingsErrorCases:
    """Test error handling in Embeddings."""

    def test_bind_embeddings_only_raises(self):
        """Test bind() raises ValueError on embeddings-only instance."""
        arr = np.random.randn(10, 512)
        embeddings = Embeddings.from_array(arr)
        with pytest.raises(ValueError, match="Cannot bind dataset"):
            embeddings.bind(None)  # type: ignore

    def test_new_embeddings_only_raises(self):
        """Test new() raises ValueError on embeddings-only instance."""
        arr = np.random.randn(10, 512)
        embeddings = Embeddings.from_array(arr)
        with pytest.raises(ValueError, match="does not have a model"):
            embeddings.new(None)  # type: ignore

    def test_batch_unbound_raises(self, simple_model):
        """Test _batch raises ValueError when no dataset is bound."""
        embeddings = Embeddings(model=simple_model)
        with pytest.raises(ValueError, match="No dataset bound"):
            list(embeddings._batch([0, 1, 2]))

    def test_initialize_storage_unbound_raises(self, simple_model):
        """Test _initialize_storage raises ValueError when no dataset is bound."""
        embeddings = Embeddings(model=simple_model)
        sample = np.random.randn(512)
        with pytest.raises(ValueError, match="No dataset bound"):
            embeddings._initialize_storage(sample)

    def test_should_use_memmap_unbound_raises(self, simple_model, tmp_path):
        """Test _should_use_memmap raises ValueError when no dataset is bound."""
        embeddings = Embeddings(model=simple_model, path=tmp_path / "embeddings.npy")
        with pytest.raises(ValueError, match="No dataset bound"):
            embeddings._should_use_memmap((512,))

    def test_getitem_invalid_type_raises(self, mock_ds, simple_model):
        """Test __getitem__ with invalid type raises TypeError."""
        embeddings = Embeddings(mock_ds, model=simple_model, batch_size=10)
        with pytest.raises(TypeError):
            _ = embeddings[1.5]  # type: ignore

    def test_getitem_invalid_iterable_type_raises(self, mock_ds, simple_model):
        """Test __getitem__ with invalid iterable types raises TypeError."""
        embeddings = Embeddings(mock_ds, model=simple_model, batch_size=10)
        with pytest.raises(TypeError, match="must be integers"):
            _ = embeddings[[0, "invalid", 2]]


class TestEmbeddingsHash:
    """Test hash functionality for Embeddings."""

    def test_hash_unbound(self, simple_model):
        """Test hash for unbound instance."""
        embeddings = Embeddings(model=simple_model)
        hash_val = hash(embeddings)
        assert isinstance(hash_val, int)

    def test_hash_embeddings_only(self):
        """Test hash for embeddings-only instance."""
        arr = np.random.randn(10, 512)
        embeddings = Embeddings.from_array(arr)
        hash_val = hash(embeddings)
        assert isinstance(hash_val, int)


class TestEmbeddingsPath:
    """Test path property and memmap conversion."""

    def test_path_setter_none_converts_memmap(self, tmp_path, mock_ds, simple_model):
        """Test setting path to None converts memmap to in-memory array."""
        path = tmp_path / "embeddings.npy"
        embeddings = Embeddings(mock_ds, model=simple_model, path=path, memory_threshold=0.0)
        # Force memmap creation
        embeddings._use_memmap = True
        embeddings._embeddings = np.memmap(path, dtype=np.float32, mode="w+", shape=(10, 768))

        # Setting path to None should convert to in-memory
        embeddings.path = None
        assert not isinstance(embeddings._embeddings, np.memmap)
        assert embeddings._path is None

    def test_path_setter_saves_embeddings(self, tmp_path, mock_ds, simple_model):
        """Test setting path saves existing embeddings."""
        embeddings = Embeddings(mock_ds, model=simple_model, batch_size=10)
        # Compute some embeddings
        _ = embeddings[0]

        # Set new path should save
        new_path = tmp_path / "new_embeddings.npy"
        embeddings.path = new_path
        assert embeddings._path == new_path
