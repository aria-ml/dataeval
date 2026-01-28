"""Tests for NumpyFlattenEncoder."""

import numpy as np
import pytest

from dataeval.encoders import NumpyFlattenEncoder


class MockDataset:
    """Simple dataset for testing."""

    def __init__(self, images: np.ndarray, labels: np.ndarray | None = None):
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        if self.labels is not None:
            return self.images[index], self.labels[index], {}
        return self.images[index]


class TestNumpyFlattenEncoderInit:
    """Test NumpyFlattenEncoder initialization."""

    def test_init_default(self):
        """Test default initialization."""
        encoder = NumpyFlattenEncoder()
        assert encoder.batch_size == 32  # Default batch size

    def test_init_with_batch_size(self):
        """Test initialization with custom batch size."""
        encoder = NumpyFlattenEncoder(batch_size=64)
        assert encoder.batch_size == 64

    def test_init_batch_size_minimum(self):
        """Test that batch size is at least 1."""
        encoder = NumpyFlattenEncoder(batch_size=0)
        assert encoder.batch_size >= 1

        encoder = NumpyFlattenEncoder(batch_size=-5)
        assert encoder.batch_size >= 1


class TestNumpyFlattenEncoderEncode:
    """Test NumpyFlattenEncoder.encode method."""

    @pytest.fixture
    def encoder(self):
        """Create an encoder for testing."""
        return NumpyFlattenEncoder(batch_size=10)

    @pytest.fixture
    def dataset(self):
        """Create a simple dataset for testing."""
        images = np.random.randn(20, 3, 16, 16)
        labels = np.arange(20)
        return MockDataset(images, labels)

    def test_encode_all_indices(self, encoder, dataset):
        """Test encoding all indices."""
        indices = list(range(len(dataset)))
        result = encoder.encode(dataset, indices)
        assert result.shape[0] == 20
        assert result.shape[1] == 3 * 16 * 16  # Flattened image

    def test_encode_subset_indices(self, encoder, dataset):
        """Test encoding subset of indices."""
        indices = [0, 5, 10, 15]
        result = encoder.encode(dataset, indices)
        assert result.shape[0] == 4

    def test_encode_empty_indices(self, encoder, dataset):
        """Test encoding with empty indices."""
        result = encoder.encode(dataset, [])
        assert result.shape[0] == 0

    def test_encode_single_index(self, encoder, dataset):
        """Test encoding single index."""
        result = encoder.encode(dataset, [0])
        assert result.shape[0] == 1
        assert result.shape[1] == 3 * 16 * 16

    def test_encode_streaming_mode(self, encoder, dataset):
        """Test encoding in streaming mode."""
        indices = list(range(15))
        batches = list(encoder.encode(dataset, indices, stream=True))

        # Should have multiple batches (batch_size=10, 15 items = 2 batches)
        assert len(batches) == 2

        # Check batch structure
        batch_indices, batch_embeddings = batches[0]
        assert len(batch_indices) == 10
        assert batch_embeddings.shape[0] == 10

        # Last batch should have remaining items
        batch_indices, batch_embeddings = batches[1]
        assert len(batch_indices) == 5
        assert batch_embeddings.shape[0] == 5

    def test_encode_streaming_empty(self, encoder, dataset):
        """Test streaming mode with empty indices."""
        batches = list(encoder.encode(dataset, [], stream=True))
        assert batches == []

    def test_encode_dataset_without_labels(self, encoder):
        """Test encoding dataset that returns only images."""
        images = np.random.randn(10, 3, 8, 8)
        dataset = MockDataset(images, labels=None)
        result = encoder.encode(dataset, list(range(10)))
        assert result.shape[0] == 10
        assert result.shape[1] == 3 * 8 * 8


class TestNumpyFlattenEncoderFlatten:
    """Test flattening behavior."""

    def test_flatten_2d_image(self):
        """Test flattening 2D images (grayscale)."""
        encoder = NumpyFlattenEncoder(batch_size=10)
        images = np.random.randn(5, 8, 8)  # Grayscale images
        dataset = MockDataset(images)

        result = encoder.encode(dataset, list(range(5)))
        assert result.shape == (5, 64)

    def test_flatten_3d_image(self):
        """Test flattening 3D images (RGB)."""
        encoder = NumpyFlattenEncoder(batch_size=10)
        images = np.random.randn(5, 3, 8, 8)  # RGB images
        dataset = MockDataset(images)

        result = encoder.encode(dataset, list(range(5)))
        assert result.shape == (5, 3 * 8 * 8)

    def test_flatten_preserves_values(self):
        """Test that flattening preserves image values."""
        encoder = NumpyFlattenEncoder(batch_size=10)
        images = np.arange(64).reshape(1, 8, 8).astype(np.float64)
        dataset = MockDataset(images)

        result = encoder.encode(dataset, [0])
        # Values should be preserved (in some order)
        np.testing.assert_array_almost_equal(np.sort(result.flatten()), np.sort(images.flatten()))


class TestNumpyFlattenEncoderRepr:
    """Test __repr__ method."""

    def test_repr(self):
        """Test repr output."""
        encoder = NumpyFlattenEncoder(batch_size=64)
        repr_str = repr(encoder)
        assert "NumpyFlattenEncoder" in repr_str
        assert "batch_size=64" in repr_str


class TestNumpyFlattenEncoderProtocol:
    """Test that NumpyFlattenEncoder conforms to EmbeddingEncoder protocol."""

    def test_protocol_conformance(self):
        """Test that NumpyFlattenEncoder has required protocol methods."""
        encoder = NumpyFlattenEncoder()

        # Check required properties
        assert hasattr(encoder, "batch_size")
        assert isinstance(encoder.batch_size, int)

        # Check required methods
        assert hasattr(encoder, "encode")
        assert callable(encoder.encode)


class TestNumpyFlattenEncoderBatching:
    """Test batching behavior."""

    def test_batching_with_small_batch(self):
        """Test encoding with small batch size."""
        encoder = NumpyFlattenEncoder(batch_size=2)
        images = np.random.randn(10, 4, 4)
        dataset = MockDataset(images)

        # Should process in multiple batches but produce correct result
        result = encoder.encode(dataset, list(range(10)))
        assert result.shape == (10, 16)

    def test_batching_with_large_batch(self):
        """Test encoding with batch size larger than dataset."""
        encoder = NumpyFlattenEncoder(batch_size=100)
        images = np.random.randn(5, 4, 4)
        dataset = MockDataset(images)

        result = encoder.encode(dataset, list(range(5)))
        assert result.shape == (5, 16)

    def test_batching_streaming_batch_count(self):
        """Test that streaming yields correct number of batches."""
        encoder = NumpyFlattenEncoder(batch_size=3)
        images = np.random.randn(10, 4, 4)
        dataset = MockDataset(images)

        batches = list(encoder.encode(dataset, list(range(10)), stream=True))

        # With batch_size=3 and 10 items, we expect 4 batches: 3, 3, 3, 1
        assert len(batches) == 4
        assert len(batches[0][0]) == 3  # First batch indices
        assert len(batches[1][0]) == 3  # Second batch indices
        assert len(batches[2][0]) == 3  # Third batch indices
        assert len(batches[3][0]) == 1  # Fourth batch indices (remaining)
