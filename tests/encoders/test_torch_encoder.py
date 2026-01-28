"""Tests for TorchEmbeddingEncoder."""

from typing import Any

import numpy as np
import pytest
import torch

from dataeval.encoders import TorchEmbeddingEncoder
from dataeval.protocols import ArrayLike, Dataset


class MockDataset(Dataset[ArrayLike]):
    """Simple dataset for testing."""

    def __init__(self, images: np.ndarray | torch.Tensor):
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> ArrayLike:
        return self.images[index]


class MockDatasetWithLabels(Dataset[tuple[ArrayLike, Any, Any]]):
    """Simple dataset for testing."""

    def __init__(self, images: np.ndarray | torch.Tensor, labels: np.ndarray | torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[ArrayLike, Any, Any]:
        return self.images[index], self.labels[index], {}


class TestTorchEmbeddingEncoderInit:
    """Test TorchEmbeddingEncoder initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        model = torch.nn.Flatten()
        encoder = TorchEmbeddingEncoder(model)
        assert encoder.batch_size > 0  # Should use default batch size
        assert encoder.device is not None
        assert encoder.layer_name is None
        assert encoder.use_output is True

    def test_init_with_batch_size(self):
        """Test initialization with custom batch size."""
        model = torch.nn.Flatten()
        encoder = TorchEmbeddingEncoder(model, batch_size=64)
        assert encoder.batch_size == 64

    def test_init_with_device(self):
        """Test initialization with specified device."""
        model = torch.nn.Flatten()
        encoder = TorchEmbeddingEncoder(model, device="cpu")
        assert encoder.device == torch.device("cpu")

    def test_init_with_layer_name(self):
        """Test initialization with layer extraction."""
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(768, 128),
        )
        encoder = TorchEmbeddingEncoder(model, batch_size=10, layer_name="0")
        assert encoder.layer_name == "0"

    def test_init_with_invalid_layer_raises(self):
        """Test that invalid layer name raises ValueError."""
        model = torch.nn.Flatten()
        with pytest.raises(ValueError, match="Invalid layer"):
            TorchEmbeddingEncoder(model, layer_name="nonexistent")

    def test_init_with_transforms(self):
        """Test initialization with transforms."""

        class MockTransform:
            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        model = torch.nn.Flatten()
        encoder = TorchEmbeddingEncoder(model, transforms=MockTransform())
        # Transform should be stored (internal implementation detail)
        assert len(encoder._transforms) == 1

    def test_init_with_multiple_transforms(self):
        """Test initialization with multiple transforms."""

        class ScaleTransform:
            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        class ShiftTransform:
            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        model = torch.nn.Flatten()
        encoder = TorchEmbeddingEncoder(model, transforms=[ScaleTransform(), ShiftTransform()])
        assert len(encoder._transforms) == 2


class TestTorchEmbeddingEncoderEncode:
    """Test TorchEmbeddingEncoder.encode method."""

    @pytest.fixture
    def encoder(self):
        """Create a simple encoder for testing."""
        return TorchEmbeddingEncoder(torch.nn.Flatten(), batch_size=10, device="cpu")

    @pytest.fixture
    def dataset(self):
        """Create a simple dataset for testing."""
        images = torch.randn(20, 3, 16, 16)
        labels = torch.arange(20)
        return MockDatasetWithLabels(images, labels)

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

    def test_encode_out_of_range_raises(self, encoder, dataset):
        """Test that out-of-range indices raise IndexError."""
        with pytest.raises(IndexError, match="out of range"):
            encoder.encode(dataset, [0, 100])

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
        images = torch.randn(10, 3, 8, 8)
        dataset = MockDataset(images)
        result = encoder.encode(dataset, list(range(10)))
        assert result.shape[0] == 10
        assert result.shape[1] == 3 * 8 * 8


class TestTorchEmbeddingEncoderLayerExtraction:
    """Test layer extraction functionality."""

    def test_extract_intermediate_layer_output(self):
        """Test extracting output from intermediate layer."""
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(768, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )
        # Extract from Linear layer (index 1)
        encoder = TorchEmbeddingEncoder(model, batch_size=10, layer_name="1", device="cpu")

        images = torch.randn(5, 3, 16, 16)
        dataset = MockDataset(images)
        result = encoder.encode(dataset, list(range(5)))

        assert result.shape == (5, 128)  # Linear(768, 128) output

    def test_extract_intermediate_layer_input(self):
        """Test extracting input to intermediate layer."""
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(768, 128),
            torch.nn.ReLU(),
        )
        # Extract input to Linear layer (index 1), with use_output=False
        encoder = TorchEmbeddingEncoder(model, batch_size=10, layer_name="1", use_output=False, device="cpu")

        images = torch.randn(5, 3, 16, 16)
        dataset = MockDataset(images)
        result = encoder.encode(dataset, list(range(5)))

        assert result.shape == (5, 768)  # Input to Linear is flattened image


class TestTorchEmbeddingEncoderTransforms:
    """Test transform functionality."""

    def test_transforms_applied(self):
        """Test that transforms are applied during encoding."""

        class DoubleTransform:
            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        model = torch.nn.Flatten()
        encoder_no_transform = TorchEmbeddingEncoder(model, batch_size=10, device="cpu")
        encoder_with_transform = TorchEmbeddingEncoder(model, batch_size=10, transforms=DoubleTransform(), device="cpu")

        images = torch.ones(3, 1, 4, 4)  # Simple ones array
        dataset = MockDataset(images)

        result_no_transform = encoder_no_transform.encode(dataset, [0])
        result_with_transform = encoder_with_transform.encode(dataset, [0])

        # With transform, values should be doubled
        np.testing.assert_array_almost_equal(result_with_transform, result_no_transform * 2)


class TestTorchEmbeddingEncoderRepr:
    """Test __repr__ method."""

    def test_repr_basic(self):
        """Test basic repr."""
        model = torch.nn.Flatten()
        encoder = TorchEmbeddingEncoder(model, batch_size=32, device="cpu")
        repr_str = repr(encoder)
        assert "TorchEmbeddingEncoder" in repr_str
        assert "batch_size=32" in repr_str
        assert "cpu" in repr_str

    def test_repr_with_layer_name(self):
        """Test repr includes layer name when set."""
        model = torch.nn.Sequential(torch.nn.Flatten())
        encoder = TorchEmbeddingEncoder(model, batch_size=32, layer_name="0", device="cpu")
        repr_str = repr(encoder)
        assert "layer_name='0'" in repr_str


class TestTorchEmbeddingEncoderProtocol:
    """Test that TorchEmbeddingEncoder conforms to EmbeddingEncoder protocol."""

    def test_protocol_conformance(self):
        """Test that TorchEmbeddingEncoder has required protocol methods."""
        model = torch.nn.Flatten()
        encoder = TorchEmbeddingEncoder(model, batch_size=10)

        # Check required properties
        assert hasattr(encoder, "batch_size")
        assert isinstance(encoder.batch_size, int)

        # Check required methods
        assert hasattr(encoder, "encode")
        assert callable(encoder.encode)
