"""Tests for TorchExtractor."""

import numpy as np
import pytest
import torch

from dataeval.extractors import TorchExtractor
from dataeval.protocols import FeatureExtractor


@pytest.mark.required
class TestTorchExtractorInit:
    """Test TorchExtractor initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        model = torch.nn.Flatten()
        extractor = TorchExtractor(model)
        assert extractor.device is not None
        assert extractor.layer_name is None
        assert extractor.use_output is True

    def test_init_with_device(self):
        """Test initialization with specified device."""
        model = torch.nn.Flatten()
        extractor = TorchExtractor(model, device="cpu")
        assert extractor.device == torch.device("cpu")

    def test_init_with_layer_name(self):
        """Test initialization with layer extraction."""
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(768, 128),
        )
        extractor = TorchExtractor(model, layer_name="0")
        assert extractor.layer_name == "0"

    def test_init_with_invalid_layer_raises(self):
        """Test that invalid layer name raises ValueError."""
        model = torch.nn.Flatten()
        with pytest.raises(ValueError, match="Invalid layer"):
            TorchExtractor(model, layer_name="nonexistent")

    def test_init_with_transforms(self):
        """Test initialization with transforms."""

        class MockTransform:
            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        model = torch.nn.Flatten()
        extractor = TorchExtractor(model, transforms=MockTransform())
        assert len(extractor._transforms) == 1

    def test_init_with_multiple_transforms(self):
        """Test initialization with multiple transforms."""

        class ScaleTransform:
            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        class ShiftTransform:
            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        model = torch.nn.Flatten()
        extractor = TorchExtractor(model, transforms=[ScaleTransform(), ShiftTransform()])
        assert len(extractor._transforms) == 2


@pytest.mark.required
class TestTorchExtractorCall:
    """Test TorchExtractor.__call__ method."""

    @pytest.fixture
    def extractor(self):
        """Create a simple extractor for testing."""
        return TorchExtractor(torch.nn.Flatten(), device="cpu")

    def test_call_batch_of_images(self, extractor):
        """Test extracting features from a batch of images."""
        images = [torch.randn(3, 16, 16) for _ in range(5)]
        result = extractor(images)
        assert result.shape[0] == 5
        assert result.shape[1] == 3 * 16 * 16

    def test_call_single_image(self, extractor):
        """Test extracting features from a single image."""
        images = [torch.randn(3, 16, 16)]
        result = extractor(images)
        assert result.shape[0] == 1
        assert result.shape[1] == 3 * 16 * 16

    def test_call_empty_list(self, extractor):
        """Test extracting features from empty list."""
        result = extractor([])
        assert result.shape[0] == 0

    def test_call_with_numpy_input(self, extractor):
        """Test extracting features from numpy arrays."""
        images = [np.random.randn(3, 8, 8).astype(np.float32) for _ in range(3)]
        result = extractor(images)
        assert result.shape[0] == 3
        assert result.shape[1] == 3 * 8 * 8


class TestTorchExtractorLayerExtraction:
    """Test layer extraction functionality."""

    def test_extract_intermediate_layer_output(self):
        """Test extracting output from intermediate layer."""
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(768, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )
        extractor = TorchExtractor(model, layer_name="1", device="cpu")

        images = [torch.randn(3, 16, 16) for _ in range(5)]
        result = extractor(images)
        assert result.shape == (5, 128)

    def test_extract_intermediate_layer_input(self):
        """Test extracting input to intermediate layer."""
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(768, 128),
            torch.nn.ReLU(),
        )
        extractor = TorchExtractor(model, layer_name="1", use_output=False, device="cpu")

        images = [torch.randn(3, 16, 16) for _ in range(5)]
        result = extractor(images)
        assert result.shape == (5, 768)


@pytest.mark.required
class TestTorchExtractorTransforms:
    """Test transform functionality."""

    def test_transforms_applied(self):
        """Test that transforms are applied during extraction."""

        class DoubleTransform:
            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        model = torch.nn.Flatten()
        extractor_no_transform = TorchExtractor(model, device="cpu")
        extractor_with_transform = TorchExtractor(model, transforms=DoubleTransform(), device="cpu")

        images = [torch.ones(1, 4, 4) for _ in range(3)]

        result_no_transform = np.asarray(extractor_no_transform(images))
        result_with_transform = np.asarray(extractor_with_transform(images))

        # With transform, values should be doubled
        np.testing.assert_array_almost_equal(result_with_transform, result_no_transform * 2)


@pytest.mark.required
class TestTorchExtractorRepr:
    """Test __repr__ method."""

    def test_repr_basic(self):
        """Test basic repr."""
        model = torch.nn.Flatten()
        extractor = TorchExtractor(model, device="cpu")
        repr_str = repr(extractor)
        assert "TorchExtractor" in repr_str
        assert "cpu" in repr_str

    def test_repr_with_layer_name(self):
        """Test repr includes layer name when set."""
        model = torch.nn.Sequential(torch.nn.Flatten())
        extractor = TorchExtractor(model, layer_name="0", device="cpu")
        repr_str = repr(extractor)
        assert "layer_name='0'" in repr_str


@pytest.mark.required
class TestTorchExtractorProtocol:
    """Test that TorchExtractor conforms to FeatureExtractor protocol."""

    def test_protocol_conformance(self):
        """Test that TorchExtractor implements FeatureExtractor protocol."""
        model = torch.nn.Flatten()
        extractor = TorchExtractor(model)
        assert isinstance(extractor, FeatureExtractor)
        assert callable(extractor)
