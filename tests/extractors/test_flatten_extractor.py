"""Tests for FlattenExtractor."""

import numpy as np
import pytest

from dataeval.extractors import FlattenExtractor
from dataeval.protocols import FeatureExtractor


@pytest.mark.required
class TestFlattenExtractorCall:
    """Test FlattenExtractor.__call__ method."""

    @pytest.fixture
    def extractor(self):
        """Create an extractor for testing."""
        return FlattenExtractor()

    def test_call_batch_of_images(self, extractor):
        """Test extracting features from a batch of images."""
        images = [np.random.randn(3, 16, 16) for _ in range(5)]
        result = extractor(images)
        assert result.shape[0] == 5
        assert result.shape[1] == 3 * 16 * 16  # Flattened image

    def test_call_single_image(self, extractor):
        """Test extracting features from a single image."""
        images = [np.random.randn(3, 16, 16)]
        result = extractor(images)
        assert result.shape[0] == 1
        assert result.shape[1] == 3 * 16 * 16

    def test_call_empty_list(self, extractor):
        """Test extracting features from empty list."""
        result = extractor([])
        assert result.shape[0] == 0

    def test_flatten_2d_image(self, extractor):
        """Test flattening 2D images (grayscale)."""
        images = [np.random.randn(8, 8) for _ in range(5)]
        result = extractor(images)
        assert result.shape == (5, 64)

    def test_flatten_3d_image(self, extractor):
        """Test flattening 3D images (RGB)."""
        images = [np.random.randn(3, 8, 8) for _ in range(5)]
        result = extractor(images)
        assert result.shape == (5, 3 * 8 * 8)

    def test_flatten_preserves_values(self, extractor):
        """Test that flattening preserves image values."""
        image = np.arange(64).reshape(8, 8).astype(np.float64)
        result = extractor([image])
        # Values should be preserved (in some order)
        np.testing.assert_array_almost_equal(np.sort(result.flatten()), np.sort(image.flatten()))


@pytest.mark.required
class TestFlattenExtractorRepr:
    """Test __repr__ method."""

    def test_repr(self):
        """Test repr output."""
        extractor = FlattenExtractor()
        repr_str = repr(extractor)
        assert "FlattenExtractor" in repr_str


@pytest.mark.required
class TestFlattenExtractorProtocol:
    """Test that FlattenExtractor conforms to FeatureExtractor protocol."""

    def test_protocol_conformance(self):
        """Test that FlattenExtractor implements FeatureExtractor protocol."""
        extractor = FlattenExtractor()
        assert isinstance(extractor, FeatureExtractor)
        assert callable(extractor)
