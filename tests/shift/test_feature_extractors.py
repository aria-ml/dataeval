"""
Tests for drift feature extractors.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from dataeval import Embeddings, Metadata
from dataeval.flags import ImageStats
from dataeval.shift._drift._univariate import DriftUnivariate
from dataeval.shift._feature_extractors import (
    EmbeddingsFeatureExtractor,
    MetadataFeatureExtractor,
    UncertaintyFeatureExtractor,
    _classifier_uncertainty,
)


class SimpleModel(nn.Module):
    def __init__(self, n_features, n_output):
        super().__init__()
        self.fc = nn.Linear(n_features, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@pytest.mark.required
class TestEmbeddingsFeatureExtractor:
    """Test EmbeddingsFeatureExtractor for caching and rehydration."""

    def test_basic_extraction(self):
        """Test basic embedding extraction from arrays."""
        model = SimpleModel(10, 5)
        extractor = EmbeddingsFeatureExtractor(model=model, batch_size=32)

        # Create simple array "dataset" (treating as data)
        data = np.random.randn(100, 10).astype(np.float32)

        # Extract embeddings
        embeddings = extractor(data)

        assert embeddings.shape == (100, 5)
        assert isinstance(embeddings, np.ndarray)

    def test_caching_same_dataset(self):
        """Test that same dataset object is cached and not re-extracted."""
        model = SimpleModel(10, 5)
        extractor = EmbeddingsFeatureExtractor(model=model, batch_size=32)

        data = np.random.randn(100, 10).astype(np.float32)

        # First extraction
        embeddings1 = extractor(data)

        # Second extraction of same object - should be cached
        embeddings2 = extractor(data)

        # Should be exact same array (from cache)
        assert embeddings1 is embeddings2
        np.testing.assert_array_equal(embeddings1, embeddings2)

    def test_different_datasets(self):
        """Test that different datasets are extracted separately."""
        model = SimpleModel(10, 5)
        extractor = EmbeddingsFeatureExtractor(model=model, batch_size=32)

        data1 = np.random.randn(100, 10).astype(np.float32)
        data2 = np.random.randn(100, 10).astype(np.float32)

        embeddings1 = extractor(data1)
        embeddings2 = extractor(data2)

        # Should be different arrays (different data)
        assert embeddings1 is not embeddings2
        # Values should be different (different data through same model)
        assert not np.allclose(embeddings1, embeddings2)

    def test_with_drift_detector(self):
        """Test integration with DriftUnivariate."""
        model = SimpleModel(10, 5)
        extractor = EmbeddingsFeatureExtractor(model=model, batch_size=32)

        # Reference data
        ref_data = np.random.randn(100, 10).astype(np.float32)

        # Create drift detector
        detector = DriftUnivariate(data=ref_data, method="ks", feature_extractor=extractor)

        # Test data (similar to reference)
        test_data = np.random.randn(50, 10).astype(np.float32)
        result = detector.predict(test_data)

        assert hasattr(result, "drifted")
        assert hasattr(result, "distances")


@pytest.mark.required
class TestFeatureExtractorRehydration:
    """Test that pre-computed Embeddings/Metadata can be reused."""

    def test_embeddings_rehydration_concept(self):
        """Test the rehydration pattern conceptually."""

        model = SimpleModel(10, 5)

        # Simulate a dataset-like object (simple wrapper)
        class SimpleDataset:
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dataset = SimpleDataset(np.random.randn(100, 10).astype(np.float32))

        # Pre-compute embeddings
        pre_embeddings = Embeddings(dataset=dataset, batch_size=32, model=model).compute()

        # Create extractor with rehydration
        extractor = EmbeddingsFeatureExtractor(embeddings=pre_embeddings)

        # When we pass the same dataset, it should use cached embeddings
        result = extractor(dataset)

        assert result.shape == (100, 5)
        # Should be using the cached embeddings
        assert len(extractor._dataset_cache) > 0


@pytest.mark.required
class TestClassifierUncertainty:
    """Test _classifier_uncertainty function."""

    def test_probs_input(self):
        """Test uncertainty calculation with probability input."""
        # Create mock probabilities (3 samples, 4 classes)
        probs = np.array([[0.7, 0.2, 0.05, 0.05], [0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0]])

        result = _classifier_uncertainty(probs, preds_type="probs")

        assert result.shape == (3, 1)
        assert isinstance(result, torch.Tensor)
        # Uniform distribution has highest entropy
        assert result[1, 0] > result[0, 0]
        assert result[1, 0] > result[2, 0]

    def test_logits_input(self):
        """Test uncertainty calculation with logits input."""
        # Create mock logits
        logits = np.array([[2.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [10.0, 0.0, 0.0, 0.0]])

        result = _classifier_uncertainty(logits, preds_type="logits")

        assert result.shape == (3, 1)
        assert isinstance(result, torch.Tensor)
        # All values should be non-negative
        assert (result >= 0).all()

    def test_invalid_probs_sum(self):
        """Test that invalid probabilities raise ValueError."""
        # Probabilities that don't sum to 1
        bad_probs = np.array([[0.5, 0.2, 0.1]])

        with pytest.raises(ValueError, match="Probabilities across labels should sum to 1"):
            _classifier_uncertainty(bad_probs, preds_type="probs")

    def test_invalid_preds_type(self):
        """Test that invalid preds_type raises NotImplementedError."""
        probs = np.array([[0.5, 0.5]])

        with pytest.raises(NotImplementedError, match="Only prediction types 'probs' and 'logits' supported"):
            _classifier_uncertainty(probs, preds_type="invalid")  # type: ignore


@pytest.mark.required
class TestEmbeddingsFeatureExtractorErrors:
    """Test error handling in EmbeddingsFeatureExtractor."""

    def test_no_model_no_embeddings(self):
        """Test that missing both model and embeddings raises ValueError."""
        with pytest.raises(ValueError, match="Either model or embeddings must be provided"):
            EmbeddingsFeatureExtractor()

    def test_invalid_embeddings_type(self):
        """Test that invalid embeddings type raises TypeError."""
        with pytest.raises(TypeError, match="embeddings must be an Embeddings instance"):
            EmbeddingsFeatureExtractor(embeddings="not_an_embeddings_object")  # type: ignore

    def test_repr(self):
        """Test __repr__ method."""
        model = SimpleModel(10, 5)
        extractor = EmbeddingsFeatureExtractor(model=model, batch_size=32)

        repr_str = repr(extractor)

        assert "EmbeddingsFeatureExtractor" in repr_str
        assert "SimpleModel" in repr_str
        assert "batch_size=32" in repr_str


@pytest.mark.required
class TestMetadataFeatureExtractor:
    """Test MetadataFeatureExtractor."""

    def test_basic_extraction_with_mock(self):
        """Test basic metadata extraction using mock."""
        # Create mock dataset
        mock_dataset = Mock()
        mock_metadata = Mock(spec=Metadata)
        mock_metadata.binned_data = np.random.randint(0, 5, (100, 3))

        with patch("dataeval.shift._feature_extractors.Metadata", return_value=mock_metadata):
            extractor = MetadataFeatureExtractor(use_binned=True)
            result = extractor(mock_dataset)

        assert result.shape == (100, 3)

    def test_use_binned_false(self):
        """Test extraction with use_binned=False."""
        mock_dataset = Mock()
        mock_metadata = Mock(spec=Metadata)
        mock_metadata.factor_data = np.random.randn(100, 3)

        with patch("dataeval.shift._feature_extractors.Metadata", return_value=mock_metadata):
            extractor = MetadataFeatureExtractor(use_binned=False)
            result = extractor(mock_dataset)

        assert result.shape == (100, 3)

    def test_invalid_metadata_type(self):
        """Test that invalid metadata type raises TypeError."""
        with pytest.raises(TypeError, match="metadata must be a Metadata instance"):
            MetadataFeatureExtractor(metadata="not_a_metadata_object")  # type: ignore

    def test_with_add_stats(self):
        """Test extraction with add_stats parameter."""
        mock_dataset = Mock()
        mock_metadata = Mock(spec=Metadata)
        mock_metadata.binned_data = np.random.randint(0, 5, (50, 2))
        mock_stats = {"stats": {"brightness": np.random.rand(50)}}

        with (
            patch("dataeval.shift._feature_extractors.Metadata", return_value=mock_metadata),
            patch("dataeval.shift._feature_extractors.calculate", return_value=mock_stats),
        ):
            extractor = MetadataFeatureExtractor(use_binned=True, add_stats=ImageStats.VISUAL_BRIGHTNESS)
            result = extractor(mock_dataset)

        assert result.shape == (50, 2)
        # Verify add_factors was called
        mock_metadata.add_factors.assert_called_once()

    def test_caching_same_dataset(self):
        """Test that same dataset is cached."""
        mock_dataset = Mock()
        mock_metadata = Mock(spec=Metadata)
        mock_metadata.binned_data = np.random.randint(0, 5, (100, 3))

        with patch("dataeval.shift._feature_extractors.Metadata", return_value=mock_metadata) as mock_cls:
            extractor = MetadataFeatureExtractor(use_binned=True)
            result1 = extractor(mock_dataset)
            result2 = extractor(mock_dataset)

        # Should be same cached object
        assert result1 is result2
        # Metadata should only be created once
        assert mock_cls.call_count == 1

    def test_metadata_rehydration(self):
        """Test using pre-computed metadata."""
        mock_dataset = Mock()
        mock_metadata = Mock(spec=Metadata)
        mock_metadata._dataset = mock_dataset
        mock_metadata.binned_data = np.random.randint(0, 5, (100, 3))
        mock_metadata.factor_data = np.random.randn(100, 3)
        mock_metadata.continuous_factor_bins = {"brightness": 10}
        mock_metadata.auto_bin_method = "uniform_width"
        mock_metadata.exclude = set()
        mock_metadata.include = set()

        extractor = MetadataFeatureExtractor(metadata=mock_metadata, use_binned=True)
        result = extractor(mock_dataset)

        # Should use the cached metadata
        assert result.shape == (100, 3)
        assert result is mock_metadata.binned_data

    def test_repr(self):
        """Test __repr__ method."""
        extractor = MetadataFeatureExtractor(use_binned=False, auto_bin_method="uniform_count")

        repr_str = repr(extractor)

        assert "MetadataFeatureExtractor" in repr_str
        assert "use_binned=False" in repr_str
        assert "uniform_count" in repr_str


@pytest.mark.required
class TestUncertaintyFeatureExtractor:
    """Test UncertaintyFeatureExtractor."""

    def test_basic_extraction_with_mock(self):
        """Test basic uncertainty extraction using mock."""
        model = SimpleModel(10, 4)
        data = np.random.randn(50, 10).astype(np.float32)

        # Mock the predict function to return probabilities
        mock_probs = np.array([[0.7, 0.2, 0.05, 0.05]] * 50)

        with patch("dataeval.shift._feature_extractors.predict", return_value=mock_probs):
            extractor = UncertaintyFeatureExtractor(model=model, preds_type="probs", batch_size=16)
            result = extractor(data)

        assert result.shape == (50, 1)
        assert isinstance(result, np.ndarray)

    def test_with_logits(self):
        """Test extraction with logits."""
        model = SimpleModel(10, 4)
        data = np.random.randn(20, 10).astype(np.float32)

        mock_logits = np.random.randn(20, 4)

        with patch("dataeval.shift._feature_extractors.predict", return_value=mock_logits):
            extractor = UncertaintyFeatureExtractor(model=model, preds_type="logits", batch_size=8)
            result = extractor(data)

        assert result.shape == (20, 1)

    def test_with_transforms(self):
        """Test extraction with transforms."""
        model = SimpleModel(10, 4)
        data = np.random.randn(30, 10).astype(np.float32)

        def transform_fn(x):
            return x * 2.0

        mock_probs = np.array([[0.25, 0.25, 0.25, 0.25]] * 30)

        with patch("dataeval.shift._feature_extractors.predict", return_value=mock_probs):
            extractor = UncertaintyFeatureExtractor(
                model=model, preds_type="probs", transforms=transform_fn, device="cpu"
            )
            result = extractor(data)

        assert result.shape == (30, 1)

    def test_apply_transforms(self):
        """Test _apply_transforms method."""
        model = SimpleModel(10, 4)

        def transform1(x):
            return x * 2.0

        def transform2(x):
            return x + 1.0

        extractor = UncertaintyFeatureExtractor(model=model, transforms=[transform1, transform2])

        x = torch.tensor([1.0, 2.0, 3.0])
        result = extractor._apply_transforms(x)

        # Should apply both transforms: (x * 2) + 1
        expected = torch.tensor([3.0, 5.0, 7.0])
        assert torch.allclose(result, expected)

    def test_repr(self):
        """Test __repr__ method."""
        model = SimpleModel(10, 4)
        extractor = UncertaintyFeatureExtractor(model=model, preds_type="logits", batch_size=64)

        repr_str = repr(extractor)

        assert "UncertaintyFeatureExtractor" in repr_str
        assert "SimpleModel" in repr_str
        assert "preds_type='logits'" in repr_str
        assert "batch_size=64" in repr_str
