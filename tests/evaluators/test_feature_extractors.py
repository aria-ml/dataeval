"""
Tests for drift feature extractors.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from dataeval.evaluators.drift import DriftUnivariate, EmbeddingsFeatureExtractor, MetadataFeatureExtractor


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
        from dataeval import Embeddings

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
class TestFeatureExtractorExports:
    """Test that feature extractors are properly exported."""

    def test_imports_from_main_module(self):
        """Test that extractors can be imported from main drift module."""
        from dataeval.evaluators.drift import (
            EmbeddingsFeatureExtractor,
            UncertaintyFeatureExtractor,
        )

        assert EmbeddingsFeatureExtractor is not None
        assert MetadataFeatureExtractor is not None
        assert UncertaintyFeatureExtractor is not None

    def test_imports_from_feature_extractors_module(self):
        """Test that extractors can be imported from feature_extractors submodule."""
        from dataeval.evaluators.drift.feature_extractors import (
            EmbeddingsFeatureExtractor,
            MetadataFeatureExtractor,
            UncertaintyFeatureExtractor,
            _classifier_uncertainty,
        )

        assert EmbeddingsFeatureExtractor is not None
        assert MetadataFeatureExtractor is not None
        assert UncertaintyFeatureExtractor is not None
        assert _classifier_uncertainty is not None
