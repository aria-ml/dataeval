"""Verify that Metadata and Embeddings classes function correctly.

Maps to meta repo test cases:
  - TC-9.1: Metadata and embeddings management
"""

import numpy as np
import pytest


@pytest.mark.test_case("9-1")
class TestMetadataEmbeddings:
    """Verify Metadata and Embeddings top-level classes."""

    def test_metadata_class_importable(self):
        from dataeval import Metadata  # noqa: F401

    def test_embeddings_class_importable(self):
        from dataeval import Embeddings  # noqa: F401

    def test_embeddings_with_flatten_extractor(self):
        from dataeval import Embeddings
        from dataeval.extractors import FlattenExtractor

        images = np.random.default_rng(0).random((10, 3, 8, 8)).astype(np.float32)
        embeddings = Embeddings(images, extractor=FlattenExtractor(), batch_size=10)
        result = np.asarray(embeddings)
        assert result.shape == (10, 3 * 8 * 8)

    def test_embeddings_supports_len(self):
        from dataeval import Embeddings
        from dataeval.extractors import FlattenExtractor

        images = np.random.default_rng(0).random((10, 3, 8, 8)).astype(np.float32)
        embeddings = Embeddings(images, extractor=FlattenExtractor(), batch_size=10)
        assert len(embeddings) == 10

    def test_embeddings_supports_indexing(self):
        from dataeval import Embeddings
        from dataeval.extractors import FlattenExtractor

        images = np.random.default_rng(0).random((10, 3, 8, 8)).astype(np.float32)
        embeddings = Embeddings(images, extractor=FlattenExtractor(), batch_size=10)
        single = embeddings[0]
        assert single is not None

    def test_metadata_protocol_attributes(self):
        """Verify that a Metadata-protocol object has the expected properties."""
        from verification.helpers import make_metadata

        meta = make_metadata()
        assert hasattr(meta, "factor_names")
        assert hasattr(meta, "factor_data")
        assert hasattr(meta, "class_labels")
        assert hasattr(meta, "is_discrete")
