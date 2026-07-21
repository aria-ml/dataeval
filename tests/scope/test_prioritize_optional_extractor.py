"""Tests that ``Prioritize`` treats ``extractor`` as optional at construction.

Mirrors the sibling ``Coverage`` contract: an extractor is only required when
``evaluate`` is handed something that must be embedded. When pre-computed
embeddings / an ``Array`` are supplied, no extractor is needed.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from dataeval.protocols import AnnotatedDataset
from dataeval.scope._prioritize import Prioritize, PrioritizeOutput


def _dataset_like(n: int = 50):
    """A minimal AnnotatedDataset-like object that is NOT an ``Array``.

    Passing this to ``evaluate`` forces the embedding-extraction code path.
    """
    mock = MagicMock(spec=AnnotatedDataset)
    mock.__len__.return_value = n
    mock.__getitem__.side_effect = lambda _: (np.random.random((1, 10, 10)), np.zeros(10), {})
    mock.metadata = {"id": "mock_dataset", "index2label": {i: str(i) for i in range(10)}}
    return mock


class TestPrioritizeOptionalExtractor:
    def test_construct_without_extractor_and_evaluate_precomputed_array(self):
        """No extractor + pre-computed embeddings array must produce a PrioritizeOutput."""
        embeddings = np.random.default_rng(0).random((50, 16)).astype(np.float32)
        class_labels = np.random.default_rng(1).integers(low=0, high=5, size=50)

        prioritizer = Prioritize()  # no extractor supplied
        result = prioritizer.evaluate(embeddings, class_labels=class_labels)

        assert isinstance(result, PrioritizeOutput)
        assert len(result.indices) == 50

    def test_factories_construct_without_extractor(self):
        """Every factory classmethod must be constructible with no extractor."""
        assert isinstance(Prioritize.knn(), Prioritize)
        assert isinstance(Prioritize.kmeans_distance(), Prioritize)
        assert isinstance(Prioritize.kmeans_complexity(), Prioritize)
        assert isinstance(Prioritize.hdbscan_distance(), Prioritize)
        assert isinstance(Prioritize.hdbscan_complexity(), Prioritize)

    def test_factory_precomputed_array_without_extractor(self):
        """A factory-built instance with no extractor evaluates pre-computed embeddings."""
        embeddings = np.random.default_rng(2).random((50, 16)).astype(np.float32)
        result = Prioritize.knn(k=5).evaluate(embeddings)
        assert isinstance(result, PrioritizeOutput)
        assert len(result.indices) == 50

    def test_evaluate_dataset_without_extractor_raises_valueerror(self):
        """A dataset that needs extraction + no extractor must raise a clear ValueError."""
        prioritizer = Prioritize()  # no extractor supplied
        with pytest.raises(ValueError, match="extractor"):
            prioritizer.evaluate(_dataset_like(50))
