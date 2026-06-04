"""batch_size resolution + I/O chunking through Embeddings."""

import numpy as np
import pytest

from dataeval import Embeddings
from dataeval.protocols import Array


class RecordingExtractor:
    """A FeatureExtractor that records the size of each chunk it is handed."""

    def __init__(self, batch_size=None):
        self.batch_size = batch_size
        self.chunk_sizes: list[int] = []

    def __call__(self, data) -> Array:
        items = list(data)
        self.chunk_sizes.append(len(items))
        return np.asarray([np.ravel(np.asarray(d)) for d in items], dtype=np.float32)


class ListDataset:
    def __init__(self, n, dim):
        self._items = [np.random.randn(dim).astype(np.float32) for _ in range(n)]

    def __getitem__(self, i):
        return self._items[i]

    def __len__(self):
        return len(self._items)


@pytest.mark.required
class TestEmbeddingsBatchSizeResolution:
    def test_embeddings_batch_size_sets_io_chunks(self):
        """Embeddings.batch_size sizes the I/O chunks handed to the extractor."""
        ext = RecordingExtractor(batch_size=2)
        emb = Embeddings(ListDataset(6, 4), extractor=ext, batch_size=6)
        emb.compute()  # materializes all embeddings, triggering extractor calls
        # Embeddings.batch_size wins for its own chunk loop: one chunk of all 6.
        assert emb.batch_size == 6
        assert ext.chunk_sizes == [6]

    def test_falls_back_to_extractor_batch_size(self):
        """When Embeddings.batch_size is None, it falls back to the extractor's."""
        ext = RecordingExtractor(batch_size=3)
        emb = Embeddings(ListDataset(6, 4), extractor=ext, batch_size=None)
        emb.compute()
        # Falls back to extractor's 3, so the dataset is chunked into 3 + 3.
        assert emb.batch_size == 3
        assert ext.chunk_sizes == [3, 3]
