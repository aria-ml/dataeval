"""Verify that Metadata and Embeddings classes function correctly.

Maps to meta repo test cases:
  - TC-9.1: Metadata and embeddings management
"""

import numpy as np


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
        assert hasattr(meta, "is_binned")

    def test_metadata_od_continuous_bins_and_exclude(self):
        """Metadata on an OD dataset bins a continuous factor as requested and drops excluded factors."""
        from dataeval import Metadata
        from verification.helpers import SimpleODDataset

        n = 12
        images = np.random.default_rng(0).random((n, 3, 16, 16)).astype(np.float32)
        labels = [np.array([i % 3, (i + 1) % 3]) for i in range(n)]
        factors = [{"brightness": float(i), "weather": "sun" if i % 2 else "rain"} for i in range(n)]
        dataset = SimpleODDataset(images, labels, factors)

        metadata = Metadata(dataset, continuous_factor_bins={"brightness": 3}, exclude=["weather"])  # pyright: ignore[reportArgumentType]

        assert "weather" not in metadata.factor_names
        assert "brightness" in metadata.factor_names
        assert metadata.continuous_factor_bins == {"brightness": 3}
        assert metadata.factor_data.shape[0] == 2 * n  # one row per detection
        column = metadata.factor_data[:, list(metadata.factor_names).index("brightness")]
        assert len(np.unique(column)) == 3

        # Without exclude the categorical factor is present (the exclusion is what removed it).
        assert "weather" in Metadata(dataset, continuous_factor_bins={"brightness": 3}).factor_names  # pyright: ignore[reportArgumentType]

    def test_embeddings_disk_cache_reused_without_recomputing(self, tmp_path):
        """With a disk-backed cache, a second access is served from the cache, not the extractor."""
        from dataeval import Embeddings
        from dataeval.extractors import FlattenExtractor

        class CountingExtractor:
            def __init__(self):
                self.calls = 0
                self._inner = FlattenExtractor()

            def __call__(self, data):
                self.calls += 1
                return self._inner(data)

        images = np.random.default_rng(0).random((10, 3, 8, 8)).astype(np.float32)
        extractor = CountingExtractor()
        # memory_threshold=0 forces the memory-mapped (on-disk) storage path.
        embeddings = Embeddings(
            images, extractor=extractor, batch_size=5, path=tmp_path / "emb.npy", memory_threshold=0.0
        )

        first = np.array(embeddings[:])
        calls_after_first = extractor.calls
        assert calls_after_first == 2  # 10 images / batch of 5
        assert (tmp_path / "emb.npy").exists()
        assert isinstance(embeddings._embeddings, np.memmap)

        second = np.array(embeddings[:])
        _ = embeddings[3]
        assert extractor.calls == calls_after_first
        np.testing.assert_array_equal(first, second)

    def test_embeddings_bind_is_lazy(self):
        """Binding a dataset computes nothing until embeddings are accessed."""
        from dataeval import Embeddings

        class CountingExtractor:
            calls = 0

            def __call__(self, data):
                type(self).calls += 1
                return np.asarray(data).reshape(len(data), -1)

        images = np.random.default_rng(0).random((6, 3, 4, 4)).astype(np.float32)
        embeddings = Embeddings(extractor=CountingExtractor(), batch_size=6)
        assert not embeddings.is_bound

        embeddings.bind(images)
        assert embeddings.is_bound
        assert CountingExtractor.calls == 0

        assert embeddings().shape == (6, 48)
        assert CountingExtractor.calls == 1

    def test_embeddings_satisfies_array_and_feature_extractor_protocols(self):
        from dataeval import Embeddings
        from dataeval.protocols import Array, FeatureExtractor

        def accepts_array(x: Array) -> tuple[int, ...]:
            assert isinstance(x, Array)
            return tuple(np.asarray(x).shape)

        def accepts_extractor(x: FeatureExtractor, data) -> int:
            assert isinstance(x, FeatureExtractor)
            return len(np.asarray(x(data)))

        images = np.random.default_rng(0).random((6, 3, 4, 4)).astype(np.float32)
        assert accepts_array(Embeddings(images, batch_size=6)) == (6, 48)
        assert accepts_extractor(Embeddings(batch_size=6), images) == 6
