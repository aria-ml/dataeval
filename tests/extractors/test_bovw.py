"""Tests for BoVWExtractor (Bag of Visual Words)."""

import numpy as np
import pytest

from dataeval.config import get_seed, set_seed
from dataeval.extractors._bovw import BoVWExtractor


@pytest.mark.optional
class TestBoVWExtractor:
    """Tests for BoVWExtractor class."""

    @pytest.fixture
    def rgb_images(self):
        """Create simple RGB test images with some texture for SIFT detection."""
        rng = np.random.default_rng(42)
        images = []
        for _ in range(5):
            # Create an image with some structure (not uniform) for SIFT to detect
            img = rng.integers(0, 256, size=(3, 64, 64), dtype=np.uint8)
            images.append(img)
        return images

    @pytest.fixture
    def grayscale_images(self):
        """Create simple grayscale test images with texture."""
        rng = np.random.default_rng(42)
        images = []
        for _ in range(5):
            img = rng.integers(0, 256, size=(1, 64, 64), dtype=np.uint8)
            images.append(img)
        return images

    @pytest.fixture
    def float_images(self):
        """Create float images normalized to [0, 1]."""
        rng = np.random.default_rng(42)
        images = []
        for _ in range(5):
            img = rng.random((3, 64, 64)).astype(np.float32)
            images.append(img)
        return images

    @pytest.fixture
    def tuple_dataset(self, rgb_images):
        """Create dataset where items are (image, label) tuples."""
        return [(img, i) for i, img in enumerate(rgb_images)]

    def test_init_default_vocab_size(self):
        """Test default vocabulary size initialization."""
        extractor = BoVWExtractor()
        assert extractor.vocab_size == 2048
        assert extractor._kmeans is None

    def test_init_custom_vocab_size(self):
        """Test custom vocabulary size initialization."""
        extractor = BoVWExtractor(vocab_size=512)
        assert extractor.vocab_size == 512

    def test_init_invalid_vocab_size(self):
        """Test that invalid vocab_size raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be at least 1"):
            BoVWExtractor(vocab_size=0)
        with pytest.raises(ValueError, match="vocab_size must be at least 1"):
            BoVWExtractor(vocab_size=-1)

    def test_fit_creates_kmeans(self, rgb_images):
        """Test that fit creates the kmeans model."""
        extractor = BoVWExtractor(vocab_size=32)
        assert extractor._kmeans is None
        extractor.fit(rgb_images)
        assert extractor._kmeans is not None

    def test_transform_before_fit_raises(self, rgb_images):
        """Test that transform before fit raises RuntimeError."""
        extractor = BoVWExtractor(vocab_size=32)
        with pytest.raises(RuntimeError, match="Extractor has not been fitted"):
            extractor.transform(rgb_images)

    def test_fit_transform_separate(self, rgb_images):
        """Test separate fit and transform calls."""
        extractor = BoVWExtractor(vocab_size=32)
        extractor.fit(rgb_images)
        embeddings = extractor.transform(rgb_images)

        assert embeddings.shape[0] == len(rgb_images)
        assert embeddings.shape[1] <= 32

    def test_transform_new_images(self, rgb_images):
        """Test transforming new images with fitted vocabulary."""
        rng = np.random.default_rng(123)
        new_images = [rng.integers(0, 256, size=(3, 64, 64), dtype=np.uint8) for _ in range(3)]

        extractor = BoVWExtractor(vocab_size=32)
        extractor.fit(rgb_images)
        embeddings = extractor.transform(new_images)

        assert embeddings.shape[0] == len(new_images)
        assert extractor._kmeans is not None
        assert embeddings.shape[1] == extractor._kmeans.n_clusters  # type: ignore

    def test_extract_rgb_images(self, rgb_images):
        """Test feature extraction from RGB images using __call__."""
        extractor = BoVWExtractor(vocab_size=32)
        embeddings = extractor(rgb_images)

        assert embeddings.shape[0] == len(rgb_images)
        assert embeddings.shape[1] <= 32
        assert extractor._kmeans is not None

    def test_extract_grayscale_images(self, grayscale_images):
        """Test feature extraction from grayscale images."""
        extractor = BoVWExtractor(vocab_size=32)
        embeddings = extractor(grayscale_images)

        assert embeddings.shape[0] == len(grayscale_images)
        assert extractor._kmeans is not None

    def test_extract_float_images(self, float_images):
        """Test feature extraction from float [0,1] normalized images."""
        extractor = BoVWExtractor(vocab_size=32)
        embeddings = extractor(float_images)

        assert embeddings.shape[0] == len(float_images)
        assert extractor._kmeans is not None

    def test_extract_tuple_dataset(self, tuple_dataset):
        """Test feature extraction from (image, label) tuple dataset."""
        extractor = BoVWExtractor(vocab_size=32)
        embeddings = extractor(tuple_dataset)

        assert embeddings.shape[0] == len(tuple_dataset)

    def test_embeddings_are_normalized(self, rgb_images):
        """Test that embeddings are L2 normalized."""
        extractor = BoVWExtractor(vocab_size=32)
        embeddings = extractor(rgb_images)

        for emb in embeddings:
            norm = np.linalg.norm(emb)
            # Norm should be 1.0 or 0.0 (for images with no features)
            assert np.isclose(norm, 1.0) or np.isclose(norm, 0.0)

    def test_vocab_size_exceeds_descriptors(self, rgb_images):
        """Test that vocab_size is capped at number of descriptors."""
        extractor = BoVWExtractor(vocab_size=100000)  # Very large vocab
        embeddings = extractor(rgb_images)

        # Should work without error, clusters capped at descriptor count
        assert embeddings.shape[0] == len(rgb_images)
        assert embeddings.shape[1] < 100000  # Should be much smaller

    def test_empty_dataset_raises(self):
        """Test that empty dataset raises appropriate error."""
        extractor = BoVWExtractor(vocab_size=32)

        with pytest.raises((ValueError, IndexError)):
            extractor([])

    def test_uniform_images_may_have_no_features(self):
        """Test handling of uniform images that may have no SIFT features."""
        # Create completely uniform images - SIFT won't find features
        uniform_images = [np.zeros((3, 64, 64), dtype=np.uint8) for _ in range(5)]

        extractor = BoVWExtractor(vocab_size=32)

        # Should raise ValueError because no features found
        with pytest.raises(ValueError, match="No SIFT features found"):
            extractor(uniform_images)

    def test_mixed_feature_images(self, rgb_images):
        """Test dataset with mix of feature-rich and uniform images."""
        # Add one uniform image to the dataset
        mixed_images = rgb_images + [np.zeros((3, 64, 64), dtype=np.uint8)]

        extractor = BoVWExtractor(vocab_size=32)
        embeddings = extractor(mixed_images)

        assert embeddings.shape[0] == len(mixed_images)
        # Last image (uniform) should have zero histogram
        assert np.allclose(embeddings[-1], 0.0)

    def test_reproducibility_with_seed(self, rgb_images):
        """Test that results are reproducible when seed is set."""
        orig = get_seed()

        set_seed(42)
        extractor1 = BoVWExtractor(vocab_size=32)
        emb1 = extractor1(rgb_images)

        set_seed(42)
        extractor2 = BoVWExtractor(vocab_size=32)
        emb2 = extractor2(rgb_images)

        # With same seed, results should be identical
        np.testing.assert_array_almost_equal(emb1, emb2)

        set_seed(orig)

    def test_repr_unfitted(self):
        """Test __repr__ for unfitted extractor."""
        extractor = BoVWExtractor(vocab_size=256)
        assert repr(extractor) == "BoVWExtractor(vocab_size=256, fitted=False)"

    def test_repr_fitted(self, rgb_images):
        """Test __repr__ for fitted extractor."""
        extractor = BoVWExtractor(vocab_size=32)
        extractor.fit(rgb_images)
        assert repr(extractor) == "BoVWExtractor(vocab_size=32, fitted=True)"


@pytest.mark.optional
class TestBoVWEdgeCases:
    """Edge case tests for BoVWExtractor."""

    def test_single_image(self):
        """Test with a single image."""
        rng = np.random.default_rng(42)
        single_image = [rng.integers(0, 256, size=(3, 64, 64), dtype=np.uint8)]

        extractor = BoVWExtractor(vocab_size=16)
        embeddings = extractor(single_image)

        assert embeddings.shape[0] == 1

    def test_large_images(self):
        """Test with larger images."""
        rng = np.random.default_rng(42)
        large_images = [rng.integers(0, 256, size=(3, 256, 256), dtype=np.uint8) for _ in range(3)]

        extractor = BoVWExtractor(vocab_size=64)
        embeddings = extractor(large_images)

        assert embeddings.shape[0] == 3

    def test_small_vocab_size(self):
        """Test with very small vocabulary size."""
        rng = np.random.default_rng(42)
        images = [rng.integers(0, 256, size=(3, 64, 64), dtype=np.uint8) for _ in range(5)]

        extractor = BoVWExtractor(vocab_size=2)
        embeddings = extractor(images)

        assert embeddings.shape[0] == 5
        assert embeddings.shape[1] == 2

    def test_refit_replaces_vocabulary(self):
        """Test that calling fit again replaces the vocabulary."""
        rng = np.random.default_rng(42)
        images1 = [rng.integers(0, 256, size=(3, 64, 64), dtype=np.uint8) for _ in range(5)]
        images2 = [rng.integers(0, 256, size=(3, 64, 64), dtype=np.uint8) for _ in range(5)]

        extractor = BoVWExtractor(vocab_size=32)
        extractor.fit(images1)
        kmeans1 = extractor._kmeans

        extractor.fit(images2)
        kmeans2 = extractor._kmeans

        # Should be different kmeans objects
        assert kmeans1 is not kmeans2

    def test_transform_consistent_after_fit(self):
        """Test that transform gives consistent results after fitting."""
        rng = np.random.default_rng(42)
        train_images = [rng.integers(0, 256, size=(3, 64, 64), dtype=np.uint8) for _ in range(5)]
        test_images = [rng.integers(0, 256, size=(3, 64, 64), dtype=np.uint8) for _ in range(3)]

        extractor = BoVWExtractor(vocab_size=32)
        extractor.fit(train_images)

        # Multiple transforms should give same result
        emb1 = extractor.transform(test_images)
        emb2 = extractor.transform(test_images)

        np.testing.assert_array_equal(emb1, emb2)
