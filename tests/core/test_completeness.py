import numpy as np
import pytest

from dataeval.core._completeness import completeness


@pytest.mark.required
class TestCompletenessUnit:
    def test_uniform_data_completeness(self):
        """Test that uniform random data has high completeness."""
        embs = np.random.random(size=(1000, 10))
        result = completeness(embs)
        # Uniform data should have completeness close to 1
        assert result["completeness"] > 0.95

    def test_perfect_correlation_completeness(self):
        """Test that perfectly correlated dimensions reduce effective dimensionality."""
        # Create data with 5 dimensions but copy first dimension to others
        n, d = 1000, 5
        embs = np.random.random(size=(n, d))
        for i in range(1, d):
            embs[:, i] = embs[:, 0]

        result = completeness(embs)
        # With perfect correlation, completeness should be low (effective dim ~1, so ~1/5)
        assert result["completeness"] < 0.3

    def test_partial_correlation_completeness(self):
        """Test that partially correlated data has intermediate completeness."""
        # Create data where first 5 dimensions are independent, last 5 are copies
        n, d = 1000, 10
        embs = np.random.random(size=(n, d))
        for i in range(5, d):
            embs[:, i] = embs[:, i - 5]

        result = completeness(embs)

        assert 0.45 < result["completeness"] < 0.55

    def test_swiss_roll_completeness(self):
        """Test completeness on swiss roll manifold."""
        # Generate swiss roll data
        n = 1000
        t = np.random.rand(n) * 4 * np.pi
        h = np.random.rand(n) * 5
        embs = np.zeros((n, 3))
        embs[:, 0] = t * np.cos(t)
        embs[:, 1] = h
        embs[:, 2] = t * np.sin(t)

        result = completeness(embs)
        # Swiss roll has intrinsic dim of 2 but uses all 3 embedding dimensions
        assert result["completeness"] * 3 > 2.5

    def test_hyperspherical_completeness(self):
        """Test completeness on hyperspherical manifold."""
        # Generate points on a unit 5-sphere in 10D space
        n, manifold_dim, embed_dim = 1000, 5, 10
        X_base = np.random.normal(0, 1, (n, manifold_dim + 1))
        norms = np.linalg.norm(X_base, axis=1, keepdims=True)
        X_base = X_base / norms

        X = np.zeros((n, embed_dim))
        X[:, : manifold_dim + 1] = X_base

        result = completeness(X)
        # Should capture close to manifold_dim+1 dimensions
        assert manifold_dim < (result["completeness"] * embed_dim) < (manifold_dim + 1.5)

    def test_rank_normalization_invariance(self):
        """Test that scaling input dimensions doesn't affect the result."""
        n, d = 1000, 5
        # Create two datasets with same structure but different scales
        embs1 = np.random.random(size=(n, d))
        # Scale each dimension differently
        scales = np.exp(np.arange(d))
        embs2 = embs1 * scales[np.newaxis, :]

        result1 = completeness(embs1)
        result2 = completeness(embs2)
        # Results should be very close
        np.testing.assert_almost_equal(result1["completeness"], result2["completeness"], decimal=4)
