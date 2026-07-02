import numpy as np
import pytest

from dataeval.core._completeness import completeness


@pytest.mark.required
class TestNearestNeighborDistancesUnit:
    def test_distances_align_with_pairs(self):
        """Each returned distance is the euclidean distance between its pair, in pair order."""
        rng = np.random.default_rng(0)
        embs = rng.random((50, 4))
        result = completeness(embs)
        pairs = result["nearest_neighbor_pairs"]
        distances = result["nearest_neighbor_distances"]

        assert len(distances) == len(pairs)
        for (i, j), d in zip(pairs, distances, strict=True):
            np.testing.assert_allclose(d, float(np.linalg.norm(embs[i] - embs[j])))

    def test_distances_sorted_decreasing_like_pairs(self):
        """Distances share the pairs' decreasing-distance ordering."""
        rng = np.random.default_rng(1)
        embs = rng.random((40, 3))
        distances = completeness(embs)["nearest_neighbor_distances"]
        assert list(distances) == sorted(distances, reverse=True)

    def test_distances_empty_for_single_point(self):
        """No neighbors, no distances."""
        result = completeness(np.random.default_rng(2).random((1, 4)))
        assert list(result["nearest_neighbor_distances"]) == []


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


@pytest.mark.required
class TestIsotropyUnit:
    def test_isotropic_gaussian_isotropy(self):
        """Test that an isotropic Gaussian has isotropy close to 1."""
        n, d = 1000, 10
        embs = np.random.randn(n, d)
        result = completeness(embs)
        assert result["isotropy"] > 0.95

    def test_near_perfect_correlation_isotropy(self):
        """Test that nearly perfectly correlated data has low isotropy."""
        n, d = 1000, 5
        embs = np.random.random(size=(n, d))
        for i in range(1, d):
            embs[:, i] = embs[:, 0]

        _, _, Vt = np.linalg.svd(embs - np.mean(embs, axis=0), full_matrices=True)
        null_direction = Vt[1]
        noise_amplitude = 1e-4
        noise = np.random.randn(n, 1) * noise_amplitude
        embs += noise @ null_direction[np.newaxis, :]

        result = completeness(embs)
        assert result["isotropy"] < 0.5 + noise_amplitude

    def test_anisotropic_gaussian_isotropy(self):
        n, ambient, intrinsic = 1000, 10, 4
        Q, _ = np.linalg.qr(np.random.randn(ambient, intrinsic))
        scales = np.exp(np.linspace(0, 3, intrinsic))
        embs = (np.random.randn(n, intrinsic) * scales) @ Q.T
        result = completeness(embs)
        assert result["isotropy"] < 0.5

    def test_isotropy_invariant_to_rotation(self):
        """Test that rotating the data does not affect isotropy."""
        n, d = 1000, 10
        embs = np.random.randn(n, d)
        # Random rotation via QR
        Q, _ = np.linalg.qr(np.random.randn(d, d))
        embs_rotated = embs @ Q
        result1 = completeness(embs)
        result2 = completeness(embs_rotated)
        np.testing.assert_almost_equal(result1["isotropy"], result2["isotropy"], decimal=2)

    def test_isotropy_invariant_to_scaling(self):
        """Test that uniformly scaling all dimensions does not affect isotropy."""
        n, d = 1000, 10
        embs = np.random.randn(n, d)
        result1 = completeness(embs)
        result2 = completeness(embs * 100)
        np.testing.assert_almost_equal(result1["isotropy"], result2["isotropy"], decimal=2)

    def test_low_rank_isotropy(self):
        """Test that a low-rank embedding has isotropy reflecting variance distribution within that rank."""
        n, ambient = 1000, 20
        intrinsic = 5
        # Embed isotropic Gaussian in higher-dimensional space
        X_low = np.random.randn(n, intrinsic)
        Q, _ = np.linalg.qr(np.random.randn(ambient, intrinsic))
        embs = X_low @ Q.T
        result = completeness(embs)
        # Isotropic within its subspace, so isotropy should be high
        assert result["isotropy"] > 0.95

    def test_anisotropic_low_rank_isotropy(self):
        """Test that an anisotropic low-rank embedding has lower isotropy than an isotropic one."""
        n, ambient, intrinsic = 1000, 20, 5
        Q, _ = np.linalg.qr(np.random.randn(ambient, intrinsic))

        # Isotropic
        X_iso = np.random.randn(n, intrinsic) @ Q.T
        result_iso = completeness(X_iso)

        # Anisotropic — vary scales within the intrinsic subspace
        scales = np.exp(np.linspace(0, 4, intrinsic))
        X_aniso = (np.random.randn(n, intrinsic) * scales) @ Q.T
        result_aniso = completeness(X_aniso)

        assert result_aniso["isotropy"] < result_iso["isotropy"]

    def test_span_and_isotropy_independent(self):
        """Test that span and isotropy can vary independently."""
        n = 1000

        # High span, low isotropy: many dimensions but very uneven variance
        scales = np.exp(np.linspace(0, 4, 10))
        embs_high_span = np.random.randn(n, 10) * scales
        # Now introduce correlations by mixing dimensions
        Q, _ = np.linalg.qr(np.random.randn(10, 10))
        embs_high_span = embs_high_span @ Q.T
        result_high_span = completeness(embs_high_span)

        # Low span, high isotropy: few dimensions but evenly used
        X_low = np.random.randn(n, 3)
        Q, _ = np.linalg.qr(np.random.randn(10, 3))
        embs_low_span = X_low @ Q.T
        result_low_span = completeness(embs_low_span)

        assert result_high_span["completeness"] > result_low_span["completeness"]
        assert result_high_span["isotropy"] < result_low_span["isotropy"]
