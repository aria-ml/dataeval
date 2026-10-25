import numpy as np
import pytest

from dataeval.core._ber import (
    _cover_hart_lower,
    _get_classes_counts,
    _knn_lowerbound,
    _plurality_nearest_tie,
    ber_knn,
    ber_mst,
)
from dataeval.exceptions import ShapeMismatchError


@pytest.mark.required
class TestBERCore:
    """Tests the core BER calculation functions."""

    @pytest.mark.parametrize(
        ("method", "k", "expected_upper", "expected_lower"),
        [
            (ber_mst, None, 0.0, 0.0),
            (ber_knn, 1, 0.0, 0.0),
            (ber_knn, 10, 0.0, 0.0),
        ],
    )
    def test_ber_on_mock_data(self, method, k, expected_upper, expected_lower):
        """Core methods correctly calculate BER with given params."""
        rng = np.random.default_rng(3)
        labels = np.concatenate([rng.choice(10, 500), np.arange(10).repeat(50)])
        data = np.ones((1000, 784)) * labels[:, np.newaxis]
        data[:, 5] += rng.choice(100, 1000)
        data[:, 10] += rng.choice(50, 1000)

        result = method(data, labels, k) if k else method(data, labels)
        assert result["upper_bound"] == pytest.approx(expected_upper)
        assert result["lower_bound"] == pytest.approx(expected_lower)

    @pytest.mark.parametrize(
        ("method", "k", "expected_upper", "expected_lower"),
        [
            (ber_mst, None, 0.01, 0.00501396658942207),
            (ber_knn, 1, 0.011, 0.005516909047465568),
            (ber_knn, 10, 0.005, 0.005),  # lower bound clamped to the achieved k-NN error
        ],
    )
    def test_ber_on_mock_data_with_mismatch(self, method, k, expected_upper, expected_lower):
        """Core methods correctly calculate BER with given params."""
        rng = np.random.default_rng(3)
        labels = np.concatenate([rng.choice(10, 500), np.arange(10).repeat(50)])
        data = np.ones((1000, 784)) * labels[:, np.newaxis]
        data[:, 5] += rng.choice(100, 1000)
        data[:, 10] += rng.choice(50, 1000)
        labels[:5] = [0, 1, 8, 3, 5]

        result = method(data, labels, k) if k else method(data, labels)
        assert result["upper_bound"] == pytest.approx(expected_upper)
        assert result["lower_bound"] == pytest.approx(expected_lower)

    @pytest.mark.parametrize(
        ("value", "classes", "k", "expected"),
        [
            (0.0, 5, 1, 0.0),
            (0.5, 2, 1, 0.5),
            (0.5, 2, 2, 0.5),  # with nearest-tie voting, 2-NN is 1-NN, so Cover-Hart applies
            (0.5, 2, 4, 0.5 / (1 + 1 / np.sqrt(3))),  # even k takes k - 1's constant
            (0.5, 2, 6, 0.5 / (1 + 1 / np.sqrt(5))),  # at k = 5 FeeBee's constant is tighter than Devroye's
            (0.5, 5, 2, 0.31010205144336445),
            (0.5, 5, 6, 0.31010205144336445),
        ],
    )
    def test_knn_lower_bound_2_classes(self, value, classes, k, expected):
        """All logical pathways are correctly calculated."""
        result = _knn_lowerbound(value, classes, k)
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("method", "k"),
        [
            (ber_mst, None),
            (ber_knn, 1),
        ],
    )
    def test_ber_redundant_shape(self, method, k):
        """Unflattened and flattened input should have equivalent outputs."""
        images = np.random.random(size=(10, 3, 3))
        labels = np.arange(10)

        args = (images, labels, k) if k else (images, labels)
        args_flat = (images.reshape(10, -1), labels, k) if k else (images.reshape(10, -1), labels)

        assert method(*args) == method(*args_flat)

    def test_class_min(self):
        """Test minimum class count validation."""
        with pytest.raises(ValueError, match="less than 2 classes"):
            _get_classes_counts(np.ones(1, dtype=np.intp))

    def test_mismatched_embeddings_and_labels_lengths(self):
        """Mismatched embeddings and class_labels lengths raise ShapeMismatchError."""
        embeddings = np.random.random(size=(10, 4))
        labels = np.arange(8)

        with pytest.raises(ShapeMismatchError):
            ber_mst(embeddings, labels)

    def test_list_labels_are_accepted(self):
        """Python lists work, as the docstring promises."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=(50, 3))
        y = (x[:, 0] > 0).astype(int)
        assert ber_knn(x.tolist(), y.tolist(), 1) == ber_knn(x, y, 1)

    @pytest.mark.parametrize("k", [1, 2, 3, 4, 6])
    def test_lower_bound_never_exceeds_bayes_error(self, k):
        """A constant posterior of 0.9 has Bayes error 0.1; no k may report a lower bound above it."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=(20000, 4))
        y = (rng.random(20000) < 0.9).astype(int)
        assert ber_knn(x, y, k)["lower_bound"] <= 0.1 + 0.005

    def test_plurality_breaks_ties_toward_the_nearest(self):
        """Mode would pick the smallest tied label; the nearest tied neighbor wins instead."""
        neighbors = np.array([[1, 0, 0, 1], [2, 1, 1, 2]])
        assert _plurality_nearest_tie(neighbors).tolist() == [1, 2]

    def test_cover_hart_lower_is_elementwise(self):
        errors = np.array([0.0, 0.1, 0.5])
        assert _cover_hart_lower(errors, 2) == pytest.approx([float(_cover_hart_lower(e, 2)) for e in errors])
        assert float(_cover_hart_lower(0.5, 2)) == pytest.approx(0.5)

    def test_lower_never_exceeds_upper(self):
        rng = np.random.default_rng(3)
        labels = np.concatenate([rng.choice(10, 500), np.arange(10).repeat(50)])
        data = np.ones((1000, 784)) * labels[:, np.newaxis]
        data[:, 5] += rng.choice(100, 1000)
        data[:, 10] += rng.choice(50, 1000)
        labels[:5] = [0, 1, 8, 3, 5]
        for k in (1, 2, 5, 10):
            result = ber_knn(data, labels, k)
            assert result["lower_bound"] <= result["upper_bound"]

    def test_groups_remove_the_duplicate_leak(self):
        """Ungrouped, every copy finds its twin; grouped, the error matches the de-duplicated data."""
        rng = np.random.default_rng(0)
        base = rng.normal(size=(300, 8))
        labels = rng.integers(0, 2, 300)
        data, dup_labels, groups = np.repeat(base, 3, axis=0), labels.repeat(3), np.arange(300).repeat(3)
        assert ber_knn(data, dup_labels, 1)["upper_bound"] == 0.0
        grouped = ber_knn(data, dup_labels, 1, groups=groups)["upper_bound"]
        assert grouped == pytest.approx(ber_knn(base, labels, 1)["upper_bound"], abs=0.02)

    def test_ber_knn_groups_needs_k_outside(self):
        with pytest.raises(ValueError, match="outside its group"):
            ber_knn(np.array([[0.0], [1.0], [5.0]]), np.array([0, 1, 1]), 2, groups=np.array([0, 0, 1]))
