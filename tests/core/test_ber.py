import numpy as np
import pytest

from dataeval.core._ber import _get_classes_counts, _knn_lowerbound, ber_knn, ber_mst


@pytest.mark.required
class TestBERCore:
    """Tests the core BER calculation functions"""

    @pytest.mark.parametrize(
        "method, k, expected_upper, expected_lower",
        [
            (ber_mst, None, 0.004, 0.0020022271742540345),
            (ber_knn, 1, 0.0, 0.0),
            (ber_knn, 10, 0.0, 0.0),
        ],
    )
    def test_ber_on_mock_data(self, method, k, expected_upper, expected_lower):
        """Core methods correctly calculate BER with given params"""
        rng = np.random.default_rng(3)
        labels = np.concatenate([rng.choice(10, 500), np.arange(10).repeat(50)])
        data = np.ones((1000, 784)) * labels[:, np.newaxis]
        data[:, 13:16] += 1
        data[-200:, 13:16] += rng.choice(5)

        result = method(data, labels, k) if k else method(data, labels)
        assert result["upper_bound"] == expected_upper
        assert result["lower_bound"] == expected_lower

    @pytest.mark.parametrize(
        "value, classes, k, expected",
        [
            (0.0, 5, 1, 0.0),
            (0.5, 2, 1, 0.5),
            (0.5, 2, 2, 0.25),
            (0.5, 2, 4, 0.3333333333333333),
            (0.5, 2, 6, 0.3394049878693466),
            (0.5, 5, 2, 0.31010205144336445),
            (0.5, 5, 6, 0.31010205144336445),
        ],
    )
    def test_knn_lower_bound_2_classes(self, value, classes, k, expected):
        """All logical pathways are correctly calculated"""
        result = _knn_lowerbound(value, classes, k)
        assert result == expected

    @pytest.mark.parametrize(
        "method, k",
        [
            (ber_mst, None),
            (ber_knn, 1),
        ],
    )
    def test_ber_redundant_shape(self, method, k):
        """Unflattened and flattened input should have equivalent outputs"""
        images = np.random.random(size=(10, 3, 3))
        labels = np.arange(10)

        args = (images, labels, k) if k else (images, labels)
        args_flat = (images.reshape(10, -1), labels, k) if k else (images.reshape(10, -1), labels)

        assert method(*args) == method(*args_flat)

    def test_class_min(self):
        """Test minimum class count validation"""
        with pytest.raises(ValueError):
            _get_classes_counts(np.ones(1, dtype=np.intp))
