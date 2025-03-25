import numpy as np
import pytest

from dataeval.metrics.estimators._ber import ber, ber_knn, ber_mst, get_classes_counts, knn_lowerbound
from dataeval.outputs._estimators import BEROutput


@pytest.mark.required
class TestFunctionalBER:
    """Tests the functional methods used in BER"""

    @pytest.mark.parametrize(
        "method, k, expected",
        [
            ("MST", None, (0.009, 0.004511306604042031)),
            ("KNN", 1, (0.0, 0.0)),
            ("KNN", 10, (0.0, 0.0)),
        ],
    )
    def test_ber_on_mock_data(self, method, k, expected):
        """Methods correctly calculate BER with given params"""
        rng = np.random.default_rng(3)
        labels = np.concatenate([rng.choice(10, 500), np.arange(10).repeat(50)])
        data = np.ones((1000, 784)) * labels[:, np.newaxis]
        data[:, 13:16] += 1
        data[-200:, 13:16] += rng.choice(5)
        result = ber(data, labels, k, method=method) if k else ber(data, labels, method=method)
        assert (result.ber, result.ber_lower) == expected

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
        result = knn_lowerbound(value, classes, k)
        assert result == expected

    @pytest.mark.parametrize(
        "method, k",
        [
            (ber_mst, None),
            (ber_knn, 1),
        ],
    )
    def test_ber_mst_redundant_shape(self, method, k):
        """Unflattened and flattened input should have equivalent outputs"""
        images = np.random.random(size=(10, 3, 3))
        labels = np.arange(10)

        args = (images, labels, k) if k else (images, labels)
        args_flat = (images, labels, k) if k else (images.reshape(10, -1), labels)

        assert method(*args) == method(*args_flat)


@pytest.mark.required
class TestAPIBER:
    """Tests the user facing BER Class"""

    def test_invalid_method(self):
        """Raises error when method is not KNN or MST"""
        with pytest.raises(ValueError):
            ber([], [], method="NOT_A_METHOD")  # type: ignore

    def test_ber_output_format(self):
        result = BEROutput(0.8, 0.2)
        assert result.dict() == {"ber": 0.8, "ber_lower": 0.2}

    def test_ber_high_dim_data_valueerror(self):
        """High dimensional data should raise valueerror"""
        embs = np.random.random(size=(100, 16, 16))
        with pytest.raises(ValueError):
            ber(embs, embs)

    def test_class_min(self):
        with pytest.raises(ValueError):
            get_classes_counts(np.ones(1, dtype=np.int_))
