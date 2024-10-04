from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from dataeval._internal.metrics.ber import ber, ber_knn, ber_mst, knn_lowerbound
from tests.conftest import mnist


class TestFunctionalBER:
    """Tests the functional methods used in BER"""

    @pytest.mark.parametrize(
        "method, k, expected",
        [
            (ber_mst, None, (0.143, 0.0745910104681437)),
            (ber_knn, 1, (0.12, 0.06214559737386353)),
            (ber_knn, 10, (0.137, 0.07132636098401203)),
        ],
    )
    def test_ber_on_mnist(self, method, k, expected):
        """Methods correctly calculate BER with given params"""
        data, labels = mnist(flatten=True)
        result = method(data, labels, k) if k else method(data, labels)
        assert result == expected

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


class TestAPIBER:
    """Tests the user facing BER Class"""

    def test_invalid_method(self):
        """Raises error when method is not KNN or MST"""
        with pytest.raises(ValueError):
            ber([], [], method="NOT_A_METHOD")  # type: ignore

    @pytest.mark.parametrize(
        "method, k, expected",
        [
            ("MST", 1, {"ber": 0.143, "ber_lower": 0.0745910104681437}),
            ("KNN", 1, {"ber": 0.12, "ber_lower": 0.06214559737386353}),
        ],
    )
    def test_ber_output_format(self, method, k, expected):
        """Confirms BER class transforms functional results into correct format"""

        # TODO: Mock patch _ber methods, just check output tuple -> dict
        images, labels = mnist(flatten=True)
        result = ber(images=images, labels=labels, k=k, method=method)
        assert result.dict() == expected

    def test_torch_inputs(self):
        """Torch class correctly calls functional numpy math"""
        mock_knn = MagicMock()
        mock_knn.return_value = (0, 0)
        from dataeval._internal.metrics.ber import BER_FN_MAP

        BER_FN_MAP["KNN"] = mock_knn

        images = torch.ones((5, 10, 10))
        labels = torch.ones(5)
        ber(images, labels)

        mock_knn.assert_called_once()
