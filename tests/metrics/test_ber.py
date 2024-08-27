from unittest.mock import MagicMock

import pytest
import torch

from dataeval._internal.metrics.ber import ber_knn, ber_mst, knn_lowerbound
from dataeval.metrics import ber


class TestFunctionalBER:
    """Tests the functional methods used in BER"""

    @pytest.mark.parametrize(
        "method, k, expected",
        [
            (ber_mst, None, (0.137, 0.07132636098401203)),
            (ber_knn, 1, (0.118, 0.061072112753426215)),
            (ber_knn, 10, (0.143, 0.0745910104681437)),
        ],
    )
    def test_ber_on_mnist(self, method, k, expected, mnist):
        """Methods correctly calculate BER with given params"""
        data, labels = mnist()
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
        result = knn_lowerbound(value, classes, k)
        assert result == expected


class TestAPIBER:
    """Tests the user facing BER Class"""

    def test_invalid_method(self):
        """Raises error when method is not KNN or MST"""
        with pytest.raises(ValueError):
            ber([], [], method="NOT_A_METHOD")  # type: ignore

    @pytest.mark.parametrize(
        "method, k, expected",
        [
            ("MST", 1, {"ber": 0.137, "ber_lower": 0.07132636098401203}),
            ("KNN", 1, {"ber": 0.118, "ber_lower": 0.061072112753426215}),
        ],
    )
    def test_ber_output_format(self, method, k, expected, mnist):
        """Confirms BER class transforms functional results into correct format"""

        # TODO: Mock patch _ber methods, just check output tuple -> dict
        images, labels = mnist()
        result = ber(images=images, labels=labels, k=k, method=method)
        assert result == expected

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
