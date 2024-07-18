import numpy as np
import pytest
import torch

from dataeval._internal.metrics.ber import _knn_lowerbound
from dataeval._internal.metrics.utils import get_classes_counts
from dataeval.metrics import BER


class TestMulticlassBER:
    @pytest.mark.parametrize(
        "method, k, expected",
        [
            ("MST", 1, {"ber": 0.137, "ber_lower": 0.07132636098401203}),
            ("KNN", 1, {"ber": 0.118, "ber_lower": 0.061072112753426215}),
            ("KNN", 10, {"ber": 0.143, "ber_lower": 0.0745910104681437}),
        ],
    )
    def test_ber_on_mnist(self, method, k, expected, mnist):
        data, labels = mnist()
        metric = BER(data, labels, method, k)
        result = metric.evaluate()
        assert result == expected

    def test_invalid_method(self):
        with pytest.raises(KeyError):
            BER(np.empty([]), np.empty([]), "NOT_A_METHOD")  # type: ignore

    def test_invalid_method_setter(self):
        b = BER(np.empty([]), np.empty([]))
        with pytest.raises(KeyError):
            b.method = "NOT_A_METHOD"  # type: ignore

    def test_class_min(self):
        with pytest.raises(ValueError):
            get_classes_counts(np.ones(20))

    def test_list_class_methods(self):
        methods = BER.methods()
        assert len(methods) == 2

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
        result = _knn_lowerbound(value, classes, k)
        assert result == expected


class TestBERArrayLike:
    @pytest.mark.parametrize(
        "arr, larr",
        [
            (np.random.randint(0, 100, (10, 10)), np.arange(10)),
            (torch.randint(0, 100, (10, 10)), torch.arange(0, 10)),
        ],
    )
    def test_arraylike(self, arr, larr):
        """Test maite.protocols.ArrayLike objects pass evaluation"""

        ber = BER(arr, larr)
        ber.evaluate()

    @pytest.mark.parametrize(
        "arr, larr",
        [
            ([[0, 0, 0], [1, 1], [0, 1], [1, 0]], [0, 0, 1, 1]),
            ([["0", "0"], ["1", "1"], ["0", "1"], ["1", "0"]], ["0", "0", "1", "1"]),
        ],
    )
    def test_invalid_array(self, arr, larr):
        """Test non-arraylike objects fail evaluation"""

        ber = BER(arr, larr)
        with pytest.raises(ValueError):
            ber.evaluate()
