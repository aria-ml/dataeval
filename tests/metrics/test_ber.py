import numpy as np
import pytest

from dataeval._internal.functional.ber import _knn_lowerbound, ber_knn, ber_mst
from dataeval.metrics import BER


class TestFunctionalBER:
    """Tests the functional methods used in BER"""

    @pytest.mark.parametrize(
        "method, k, expected",
        [
            (ber_mst, 1, (0.137, 0.07132636098401203)),
            (ber_knn, 1, (0.118, 0.061072112753426215)),
            (ber_knn, 10, (0.143, 0.0745910104681437)),
        ],
    )
    def test_ber_on_mnist(self, method, k, expected, mnist):
        """Methods correctly calculate BER with given params"""
        data, labels = mnist()
        result = method(data, labels, k)
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
        result = _knn_lowerbound(value, classes, k)
        assert result == expected


class TestAPIBER:
    """Tests the user facing BER Class"""

    def test_invalid_method(self):
        """Raises error when method is not KNN or MST"""
        with pytest.raises(KeyError):
            BER(np.empty([]), np.empty([]), "NOT_A_METHOD")  # type: ignore

    def test_invalid_method_setter(self):
        """Raises error when method key is not KNN or MST"""
        b = BER(np.empty([]), np.empty([]))
        with pytest.raises(KeyError):
            b.method = "NOT_A_METHOD"  # type: ignore

    def test_list_class_methods(self):
        methods = BER.methods()
        assert len(methods) == 2

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
        data, labels = mnist()
        ber = BER(data=data, labels=labels, method=method, k=k)
        result = ber.evaluate()
        assert result == expected
