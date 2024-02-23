import numpy as np
import pytest

from daml._internal.metrics.utils import get_classes_counts
from daml.metrics.ber import BER, BEROutput


class TestMulticlassBER:
    @pytest.mark.parametrize(
        "method, expected",
        [
            ("MST", BEROutput(ber=0.137, ber_lower=0.07132636098401203)),
            ("KNN", BEROutput(ber=0.118, ber_lower=0.061072112753426215)),
        ],
    )
    def test_ber_on_mnist(self, method, expected, mnist):
        data, labels = mnist()
        metric = BER(data, labels, method)
        result = metric.evaluate()
        assert result == expected

    def test_invalid_method(self):
        with pytest.raises(KeyError):
            BER(np.empty([]), np.empty([]), "NOT_A_METHOD")  # type: ignore

    def test_class_min(self):
        with pytest.raises(ValueError):
            get_classes_counts(np.ones(20))

    def test_list_class_methods(self):
        methods = BER.methods()
        assert len(methods) == 2
