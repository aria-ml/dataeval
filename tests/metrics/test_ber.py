import numpy as np
import pytest

from daml.metrics.ber import BER_FNN, BER_MST, BEROutput


class TestMulticlassBER:
    @pytest.mark.parametrize(
        "ber_metric, expected",
        [
            (BER_MST, BEROutput(ber=0.137, ber_lower=0.07132636098401203)),
            (BER_FNN, BEROutput(ber=0.118, ber_lower=0.061072112753426215)),
        ],
    )
    def test_ber_on_mnist(self, ber_metric, expected, mnist):
        metric = ber_metric(*mnist())
        result = metric.evaluate()
        assert result == expected

    @pytest.mark.parametrize("ber_metric", [BER_MST, BER_FNN])
    def test_class_min(self, ber_metric):
        value = None
        covariates = np.ones(20)
        labels = np.ones(20)
        metric = ber_metric(covariates, labels)
        with pytest.raises(ValueError):
            value = metric.evaluate()
            assert value is not None
