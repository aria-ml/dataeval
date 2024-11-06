import numpy as np

from dataeval.metrics.estimators.uap import uap


class TestUAP:
    def test_uap(self):
        labels = np.arange(10).repeat(100)
        scores = np.zeros((1000, 10), dtype=float)
        result = uap(labels, scores)
        assert result.uap > 0
