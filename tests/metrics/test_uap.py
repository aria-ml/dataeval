import numpy as np

from dataeval.metrics import uap


class TestUAP:
    def test_uap(self, mnist):
        _, labels = mnist()
        scores = np.zeros((1000, 10), dtype=float)
        result = uap(labels, scores)
        assert result["uap"] > 0
