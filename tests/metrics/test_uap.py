import numpy as np

from daml.metrics.uap import UAP_EMP


class TestUAP:
    def test_UAP_EMP(self, mnist):
        _, labels = mnist()
        scores = np.zeros((1000, 10), dtype=float)
        metric = UAP_EMP(labels, scores)
        value = metric.evaluate()
        assert value.uap > 0
