import numpy as np

from daml.metrics import UAP


class TestUAP:
    def test_UAP_EMP(self, mnist):
        _, labels = mnist()
        scores = np.zeros((1000, 10), dtype=float)
        metric = UAP(labels, scores)
        value = metric.evaluate()
        assert value["uap"] > 0
