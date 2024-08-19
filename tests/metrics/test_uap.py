import numpy as np

from dataeval._internal.functional.uap import uap
from dataeval.metrics import UAP


class TestAPIUAP:
    def test_UAP_EMP(self, mnist):
        _, labels = mnist()
        scores = np.zeros((1000, 10), dtype=float)
        metric = UAP()
        value = metric.evaluate(labels, scores)
        assert value["uap"] > 0


class TestFunctionalUAP:
    def test_uap(self, mnist):
        _, labels = mnist()
        scores = np.zeros((1000, 10), dtype=float)
        value = uap(labels, scores)
        assert value > 0
