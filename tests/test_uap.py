import numpy as np

from daml.datasets import DamlDataset
from daml.metrics.uap import UAP_EMP, UAP_MST


class TestUAP:
    def test_multiclass_UAP_MST_with_mnist(self, mnist):
        """
        Load a slice of the MNIST dataset and pass into the UAP
        evaluate function.
        """

        metric = UAP_MST(DamlDataset(*mnist()))
        output = metric.evaluate()
        assert output.uap == 1.0

    def test_uap_with_pytorch(self):
        pass

    def test_UAP_EMP(self, mnist):
        scores = np.zeros((1000, 10), dtype=float)
        metric = UAP_EMP(DamlDataset(*mnist()), scores)
        value = metric.evaluate()
        assert value.uap > 0
