import numpy as np

from daml.metrics.uap import UAP_EMP, UAP_MST


class TestUAP:
    def test_multiclass_UAP_MST_with_mnist(self, mnist):
        """
        Load a slice of the MNIST dataset and pass into the UAP
        evaluate function.
        """

        metric = UAP_MST(*mnist())
        output = metric.evaluate()
        assert output.uap == 1.0

    def test_uap_with_pytorch(self):
        pass

    def test_UAP_EMP(self, mnist):
        _, labels = mnist()
        scores = np.zeros((1000, 10), dtype=float)
        metric = UAP_EMP(labels, scores)
        value = metric.evaluate()
        assert value.uap > 0
