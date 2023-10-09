import numpy as np
import pytest

from daml.metrics.ber import BER, BEROutput


class TestMulticlassBER:
    def test_multiclass_ber_with_mnist(self):
        """
        Load a slice of the MNIST dataset and pass into the BER multiclass
        evaluate function.
        """

        expected_result = BEROutput(ber=0.137, ber_lower=0.07132636098401203)
        path = "tests/datasets/mnist.npz"
        with np.load(path, allow_pickle=True) as fp:
            covariates, labels = fp["x_train"][:1000], fp["y_train"][:1000]

        metric = BER()
        actual_result = metric.evaluate(
            X=covariates,
            y=labels,
        )
        assert actual_result == expected_result

    def test_class_max(self):
        value = None
        covariates = np.ones(20)
        labels = np.array(range(20))
        metric = BER()
        with pytest.raises(ValueError):
            value = metric.evaluate(
                X=covariates,
                y=labels,
            )
            assert value is not None

    def test_class_min(self):
        value = None
        covariates = np.ones(20)
        labels = np.ones(20)
        metric = BER()
        with pytest.raises(ValueError):
            value = metric.evaluate(
                X=covariates,
                y=labels,
            )
            assert value is not None
