import numpy as np
import pytest

from daml.metrics.ber import BEROutput
from daml.metrics.ber.aria import BER_FNN, BER_MST


class TestMulticlassBER:
    @pytest.mark.parametrize(
        "input, output",
        [
            (BER_MST, BEROutput(ber=0.137, ber_lower=0.07132636098401203)),
            (BER_FNN, BEROutput(ber=0.118, ber_lower=0.061072112753426215)),
        ],
    )
    def test_multiclass_ber_with_mnist(self, input, output):
        """
        Load a slice of the MNIST dataset and pass into the BER multiclass
        evaluate function.
        """

        path = "tests/datasets/mnist.npz"
        with np.load(path, allow_pickle=True) as fp:
            covariates, labels = fp["x_train"][:1000], fp["y_train"][:1000]

        metric = input()
        value = metric.evaluate(
            X=covariates,
            y=labels,
        )
        assert value == output

    @pytest.mark.parametrize(
        "input",
        [
            BER_MST,
            BER_FNN,
        ],
    )
    def test_class_max(self, input):
        value = None
        covariates = np.ones(20)
        labels = np.array(range(20))
        metric = input()
        with pytest.raises(ValueError):
            value = metric.evaluate(
                X=covariates,
                y=labels,
            )
            assert value is not None

    @pytest.mark.parametrize(
        "input",
        [
            BER_MST,
            BER_FNN,
        ],
    )
    def test_class_min(self, input):
        value = None
        covariates = np.ones(20)
        labels = np.ones(20)
        metric = input()
        with pytest.raises(ValueError):
            value = metric.evaluate(
                X=covariates,
                y=labels,
            )
            assert value is not None
