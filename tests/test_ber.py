from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from daml.datasets import DamlDataset
from daml.metrics.ber import BER_FNN, BER_MST, BEROutput


class TestMulticlassBER:
    @pytest.mark.parametrize(
        "ber_metric, output",
        [
            (BER_MST, BEROutput(ber=0.137, ber_lower=0.07132636098401203)),
            (BER_FNN, BEROutput(ber=0.118, ber_lower=0.061072112753426215)),
        ],
    )
    def test_multiclass_ber_with_mnist(self, ber_metric, output):
        """
        Load a slice of the MNIST dataset and pass into the BER multiclass
        evaluate function.
        """

        path = "tests/datasets/mnist.npz"
        with np.load(path, allow_pickle=True) as fp:
            covariates, labels = fp["x_train"][:1000], fp["y_train"][:1000]
        dataset = DamlDataset(covariates, labels)
        metric = ber_metric()
        value = metric.evaluate(dataset)
        assert value == output

    @pytest.mark.parametrize(
        "ber_metric",
        [
            BER_MST,
            BER_FNN,
        ],
    )
    def test_class_min(self, ber_metric):
        value = None
        covariates = np.ones(20)
        labels = np.ones(20)
        dataset = DamlDataset(covariates, labels)
        metric = ber_metric()
        with pytest.raises(ValueError):
            value = metric.evaluate(dataset)
            assert value is not None


@pytest.mark.parametrize(
    "ber_metric",
    [
        BER_FNN,
        BER_MST,
    ],
)
class TestBEREncode:
    def test_encode_default(self, ber_metric):
        """Default encode is False"""
        m = ber_metric()
        m.create_encoding = MagicMock()
        images = np.array([1, 1])
        labels = np.array([1, 2])
        x1 = DamlDataset(images, labels)
        m.evaluate(x1)

        assert m.create_encoding.call_count == 0

    def test_encode_fit_dataset(self, ber_metric):
        """After fitting a dataset, encode is set to True"""
        m = ber_metric()

        # Create two sets of images with different labels
        images_1 = np.ones(shape=(1, 32, 32, 1))
        labels_1 = np.ones(shape=(1, 1))
        images_2 = np.ones(shape=(1, 32, 32, 1))
        labels_2 = np.ones(shape=(1, 1)) * 2

        # Combine them into one dataset
        images = np.concatenate([images_1, images_2])
        labels = np.concatenate([labels_1, labels_2])

        x1 = DamlDataset(images, labels)
        m.fit_dataset(x1, epochs=1)

        # Encode set to True by fit_dataset
        m.evaluate(x1)

    def test_encode_override_true(self, ber_metric):
        """Override default encode with True raises TypeError (no model fit)"""
        m = ber_metric()
        images = np.array([1, 1])
        labels = np.array([1, 2])
        x1 = DamlDataset(images, labels)

        with pytest.raises(TypeError):
            m.evaluate(x1, True)

    @mock.patch("daml._internal.metrics.aria.ber.permute_to_torch")
    def test_encode_override_false(self, mock_permute, ber_metric):
        """Override encode with False after fitting dataset"""
        m = ber_metric()
        m.encode = True

        images = np.array([1, 1])
        labels = np.array([1, 2])
        x1 = DamlDataset(images, labels)

        assert m.encode
        m.evaluate(x1, False)

        assert mock_permute.call_count == 0
