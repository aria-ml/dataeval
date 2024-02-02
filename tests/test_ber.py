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
    def test_multiclass_ber_with_mnist(self, ber_metric, output, mnist):
        """
        Load a slice of the MNIST dataset and pass into the BER multiclass
        evaluate function.
        """

        dataset = DamlDataset(*mnist())
        metric = ber_metric(dataset)
        value = metric.evaluate()
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
        metric = ber_metric(dataset)
        with pytest.raises(ValueError):
            value = metric.evaluate()
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
        images = np.array([1, 1])
        labels = np.array([1, 2])
        x1 = DamlDataset(images, labels)
        m = ber_metric(x1)
        m.create_encoding = MagicMock()
        m.evaluate()

        assert m.create_encoding.call_count == 0

    def test_encode_fit_dataset(self, ber_metric):
        """After fitting a dataset, encode is set to True"""

        # Create two sets of images with different labels
        images_1 = np.ones(shape=(1, 32, 32, 1))
        labels_1 = np.ones(shape=(1, 1))
        images_2 = np.ones(shape=(1, 32, 32, 1))
        labels_2 = np.ones(shape=(1, 1)) * 2

        # Combine them into one dataset
        images = np.concatenate([images_1, images_2])
        labels = np.concatenate([labels_1, labels_2])

        x1 = DamlDataset(images, labels)
        m = ber_metric(x1)

        # Encode set to True by fit_dataset
        m.evaluate()
