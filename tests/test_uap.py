from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from daml.datasets import DamlDataset
from daml.metrics.uap import UAP_EMP, UAP_MST


class TestUAP:
    def test_multiclass_UAP_MST_with_mnist(self, mnist):
        """
        Load a slice of the MNIST dataset and pass into the UAP
        evaluate function.
        """

        metric = UAP_MST()
        output = metric.evaluate(DamlDataset(*mnist))
        assert output.uap == 1.0

    def test_uap_with_pytorch(self):
        pass

    def test_UAP_EMP(self, mnist):
        scores = np.zeros((1000, 10), dtype=float)
        metric = UAP_EMP()
        value = metric.evaluate(DamlDataset(*mnist), scores)
        assert value.uap > 0

    def test_UAP_MST_encode_without_model(self):
        uap = UAP_MST(encode=True)
        uap.model = None
        with pytest.raises(TypeError):
            uap.evaluate(DamlDataset(np.ndarray([])))

    def test_UAP_MST_encode_with_untrained_model(self):
        uap = UAP_MST(encode=True)
        uap.model = MagicMock()
        with pytest.raises(TypeError):
            uap.evaluate(DamlDataset(np.ndarray([])))

    @mock.patch("daml._internal.metrics.aria.uap.permute_to_torch")
    @mock.patch("daml._internal.metrics.aria.uap.UAP_MST._uap")
    def test_UAP_MST_encode_with_model(self, mock_permute, mock_uap):
        mock_model = MagicMock()
        mock_model.encode.return_value.numpy.return_value = np.ndarray([])

        uap = UAP_MST(encode=True)
        uap.model = mock_model
        uap._is_trained = True

        uap.evaluate(DamlDataset(np.ndarray([])))

        assert mock_permute.call_count == 1
        assert mock_uap.call_count == 1
