from unittest.mock import MagicMock, patch

import torch

from dataeval.metrics.torch import BER as BER


class TestTorchBER:
    @patch("dataeval._internal.metrics.ber.ber_knn")
    def test_torch_inputs(self, mock_knn: MagicMock):
        """Torch class correctly calls functional numpy math"""
        mock_knn.return_value = (0, 0)
        images = torch.ones((5, 10, 10))
        labels = torch.ones(5)

        BER().evaluate(images, labels)

        mock_knn.assert_called_once()
