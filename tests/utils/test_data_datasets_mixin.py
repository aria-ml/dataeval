from unittest.mock import patch

import torch

from dataeval.utils.data.datasets._mixin import BaseDatasetTorchMixin


class TestBaseTorchMixin:
    class MockDataset(BaseDatasetTorchMixin):
        index2label = {0: "zero", 1: "one"}

    def test_as_array(self):
        mock = self.MockDataset()
        assert torch.equal(mock._as_array([1, 2, 3]), torch.tensor([1, 2, 3]))

    def test_one_hot_encode_int(self):
        mock = self.MockDataset()
        assert torch.equal(mock._one_hot_encode(1), torch.tensor([0.0, 1.0]))

    def test_one_hot_encode_list(self):
        mock = self.MockDataset()
        assert torch.equal(mock._one_hot_encode([0, 1]), torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    @patch("dataeval.utils.data.datasets._mixin.Image")
    def test_read_file(self, mock_image):
        mock = self.MockDataset()
        mock_image.open.return_value = [[[0], [0]], [[0], [0]]]  # (2, 2, 1)
        assert torch.equal(mock._read_file("any"), torch.zeros(1, 2, 2))
