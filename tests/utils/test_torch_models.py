import pytest
import torch

from dataeval.utils.models import AE


@pytest.mark.required
class TestAE:
    def test_encode_output_shape(self):
        ae = AE(input_shape=(1, 32, 32))
        images = torch.ones(size=[1, 1, 32, 32])
        encoded = ae.encoder(images)
        assert encoded.shape == (1, 256)
