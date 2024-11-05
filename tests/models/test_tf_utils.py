from unittest.mock import patch

import pytest

from dataeval.utils.tensorflow._internal.autoencoder import AE, AEGMM, VAE, VAEGMM
from dataeval.utils.tensorflow._internal.pixelcnn import PixelCNN
from dataeval.utils.tensorflow._internal.utils import create_model


class TestTensorflowModels:
    input_shape = (32, 32, 3)

    @pytest.mark.parametrize("model_type", [AE, AEGMM, PixelCNN, VAE, VAEGMM])
    def test_create_model(self, model_type):
        with patch(f"dataeval.utils.tensorflow._internal.utils.{model_type.__qualname__}") as mock_model:
            create_model(model_type.__name__, self.input_shape)
            assert mock_model.called

    def test_create_model_invalid_class(self):
        with pytest.raises(TypeError):
            create_model("not_a_valid_class", self.input_shape)  # type: ignore
