from unittest.mock import patch

import pytest

from daml._internal.models.tensorflow.autoencoder import AE, AEGMM, VAE, VAEGMM
from daml._internal.models.tensorflow.pixelcnn import PixelCNN
from daml._internal.models.tensorflow.utils import create_model


class TestTensorflowModels:
    input_shape = (32, 32, 3)

    @pytest.mark.parametrize("model_type", [AE, AEGMM, PixelCNN, VAE, VAEGMM])
    def test_create_model(self, model_type):
        with patch(f"daml._internal.models.tensorflow.utils.{model_type.__qualname__}") as mock_model:
            create_model(mock_model, self.input_shape)
            assert mock_model.called

    def test_create_model_invalid_class(self):
        with pytest.raises(TypeError):
            create_model("not_a_valid_class", self.input_shape)  # type: ignore
