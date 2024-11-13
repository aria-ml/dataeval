from unittest.mock import patch

import pytest

from dataeval.utils.tensorflow._internal.models import AE, AEGMM, VAE, VAEGMM, PixelCNN
from dataeval.utils.tensorflow._internal.utils import create_model


class TestTensorflowModels:
    input_shape = (32, 32, 3)

    @pytest.mark.parametrize("model_type", [AE, AEGMM, PixelCNN, VAE, VAEGMM])
    def test_create_model(self, model_type):
        with patch("dataeval.utils.tensorflow._internal.utils.tf_models") as mock_models:
            create_model(model_type.__name__, self.input_shape)
            assert getattr(mock_models, model_type.__name__).called

    def test_create_model_invalid_class(self):
        with pytest.raises(TypeError):
            create_model("not_a_valid_class", self.input_shape)  # type: ignore
