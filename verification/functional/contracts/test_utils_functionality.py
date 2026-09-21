"""Verify that public evaluation support utilities are available and functional.

Maps to meta repo test cases:
  - TC-12.1: Utility components (Thresholds, Preprocessing, Data, ONNX)
"""

import numpy as np
import pytest


@pytest.mark.test_case("12-1")
class TestUtilsFunctionality:
    """Verify evaluation support and data preparation utilities."""

    def test_utils_thresholds_zscore(self):
        from dataeval.utils.thresholds import ZScoreThreshold

        threshold = ZScoreThreshold(3.0)
        assert threshold.upper_multiplier == 3.0
        assert callable(threshold)

    def test_utils_thresholds_iqr(self):
        from dataeval.utils.thresholds import IQRThreshold

        threshold = IQRThreshold(1.5)
        assert threshold.lower_multiplier == 1.5
        assert threshold.upper_multiplier == 1.5
        assert callable(threshold)

    def test_utils_preprocessing_box_conversion(self):
        from dataeval.utils.preprocessing import to_int_box

        # Outward rounding: floor the top-left, ceil the bottom-right, so the
        # integer box always covers the float box it came from.
        assert to_int_box((1.2, 2.7, 3.4, 4.9)) == (1, 2, 4, 5)

    def test_utils_preprocessing_canonical_grayscale(self):
        from dataeval.utils.preprocessing import to_canonical_grayscale

        image = np.zeros((3, 4, 4), dtype=np.uint8)
        gray = to_canonical_grayscale(image)
        assert gray.dtype == np.uint8
        assert gray.shape[-2:] == (4, 4)

    def test_utils_preprocessing_rescale(self):
        from dataeval.utils.preprocessing import rescale

        img = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        rescaled = rescale(img, depth=8).astype(np.uint8)
        assert rescaled[0] == 0
        assert rescaled[-1] == 255

    def test_utils_data_validation_helpers(self):
        from dataeval.utils.data import DatasetKind, validate_dataset

        images = [np.zeros((3, 4, 4), dtype=np.uint8) for _ in range(3)]
        assert validate_dataset(images, expected="image_only") == "image_only"
        assert "object_detection" in DatasetKind.__args__

    def test_utils_data_rejects_wrong_shape(self):
        from dataeval.exceptions import MaiteShapeError
        from dataeval.utils.data import validate_dataset

        images = [np.zeros((3, 4, 4), dtype=np.uint8) for _ in range(3)]
        with pytest.raises(MaiteShapeError):
            validate_dataset(images, expected="object_detection")

    def test_utils_onnx_graph_utilities(self):
        from dataeval.utils import onnx

        assert callable(onnx.find_embedding_layer)
        assert callable(onnx.to_encoding_model)

    def test_utils_models_architectures(self):
        import torch

        from dataeval.utils.models import AE, VAE, GMMDensityNet

        autoencoder = AE(input_shape=(1, 8, 8))
        reconstruction = autoencoder(torch.zeros(2, 1, 8, 8))
        assert reconstruction.shape == (2, 1, 8, 8)
        assert isinstance(VAE(input_shape=(1, 8, 8)), torch.nn.Module)
        assert isinstance(GMMDensityNet(latent_dim=4), torch.nn.Module)

    def test_utils_losses_elbo(self):
        import torch

        from dataeval.utils.losses import ELBOLoss

        loss = ELBOLoss()
        value = loss(torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 3))
        assert torch.is_tensor(value)

    def test_utils_training_helpers(self):
        from dataeval.utils.training import predict, train

        assert callable(train)
        assert callable(predict)
