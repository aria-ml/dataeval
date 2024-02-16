import numpy as np
import pytest
import torch

from daml._internal.models.pytorch.utils import (
    numpy_to_pytorch,
    permute_to_numpy,
    permute_to_torch,
    pytorch_to_numpy,
)
from daml.metrics.outlier_detection import OD_AE, OD_AEGMM, OD_LLR, OD_VAE, OD_VAEGMM


class TestDatasetType:
    def test_dataset_type_is_none(self):
        """OD_AE does not have a dataset type requirement"""
        metric = OD_AE()
        images = np.array([])
        metric._check_dtype(images)

    @pytest.mark.parametrize("method", [OD_AE, OD_AEGMM, OD_VAE, OD_VAEGMM, OD_LLR])
    def test_dataset_type_is_incorrect(self, method):
        """Methods with type requirements raise TypeError"""
        images = np.ones(shape=[1, 32, 32, 3]).astype(int)

        metric = method()
        if metric._dataset_type:
            with pytest.raises(TypeError):
                metric._check_dtype(images)
        else:
            metric._check_dtype(images)

    def test_dataset_type_is_not_numpy(self):
        """
        Methods require numpy for TensorFlow models
        TODO: Add TensorFlow Tensors and automatic format conversions to AlibiBase
        """
        metric = OD_AEGMM()

        pt_images = torch.ones(size=[1, 3, 32, 32])
        with pytest.raises(TypeError):
            metric._check_dtype(pt_images)  # type: ignore


class TestFlatten:
    @pytest.mark.parametrize(
        "count",
        [1, 10],
    )
    @pytest.mark.parametrize(
        "img_dims",
        [(1, 1), (32, 32), (16, 64)],
    )
    @pytest.mark.parametrize(
        "channels",
        [1, 5],
    )
    def test_flatten_dataset_is_true(self, count, img_dims, channels):
        """Prove that the flatten dataset only affects the image shape, not batch"""
        # Define data
        images = np.ones(shape=[count, img_dims[0], img_dims[1], channels])
        # Define model
        metric = OD_AE()
        metric._flatten_dataset = True
        new_dataset = metric._format_images(images)
        output_shape = img_dims[0] * img_dims[1] * channels

        assert new_dataset.shape[0] == count
        assert new_dataset.shape[1] == output_shape


class TestConversions:
    def test_pt_is_pt(self):
        """Convert PyTorch Tensor to Tensor returns unmodified Tensor"""
        images = torch.ones(size=(1, 3, 32, 32))  # NCHW
        result = numpy_to_pytorch(images)  # type: ignore

        assert isinstance(images, torch.Tensor)
        assert images is result  # If unmodified, points to same object

    def test_np_is_np(self):
        """Convert NumPy NDArray to NDArray returns unmodified NDArray"""
        images = np.ones(shape=(1, 32, 32, 3))  # NHWC
        result = pytorch_to_numpy(images)  # type: ignore

        assert isinstance(images, np.ndarray)
        assert images is result  # If unmodified, points to same object

    def test_pt_np_conv(self):
        """Convert PyTorch Tensor to NumPy NDArray"""
        pt_images = torch.ones(size=(1, 3, 32, 32))
        np_images = pytorch_to_numpy(pt_images)

        assert isinstance(np_images, np.ndarray)
        assert np_images.shape == pt_images.shape

    def test_np_pt_conv(self):
        """Convert NumPy NDArray to PyTorch Tensor"""
        np_images = np.ones(shape=(1, 32, 32, 3))
        pt_images = numpy_to_pytorch(np_images)

        assert isinstance(pt_images, torch.Tensor)
        assert pt_images.shape == np_images.shape

    def test_pt_np_permute(self):
        """Convert PyTorch Tensor to NumPy NDArray and modify order of CHW to HWC"""
        pt_images = torch.ones(size=(1, 3, 32, 32))
        np_images = permute_to_numpy(pt_images)

        assert isinstance(np_images, np.ndarray)
        assert np_images.shape == (1, 32, 32, 3)

    def test_np_pt_permute(self):
        """Convert NumPy NDArray to PyTorch Tensor and modify order of HWC to CHW"""
        np_images = np.ones(shape=(1, 32, 32, 3))
        pt_images = permute_to_torch(np_images)

        assert isinstance(pt_images, torch.Tensor)
        assert pt_images.shape == (1, 3, 32, 32)

    def test_pt_np_typeerror(self):
        with pytest.raises(TypeError):
            pytorch_to_numpy(0)  # type: ignore

    def test_np_pt_typeerror(self):
        with pytest.raises(TypeError):
            numpy_to_pytorch(0)  # type: ignore
