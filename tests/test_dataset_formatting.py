import numpy as np
import pytest

from daml.metrics.outlier_detection import AE, AEGMM, LLR, VAE, VAEGMM

from .utils import MockImageClassificationGenerator


class TestDatasetType:
    def test_dataset_type_is_none(self):
        metric = AE()

        all_ones = MockImageClassificationGenerator(
            limit=1, labels=1, img_dims=(32, 32), channels=3
        )
        all_ones_images = all_ones.dataset.images
        metric.check_dtype(all_ones_images, None)

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(int, marks=pytest.mark.functional),
            np.float64,
            np.float32,
            pytest.param(np.float16, marks=pytest.mark.functional),
            pytest.param(float, marks=pytest.mark.functional),
        ],
    )
    @pytest.mark.parametrize(
        "method",
        [
            AE,
            AEGMM,
            VAE,
            VAEGMM,
            # remove functional marker after issue #94 is resolved
            pytest.param(LLR, marks=pytest.mark.functional),
        ],
    )
    def test_dataset_type_is_incorrect(self, dtype, method):
        metric = method()

        all_ones = MockImageClassificationGenerator(
            limit=1, labels=1, img_dims=(32, 32), channels=3
        )
        # TODO Should move dtype casting into MockImageClassificationGenerator
        # Cast dtype to force error when dataset dtype is not the required
        images = all_ones.dataset.images.astype(dtype)

        metric_dtype = metric._DATASET_TYPE
        if dtype == metric_dtype or metric_dtype is None:
            metric.check_dtype(images, metric_dtype)
        else:
            with pytest.raises(TypeError):
                metric.check_dtype(images, metric_dtype)

    def test_dataset_type_is_not_numpy(self):
        metric = AEGMM()

        all_ones = MockImageClassificationGenerator(
            limit=1, labels=1, img_dims=(32, 32), channels=3
        )
        images_list = list(all_ones.dataset.images)
        with pytest.raises(TypeError):
            metric.check_dtype(images_list, metric._DATASET_TYPE)  # type: ignore


class TestFlatten:
    def test_flatten_dataset_is_none(self):
        """Input and output shape are equivalent if no flatten is done"""
        # Define data
        all_ones = MockImageClassificationGenerator(
            limit=1, labels=1, img_dims=(32, 32), channels=3
        )
        # Define model
        metric = AE()

        images = all_ones.dataset.images
        new_dataset = metric.format_dataset(
            images, flatten_dataset=None  # type: ignore
        )
        assert new_dataset.shape == images.shape

    @pytest.mark.parametrize(
        "limit",
        [
            1,
            pytest.param(5, marks=pytest.mark.functional),
            pytest.param(25, marks=pytest.mark.functional),
        ],
    )
    @pytest.mark.parametrize(
        "img_dims",
        [
            (1, 1),
            pytest.param((32, 32), marks=pytest.mark.functional),
            pytest.param((16, 64), marks=pytest.mark.functional),
        ],
    )
    @pytest.mark.parametrize(
        "channels",
        [
            1,
            pytest.param(3, marks=pytest.mark.functional),
            pytest.param(5, marks=pytest.mark.functional),
        ],
    )
    def test_flatten_dataset_is_true(self, limit, img_dims, channels):
        """Prove that the flatten dataset only affects the image shape, not batch"""
        # Define data
        all_ones = MockImageClassificationGenerator(
            limit=limit, labels=1, img_dims=img_dims, channels=channels
        )
        # Define model
        metric = AE()
        images = all_ones.dataset.images
        new_dataset = metric.format_dataset(images, flatten_dataset=True)
        output_shape = img_dims[0] * img_dims[1] * channels

        assert new_dataset.shape[0] == limit
        assert new_dataset.shape[1] == output_shape
