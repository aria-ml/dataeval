import numpy as np
import pytest

import daml
from daml._internal.utils import Metrics

from .utils import MockImageClassificationGenerator


class TestDatasetType:
    def test_dataset_type_is_none(self):
        metric = daml.load_metric(
            metric=Metrics.OutlierDetection,
            provider=Metrics.Provider.AlibiDetect,
            method=Metrics.Method.AutoEncoder,
        )

        all_ones = MockImageClassificationGenerator(
            limit=1, labels=1, img_dims=(32, 32), channels=3
        )
        all_ones_images = all_ones.dataset.images
        metric.check_dtype(all_ones_images, None)

    @pytest.mark.parametrize("dtype", [int, np.float64, np.float32, np.float16, float])
    @pytest.mark.parametrize(
        "method",
        [
            Metrics.Method.AutoEncoder,
            Metrics.Method.AutoEncoderGMM,
            Metrics.Method.VariationalAutoEncoder,
            Metrics.Method.VariationalAutoEncoderGMM,
            Metrics.Method.LLR,
        ],
    )
    def test_dataset_type_is_incorrect(self, dtype, method):
        metric = daml.load_metric(
            metric=Metrics.OutlierDetection,
            provider=Metrics.Provider.AlibiDetect,
            method=method,
        )

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
        metric = daml.load_metric(
            metric=Metrics.OutlierDetection,
            provider=Metrics.Provider.AlibiDetect,
            method=Metrics.Method.AutoEncoderGMM,
        )

        all_ones = MockImageClassificationGenerator(
            limit=1, labels=1, img_dims=(32, 32), channels=3
        )
        images_list = list(all_ones.dataset.images)
        with pytest.raises(TypeError):
            metric.check_dtype(images_list, metric._DATASET_TYPE)


class TestFlatten:
    def test_flatten_dataset_is_none(self):
        """Input and output shape are equivalent if no flatten is done"""
        # Define data
        all_ones = MockImageClassificationGenerator(
            limit=1, labels=1, img_dims=(32, 32), channels=3
        )
        # Define model
        metric = daml.load_metric(
            metric=Metrics.OutlierDetection,
            provider=Metrics.Provider.AlibiDetect,
            method=Metrics.Method.AutoEncoder,
        )

        images = all_ones.dataset.images
        new_dataset = metric.format_dataset(images, flatten_dataset=None)
        assert new_dataset.shape == images.shape

    @pytest.mark.parametrize("limit", [1, 5, 25])
    @pytest.mark.parametrize("img_dims", [(1, 1), (32, 32), (16, 64)])
    @pytest.mark.parametrize("channels", [1, 3, 5])
    def test_flatten_dataset_is_true(self, limit, img_dims, channels):
        """Prove that the flatten dataset only affects the image shape, not batch"""
        # Define data
        all_ones = MockImageClassificationGenerator(
            limit=limit, labels=1, img_dims=img_dims, channels=channels
        )
        # Define model
        metric = daml.load_metric(
            metric=Metrics.OutlierDetection,
            provider=Metrics.Provider.AlibiDetect,
            method=Metrics.Method.AutoEncoder,
        )
        images = all_ones.dataset.images
        new_dataset = metric.format_dataset(images, flatten_dataset=True)
        output_shape = img_dims[0] * img_dims[1] * channels

        assert new_dataset.shape[0] == limit
        assert new_dataset.shape[1] == output_shape
