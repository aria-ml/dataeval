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
        metric._check_dtype(all_ones_images)

    @pytest.mark.parametrize("method", [AE, AEGMM, VAE, VAEGMM, LLR])
    def test_dataset_type_is_incorrect(self, method):
        all_ones = MockImageClassificationGenerator(
            limit=1, labels=1, img_dims=(32, 32), channels=3
        )

        images = all_ones.dataset.images.astype(int)

        metric = method()
        if metric._dataset_type:
            with pytest.raises(TypeError):
                metric._check_dtype(images)
        else:
            metric._check_dtype(images)

    def test_dataset_type_is_not_numpy(self):
        metric = AEGMM()

        all_ones = MockImageClassificationGenerator(
            limit=1, labels=1, img_dims=(32, 32), channels=3
        )
        images_list = list(all_ones.dataset.images)
        with pytest.raises(TypeError):
            metric._check_dtype(images_list)  # type: ignore


class TestFlatten:
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
        metric._flatten_dataset = True
        new_dataset = metric._format_dataset(images)
        output_shape = img_dims[0] * img_dims[1] * channels

        assert new_dataset.shape[0] == limit
        assert new_dataset.shape[1] == output_shape
