from os import remove
from os.path import exists
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from daml._internal.models.tensorflow.alibi import create_model
from daml.datasets import DamlDataset
from daml.metrics.outlier_detection import (
    OD_AE,
    OD_AEGMM,
    OD_LLR,
    OD_VAE,
    OD_VAEGMM,
    Threshold,
    ThresholdType,
)
from tests.utils import MockImageClassificationGenerator


@pytest.mark.functional
@pytest.mark.parametrize("method", [OD_AE, OD_AEGMM, OD_VAE, OD_VAEGMM, OD_LLR])
class TestAlibiDetect_Functional:
    # Test main functionality of the program
    def test_label_5s_as_outliers(self, method):
        """Functional testing of  OutlierDection

        The AE model is being trained on all 1's and tested on all 5's.
        When evaluating, the model should say all 1's are not outliers
        and all 5's are outliers
        """

        input_shape = (32, 32, 3)

        # Initialize the autoencoder-based outlier detector from alibi-detect
        metric = method()

        # Initialize a dataset of 32 images of size 32x32x3, containing all 1's
        all_ones = MockImageClassificationGenerator(
            limit=1, labels=1, img_dims=input_shape
        )

        # Initialize a dataset of 32 images of size 32x32x3, containing all 5's
        all_fives = MockImageClassificationGenerator(
            limit=1, labels=5, img_dims=input_shape
        )

        # Get model input from each dataset
        X_all_ones = all_ones.dataset.images
        X_all_fives = all_fives.dataset.images

        if metric._dataset_type is not None:
            X_all_ones = X_all_ones.astype(metric._dataset_type)
            X_all_fives = X_all_fives.astype(metric._dataset_type)

        all_ones_ds = DamlDataset(X_all_ones)
        all_fives_ds = DamlDataset(X_all_fives)
        # Train the detector on the dataset of all 1's
        metric.fit_dataset(dataset=all_ones_ds, epochs=10, verbose=False)

        # Evaluate the detector on the dataset of all 1's
        preds_on_ones = metric.evaluate(all_ones_ds).is_outlier

        # Evaluate the detector on the dataset of all 5's
        preds_on_fives = metric.evaluate(all_fives_ds).is_outlier

        # We expect all elements to not be outliers
        num_errors_on_ones = np.sum(np.where(preds_on_ones != 0))
        # We expect all elements to be outliers
        num_errors_on_fives = np.sum(np.where(preds_on_fives != 1))

        # Test
        # Assert there is a prediction for each image
        assert len(preds_on_ones) == len(X_all_ones)
        assert len(preds_on_fives) == len(X_all_fives)
        # Assert there are no errors
        assert num_errors_on_ones == 0
        assert num_errors_on_fives == 0

    def test_different_input_shape(self, method):
        """
        Confirm autoencoders can encode and decode arbitrary image sizes
        without breaking fit_dataset or evaluate
        """

        input_shape = (38, 38, 2)
        metric = method()
        # Initialize the autoencoder-based outlier detector from alibi-detect

        # Initialize a dataset of 32 images of size 32x32x3, containing all 1's
        all_ones = MockImageClassificationGenerator(
            limit=1, labels=1, img_dims=input_shape
        )

        X_all_ones = all_ones.dataset.images
        if metric._dataset_type is not None:
            X_all_ones = X_all_ones.astype(metric._dataset_type)

        all_ones_ds = DamlDataset(X_all_ones)

        # Train the detector on the dataset of all 1's
        metric.fit_dataset(dataset=all_ones_ds, epochs=10, verbose=False)

        # Evaluate the detector on the dataset of all 1's
        preds_on_ones = metric.evaluate(all_ones_ds).is_outlier

        num_errors_on_ones = np.sum(np.where(preds_on_ones != 0))

        # Test
        assert num_errors_on_ones == 0


@pytest.mark.parametrize(
    "method, classnames",
    [
        (OD_AE, ["daml._alibi_detect.od.OutlierAE"]),
        (OD_AEGMM, ["daml._alibi_detect.od.OutlierAEGMM"]),
        (
            OD_LLR,
            [
                "daml._alibi_detect.od.LLR",
                "daml._internal.models.tensorflow.alibi.PixelCNN",
            ],
        ),
        (OD_VAE, ["daml._alibi_detect.od.OutlierVAE"]),
        (OD_VAEGMM, ["daml._alibi_detect.od.OutlierVAEGMM"]),
    ],
)
class TestAlibiDetect:
    input_shape = (32, 32, 3)
    all_ones = MockImageClassificationGenerator(limit=3, labels=1, img_dims=input_shape)

    _is_mock_expected: Optional[bool] = True

    @pytest.fixture(scope="function", autouse=True)
    def mock_classes_and_validate(self, classnames):
        mocks = list()
        for cls in classnames:
            mock = patch(cls)
            mocks.append((mock, mock.start()))
        yield
        for mock in mocks:
            if self._is_mock_expected is not None:
                assert mock[1].called == self._is_mock_expected
            mock[0].stop()
        self._is_mock_expected = True

    # Ensure that the program fails upon wrong order of operations
    def test_eval_before_fit_fails(self, method):
        self._is_mock_expected = False
        """Raises error when evaluate is called before fitting a model"""
        # Load metric and create model
        metric = method()

        # Create Daml dataset
        dataset = DamlDataset(self.all_ones.dataset.images)

        # Test
        with pytest.raises(TypeError):
            metric.evaluate(dataset)

    # Ensure that the program fails upon testing on a dataset of different
    # shape than what was trained on
    def test_wrong_dataset_dims_fails(self, method):
        self._is_mock_expected = None
        """
        Raises an error when image shape in evaluate
        does not match the detector input
        """
        # Load metric with an incorrect input shape
        faulty_input_shape = (31, 32, 3)
        metric = method()
        metric.detector = 1  # detector cannot be None
        metric._input_shape = faulty_input_shape

        # Create daml dataset
        images = self.all_ones.dataset.images
        if metric._dataset_type:
            dataset = images.astype(metric._dataset_type)
        dataset = DamlDataset(images=images)

        # metric.fit_dataset(dataset=dataset, epochs=1, verbose=False)
        metric.is_trained = True

        with pytest.raises(ValueError):
            metric.evaluate(dataset)

    def test_missing_detector(self, method):
        self._is_mock_expected = False
        """
        Raises error if fit_dataset or evaluate are called without a detector
        """
        # Load metric
        metric = method()

        # Create Daml datasets
        images = self.all_ones.dataset.images
        dataset = DamlDataset(images)

        # Test
        with pytest.raises(TypeError):
            metric.evaluate(dataset=dataset)

    def test_initialize_fit_evaluate(self, method):
        """What does this test? Is this a pseudo functional test?"""
        # Load metric and create model
        metric = method()

        # Load dataset
        images = self.all_ones.dataset.images
        if metric._dataset_type:
            images = images.astype(metric._dataset_type)
        dataset = DamlDataset(images)

        # Test
        metric.fit_dataset(dataset=dataset, epochs=1, verbose=False)
        metric.evaluate(dataset)

    def test_set_threshold_value(self, method):
        """Asserts that the threshold is set by fit_dataset"""
        # Load metric and create model
        metric = method()

        # Load dataset and set type
        images = self.all_ones.dataset.images
        if metric._dataset_type:
            images = images.astype(metric._dataset_type)
        dataset = DamlDataset(images)

        # Test
        metric.fit_dataset(
            dataset=dataset,
            epochs=1,
            verbose=False,
            threshold=Threshold(0.015, ThresholdType.VALUE),
        )
        assert metric.detector.threshold == 0.015

    def test_set_prediction_args(self, method):
        self._is_mock_expected = False
        metric = method()
        expected = {
            k: v
            for k, v in metric._predict_kwargs.items()
            if k != "return_instance_score"
        }
        metric.set_prediction_args(return_instance_score=False)
        assert not metric._predict_kwargs["return_instance_score"]
        for key in expected:
            assert metric._predict_kwargs[key] == expected[key]


class TestAlibiModels:
    input_shape = (32, 32, 3)
    all_ones = MockImageClassificationGenerator(limit=3, labels=1, img_dims=input_shape)

    def test_export_model(self):
        metric = OD_AE()
        dataset = DamlDataset(self.all_ones.dataset.images)
        metric.fit_dataset(dataset=dataset, epochs=1, verbose=False)

        metric.export_model("model.keras")
        if exists("model.keras"):
            remove("model.keras")
        else:
            pytest.fail("model not exported")

    def test_export_model_no_detector(self):
        metric = OD_AE()
        with pytest.raises(RuntimeError):
            metric.export_model("model.keras")

    def test_export_model_no_model(self):
        metric = OD_AE()
        dataset = DamlDataset(self.all_ones.dataset.images)
        metric.fit_dataset(dataset=dataset, epochs=1, verbose=False)
        metric._model_param_name = "llr"
        with pytest.raises(ValueError):
            metric.export_model("model.keras")

    @patch("daml._internal.models.tensorflow.alibi.create_model")
    def test_fit_dataset_with_model_calls_infer_and_fit(self, create_model_fn):
        metric = OD_AE()
        dataset = DamlDataset(self.all_ones.dataset.images)
        metric.detector = MagicMock()
        metric._input_shape = self.input_shape
        metric.fit_dataset(dataset=dataset, epochs=1, verbose=False)
        assert metric.detector.infer_threshold.called
        assert metric.detector.fit.called
        assert not create_model_fn.called

    def test_create_model_invalid_class(self):
        with pytest.raises(TypeError):
            create_model("not_a_valid_class", self.input_shape)
