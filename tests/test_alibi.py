from typing import Optional
from unittest.mock import patch

import numpy as np
import pytest

from daml.metrics.outlier_detection import (
    AE,
    AEGMM,
    LLR,
    VAE,
    VAEGMM,
    Threshold,
    ThresholdType,
)

from .utils import MockImageClassificationGenerator


@pytest.mark.functional
@pytest.mark.parametrize("method", [AE, AEGMM, VAE, VAEGMM, LLR])
class TestAlibiDetect_Functional:
    # Test main functionality of the program
    def test_label_5s_as_outliers(self, method):
        """Functional testing of  OutlierDection

        The AE model is being trained on all 1's and tested on all 5's.
        When evaluating, the model should say all 1's are not outliers
        and all 5's are outliers
        """

        input_shape = (32, 32, 3)

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

        # Initialize the autoencoder-based outlier detector from alibi-detect
        metric = method()

        metric.initialize_detector(input_shape)

        # TODO Need to create a helper function to handle this
        if metric._dataset_type is not None:
            X_all_ones = X_all_ones.astype(metric._dataset_type)
            X_all_fives = X_all_fives.astype(metric._dataset_type)

        # Train the detector on the dataset of all 1's
        metric.fit_dataset(dataset=X_all_ones, epochs=10, verbose=False)

        # Evaluate the detector on the dataset of all 1's
        preds_on_ones = metric.evaluate(X_all_ones).is_outlier
        # print(preds_on_ones)
        # ones_outliers = preds_on_ones
        # Evaluate the detector on the dataset of all 5's
        preds_on_fives = metric.evaluate(X_all_fives).is_outlier
        # fives_outliers = preds_on_fives

        # We expect all elements to not be outliers
        num_errors_on_ones = np.sum(np.where(preds_on_ones != 0))
        # We expect all elements to be outliers
        num_errors_on_fives = np.sum(np.where(preds_on_fives != 1))

        assert len(preds_on_ones) == len(X_all_ones)
        assert len(preds_on_fives) == len(X_all_fives)

        assert num_errors_on_ones == 0
        assert num_errors_on_fives == 0

    def test_different_input_shape(self, method):
        """Testing of  Detection under different input size"""

        input_shape = (38, 38, 2)

        # Initialize a dataset of 32 images of size 32x32x3, containing all 1's
        all_ones = MockImageClassificationGenerator(
            limit=1, labels=1, img_dims=input_shape
        )

        # Get model input from each dataset
        X_all_ones = all_ones.dataset.images

        metric = method()

        # Initialize the autoencoder-based outlier detector from alibi-detect
        metric.initialize_detector(input_shape)

        # TODO Need to create a helper function to handle this
        if metric._dataset_type is not None:
            X_all_ones = X_all_ones.astype(metric._dataset_type)

        # Train the detector on the dataset of all 1's
        metric.fit_dataset(dataset=X_all_ones, epochs=10, verbose=False)

        # Evaluate the detector on the dataset of all 1's
        preds_on_ones = metric.evaluate(X_all_ones).is_outlier

        num_errors_on_ones = np.sum(np.where(preds_on_ones != 0))

        assert num_errors_on_ones == 0


@pytest.mark.parametrize(
    "method, classnames",
    [
        (AE, ["alibi_detect.od.OutlierAE"]),
        (AEGMM, ["alibi_detect.od.OutlierAEGMM"]),
        (
            LLR,
            ["alibi_detect.od.LLR", "daml._internal.metrics.alibi_detect.llr.PixelCNN"],
        ),
        (VAE, ["alibi_detect.od.OutlierVAE"]),
        (VAEGMM, ["alibi_detect.od.OutlierVAEGMM"]),
    ],
)
class TestAlibiDetect:
    input_shape = (32, 32, 3)
    all_ones = MockImageClassificationGenerator(limit=3, labels=1, img_dims=input_shape)
    dataset = all_ones.dataset.images

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

    # Ensure that the program fails upon wrong order of operations
    def test_eval_before_fit_fails(self, method):
        """Testing incorrect order of operations for fitting and evaluating"""

        dataset = self.all_ones.dataset.images
        metric = method()
        metric.initialize_detector(self.input_shape)

        # force metric.is_trained = False
        # Evaluate dataset before fitting it
        with pytest.raises(TypeError):
            metric.evaluate(dataset)

    # Ensure that the program fails upon testing on a dataset of different
    # shape than what was trained on
    def test_wrong_dataset_dims_fails(self, method):
        """Testing incorrect order of operations for fitting and evaluating"""

        faulty_input_shape = (11, 32, 3)

        all_ones = MockImageClassificationGenerator(
            limit=3,
            labels=1,
            img_dims=self.input_shape[:1],
            channels=self.input_shape[2],
        )

        faulty_all_ones = MockImageClassificationGenerator(
            limit=3,
            labels=1,
            img_dims=faulty_input_shape[:1],
            channels=faulty_input_shape[2],
        )

        dataset = all_ones.dataset.images
        faulty_dataset = faulty_all_ones.dataset.images

        metric = method()

        if metric._dataset_type:
            dataset = dataset.astype(metric._dataset_type)
            faulty_dataset = faulty_dataset.astype(metric._dataset_type)

        metric.initialize_detector(self.input_shape)

        metric.fit_dataset(dataset=dataset, epochs=1, verbose=False)

        # force
        # metric.is_trained = False
        # Evaluate dataset before fitting it
        with pytest.raises(TypeError):
            metric.evaluate(faulty_dataset)

    def test_missing_detector(self, method):
        self._is_mock_expected = False

        dataset = self.all_ones.dataset.images

        metric = method()

        with pytest.raises(TypeError):
            metric.fit_dataset(dataset=dataset, epochs=1, verbose=False)

        with pytest.raises(TypeError):
            metric.evaluate(dataset=dataset)

    def test_initialize_fit_evaluate(self, method):
        metric = method()
        metric.initialize_detector(self.input_shape)
        if metric._dataset_type:
            dataset = self.dataset.astype(metric._dataset_type)
        else:
            dataset = self.dataset
        metric.fit_dataset(dataset=dataset, epochs=1, verbose=False)
        metric.evaluate(dataset)

    def test_set_threshold_value(self, method):
        metric = method()
        metric.initialize_detector(self.input_shape)
        if metric._dataset_type:
            dataset = self.dataset.astype(metric._dataset_type)
        else:
            dataset = self.dataset
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

    def test_set_model_is_retained(self, method):
        self._is_mock_expected = None
        metric = method()
        metric._model_kwargs.update({"model": 0})
        metric.initialize_detector(self.input_shape)
        assert metric._model_kwargs["model"] == 0
