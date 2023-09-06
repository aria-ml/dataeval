import numpy as np
import pytest

import daml
from daml._internal.MetricClasses import Metrics

from .utils import MockImageClassificationGenerator


@pytest.mark.parametrize(
    "input",
    [
        Metrics.Method.AutoEncoder,
        Metrics.Method.VariationalAutoEncoder,
        Metrics.Method.AutoEncoderGMM,
        Metrics.Method.VariationalAutoEncoderGMM,
        Metrics.Method.LLR,
    ],
)
class TestAlibi:
    # Test main functionality of the program
    @pytest.mark.functional
    def test_label_5s_as_outliers(self, input):
        """Functional testing of Alibi OutlierDection

        The AlibiAE model is being trained on all 1's and tested on all 5's.
        When evaluating, the model should say all 1's are not outliers
        and all 5's are outliers
        """
        # Initialize a dataset of 32 images of size 32x32x3, containing all 1's
        all_ones = MockImageClassificationGenerator(
            limit=1, labels=1, img_dims=(32, 32), channels=3
        )

        # Initialize a dataset of 32 images of size 32x32x3, containing all 5's
        all_fives = MockImageClassificationGenerator(
            limit=1, labels=5, img_dims=(32, 32), channels=3
        )

        # Get model input from each dataset
        X_all_ones = all_ones.dataset.images
        X_all_fives = all_fives.dataset.images

        # Initialize the autoencoder-based outlier detector from alibi-detect
        metric = daml.load_metric(
            metric=Metrics.OutlierDetection,
            provider=Metrics.Provider.AlibiDetect,
            method=input,
        )

        # TODO Need to create a helper function to handle this
        if metric._DATASET_TYPE is not None:
            X_all_ones = X_all_ones.astype(metric._DATASET_TYPE)
            X_all_fives = X_all_fives.astype(metric._DATASET_TYPE)

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

    # Ensure that the program fails upon wrong order of operations
    def test_eval_before_fit_fails(self, input):
        """Testing incorrect order of operations for fitting and evaluating"""

        all_ones = MockImageClassificationGenerator(
            limit=3, labels=1, img_dims=(32, 32), channels=3
        )

        X = all_ones.dataset.images

        metric = daml.load_metric(
            metric=Metrics.OutlierDetection,
            provider=Metrics.Provider.AlibiDetect,
            method=input,
        )

        # force
        # metric.is_trained = False
        # Evaluate dataset before fitting it
        with pytest.raises(TypeError):
            metric.evaluate(X)

    def test_missing_detector(self, input):
        all_ones = MockImageClassificationGenerator(
            limit=1, labels=1, img_dims=(32, 32), channels=3
        )

        X = all_ones.dataset.images

        metric = daml.load_metric(
            metric=Metrics.OutlierDetection,
            provider=Metrics.Provider.AlibiDetect,
            method=input,
        )

        # Force the detector to not be initialized
        metric.detector = None
        with pytest.raises(TypeError):
            metric.fit_dataset(dataset=X, epochs=1, verbose=False)

        with pytest.raises(TypeError):
            metric.evaluate(dataset=X)
