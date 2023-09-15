import numpy as np
import pytest

import daml
from daml._internal.MetricOutputs import BEROutput
from daml._internal.utils import Metrics


class TestMulticlassBER:
    def test_multiclass_ber_with_randos(self):
        """
        Two classes with a single, identically distributed covariate.
        Should spit out a large number. Used numpy.random to generate
        datasets and then hardcoded those to avoid the random numbers
        in the pipeline.
        """
        # The expected output...
        expected_result = BEROutput(ber=0.45)

        # Initialize a 2nxp  (2nx1) numpy array of standard gaussians
        # np.random.seed(37)

        # Initialize a numpy array of random numbers distributed around 1
        # covariates = np.random.normal(
        #     size=(2 * n, 1),
        # )
        covariates = np.array(
            [
                [-0.05446361],
                [0.67430807],
                [0.34664703],
                [-1.30034617],
                [1.51851188],
                [0.98982371],
                [0.2776809],
                [-0.44858935],
                [0.96196624],
                [-0.82757864],
                [0.53465707],
                [1.22838619],
                [0.51959233],
                [-0.06335482],
                [-0.03479336],
                [0.04556555],
                [1.44802513],
                [1.89350553],
                [0.4030323],
                [0.19242609],
            ]
        )

        # Initialize the class labels. First n are 0, next n are 1
        # labels = np.concatenate((np.zeros((n, 1)), np.ones((n, 1))))
        labels = np.array(
            [
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
            ]
        )

        metric = daml.load_metric(
            metric=Metrics.BER,
            provider=Metrics.Provider.ARiA,
            method=Metrics.Method.MultiClassBER,
        )

        assert (
            metric.evaluate(
                X=covariates,
                y=labels,
            )
            == expected_result
        )

    def test_class_max(self):
        value = None
        covariates = np.ones(20)
        labels = np.array(range(20))
        metric = daml.load_metric(
            metric=Metrics.BER,
            provider=Metrics.Provider.ARiA,
            method=Metrics.Method.MultiClassBER,
        )
        with pytest.raises(ValueError):
            value = metric.evaluate(
                X=covariates,
                y=labels,
            )
            assert value is not None

    def test_class_min(self):
        value = None
        covariates = np.ones(20)
        labels = np.ones(20)
        metric = daml.load_metric(
            metric=Metrics.BER,
            provider=Metrics.Provider.ARiA,
            method=Metrics.Method.MultiClassBER,
        )
        with pytest.raises(ValueError):
            value = metric.evaluate(
                X=covariates,
                y=labels,
            )
            assert value is not None
