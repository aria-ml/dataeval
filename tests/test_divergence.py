import numpy as np
import pytest

import daml
from daml._internal.utils import Metrics


class TestDpDivergence:
    @pytest.mark.parametrize(
        "input, output",
        [
            (
                Metrics.Algorithm.MinimumSpanningTree,
                {
                    Metrics.Method.DpDivergence: -0.96875,
                    "Error": 63,
                },
            ),
            (
                Metrics.Algorithm.FirstNearestNeighbor,
                {
                    Metrics.Method.DpDivergence: 1.0,
                    "Error": 0.0,
                },
            ),
        ],
    )
    def test_dp_divergence(self, input, output):
        """Unit testing of Dp Divergence

        TBD
        """
        # Initialize a dataset of 32 images of size 32x32x3, containing all 1's
        all_ones = np.ones(shape=(32, 32))
        all_fives = all_ones * 13

        # Initialize the autoencoder-based outlier detector from alibi-detect
        metric = daml.load_metric(
            metric=Metrics.Divergence,
            provider=Metrics.Provider.ARiA,
            method=Metrics.Method.DpDivergence,
        )
        result = metric.evaluate(
            dataset_a=all_ones,
            dataset_b=all_fives,
            algorithm=input,
        )
        assert result == output

    def test_invalid_algorithm(self):
        with pytest.raises(ValueError):
            metric = daml.load_metric(
                metric=Metrics.Divergence,
                provider=Metrics.Provider.ARiA,
                method=Metrics.Method.DpDivergence,
            )
            result = metric.evaluate(
                dataset_a=np.array([1, 2, 3]),
                dataset_b=np.array([4, 5, 6]),
                algorithm="not valid",
            )
            # should not get here!
            assert result == (1.0, 0.0)
