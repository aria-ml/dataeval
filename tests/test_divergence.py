import numpy as np
import pytest

from daml.metrics.divergence import FNN, MST, DivergenceOutput


class TestDpDivergence:
    @pytest.mark.parametrize(
        "input, output",
        [
            (
                MST,
                DivergenceOutput(
                    dpdivergence=0.96875,
                    error=1,
                ),
            ),
            (
                FNN,
                DivergenceOutput(
                    dpdivergence=1.0,
                    error=0.0,
                ),
            ),
        ],
    )
    def test_dp_divergence(self, input, output):
        """Unit testing of Dp Divergence

        TBD
        """
        # Initialize a dataset of 32 images of size 32x32x3, containing all 1's
        all_ones = np.ones(shape=(32, 32))
        all_fives = all_ones * 5

        # Initialize the autoencoder-based outlier detector from alibi-detect
        metric = input()
        result = metric.evaluate(dataset_a=all_ones, dataset_b=all_fives)
        assert result == output
