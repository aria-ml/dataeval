import numpy as np
import pytest

from daml.metrics.divergence import HP_FNN, HP_MST, DivergenceOutput


class TestDpDivergence:
    @pytest.mark.parametrize(
        "input, output",
        [
            (
                HP_MST,
                DivergenceOutput(dpdivergence=0.8377897755491117, error=81.0),
            ),
            (
                HP_FNN,
                DivergenceOutput(
                    dpdivergence=0.8618209199122062,
                    error=69.0,
                ),
            ),
        ],
    )
    def test_dp_divergence(self, input, output):
        """Unit testing of Dp Divergence

        TBD
        """
        path = "tests/datasets/mnist.npz"
        with np.load(path, allow_pickle=True) as fp:
            covariates, labels = fp["x_train"][:1000], fp["y_train"][:1000]

        inds = np.array([x in [0, 2, 4, 6, 8] for x in labels])
        rev_inds = np.invert(inds)
        even = covariates[inds, :, :]
        odd = covariates[rev_inds, :, :]
        even = even.reshape((even.shape[0], -1))
        odd = odd.reshape((odd.shape[0], -1))
        metric = input()
        result = metric.evaluate(
            dataset_a=even,
            dataset_b=odd,
        )
        assert result == output
