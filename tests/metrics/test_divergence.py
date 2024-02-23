import numpy as np
import pytest

from daml.metrics.divergence import Divergence, DivergenceOutput

np.random.seed(0)


class TestDpDivergence:
    @pytest.mark.parametrize(
        "method, output",
        [
            (
                "MST",
                DivergenceOutput(
                    dpdivergence=0.8377897755491117,
                    error=81.0,
                ),
            ),
            (
                "FNN",
                DivergenceOutput(
                    dpdivergence=0.8618209199122062,
                    error=69.0,
                ),
            ),
        ],
    )
    def test_dp_divergence(self, mnist, method, output):
        """Unit testing of Dp Divergence

        TBD
        """

        covariates, labels = mnist(add_channels="channels_last")

        inds = np.array([x % 2 == 0 for x in labels])
        rev_inds = np.invert(inds)
        even = covariates[inds, :, :]
        odd = covariates[rev_inds, :, :]
        even = even.reshape((even.shape[0], -1))
        odd = odd.reshape((odd.shape[0], -1))
        metric = Divergence(even, odd, method)
        result = metric.evaluate()
        assert result == output


class TestDivergenceOutput:
    def test_divergenceoutput_eq(self):
        assert DivergenceOutput(1.0, 1.0) == DivergenceOutput(1.0, 1.0)

    def test_divergenceoutput_ne(self):
        assert DivergenceOutput(1.0, 1.0) != DivergenceOutput(0.9, 0.9)

    def test_divergenceoutput_ne_type(self):
        assert DivergenceOutput(1.0, 1.0) != (1.0, 1.0)
