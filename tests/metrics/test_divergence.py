import numpy as np
import pytest

from daml.metrics import Divergence

np.random.seed(0)


class TestDpDivergence:
    @pytest.mark.parametrize(
        "method, output",
        [
            ("MST", {"dpdivergence": 0.8377897755491117, "error": 81}),
            ("FNN", {"dpdivergence": 0.8618209199122062, "error": 69}),
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
