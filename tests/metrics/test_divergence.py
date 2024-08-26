import numpy as np
import pytest

from dataeval.metrics import divergence

np.random.seed(0)


class TestDivergence:
    @pytest.mark.parametrize(
        "method, output",
        [
            ("MST", {"divergence": 0.8377897755491117, "error": 81}),
            ("FNN", {"divergence": 0.8618209199122062, "error": 69}),
        ],
    )
    def test_divergence(self, mnist, method, output):
        """Unit testing of Divergence

        TBD
        """

        covariates, labels = mnist(add_channels="channels_last")

        inds = np.array([x % 2 == 0 for x in labels])
        rev_inds = np.invert(inds)
        even = covariates[inds, :, :]
        odd = covariates[rev_inds, :, :]
        even = even.reshape((even.shape[0], -1))
        odd = odd.reshape((odd.shape[0], -1))
        result = divergence(even, odd, method)
        assert result == output
