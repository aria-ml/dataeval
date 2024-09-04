import numpy as np
import pytest

from dataeval._internal.metrics.divergence import divergence_fnn, divergence_mst
from dataeval.metrics import divergence

np.random.seed(0)


class TestDivergence:
    @pytest.mark.parametrize(
        "method, output",
        [
            ("MST", {"divergence": 0.8377897755491117, "errors": 81}),
            ("FNN", {"divergence": 0.8618209199122062, "errors": 69}),
        ],
    )
    def test_divergence(self, mnist, method, output):
        """Unit testing of Divergence"""

        covariates, labels = mnist(add_channels="channels_last")

        inds = np.array([x % 2 == 0 for x in labels])
        rev_inds = np.invert(inds)
        even = covariates[inds, :, :]
        odd = covariates[rev_inds, :, :]
        even = even.reshape((even.shape[0], -1))
        odd = odd.reshape((odd.shape[0], -1))
        result = divergence(even, odd, method)
        assert result._asdict() == output

    @pytest.mark.parametrize(
        "method, expected_errors",
        [
            (divergence_mst, 9),
            (divergence_fnn, 45),
        ],
    )
    def test_divergence_funcs(self, method, expected_errors):
        """Test math funcs give deterministic error outputs"""
        images = np.ones((10, 3, 3))
        labels = np.arange(10)

        assert method(images, labels) == expected_errors

    @pytest.mark.parametrize(
        "method",
        [
            divergence_mst,
            divergence_fnn,
        ],
    )
    def test_div_func_flatten(self, method):
        """3x3 is equal to 9x1 when flattened, so errors should be equal as well"""
        images = np.random.random(size=(10, 3, 3))
        labels = np.arange(10)

        assert method(images, labels) == method(images.reshape((10, -1)), labels)
