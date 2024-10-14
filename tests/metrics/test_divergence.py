import numpy as np
import pytest

from dataeval._internal.metrics.divergence import divergence, divergence_fnn, divergence_mst
from tests.conftest import mnist, skip_mnist

np.random.seed(0)


class TestDivergence:
    @skip_mnist
    @pytest.mark.parametrize(
        "method, output",
        [
            ("MST", {"divergence": 0.838, "errors": 81}),
            ("FNN", {"divergence": 0.864, "errors": 68}),
        ],
    )
    def test_divergence(self, method, output):
        """Unit testing of Divergence"""

        covariates, labels = mnist(channels="channels_last")

        inds = np.array([x % 2 == 0 for x in labels])
        rev_inds = np.invert(inds)
        even = covariates[inds, :, :]
        odd = covariates[rev_inds, :, :]
        even = even.reshape((even.shape[0], -1))
        odd = odd.reshape((odd.shape[0], -1))
        result = divergence(even, odd, method)
        assert result.dict() == output

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
