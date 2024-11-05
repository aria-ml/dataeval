import numpy as np
import pytest

from dataeval.metrics.estimators.divergence import divergence, divergence_fnn, divergence_mst

np.random.seed(0)


class TestDivergence:
    @pytest.mark.parametrize(
        "method, output",
        [
            ("MST", {"divergence": 0.9819993519766712, "errors": 9}),
            ("FNN", {"divergence": 1.0, "errors": 0.0}),
        ],
    )
    def test_divergence_mock_data(self, method, output):
        """Unit testing of Divergence"""
        rng = np.random.default_rng(3)
        labels = np.concatenate([rng.choice(10, 500), np.arange(10).repeat(50)])
        covariates = np.ones((1000, 28, 28)) * labels[:, np.newaxis, np.newaxis]
        covariates[:, 13:16, 13:16] += 1
        covariates[-200:, 13:16, 13:16] += rng.choice(5)
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
