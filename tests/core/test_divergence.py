import numpy as np
import pytest

from dataeval.core._divergence import (
    _compute_fnn_errors,
    _compute_mst_errors,
    divergence_fnn,
    divergence_mst,
)

np.random.seed(0)


@pytest.mark.required
class TestDivergenceCore:
    """Tests the core divergence calculation functions."""

    @pytest.mark.parametrize(
        ("method", "expected_errors"),
        [
            (_compute_mst_errors, 9),  # all 9 edges of MST will connect different labels
            (_compute_fnn_errors, 10),  # FNN gets every sample in this case.
        ],
    )
    def test_divergence_error_funcs(self, method, expected_errors):
        """Test error calculation funcs give deterministic error outputs."""
        images = np.ones((10, 3, 3))
        labels = np.arange(10)

        assert method(images, labels) == expected_errors

    @pytest.mark.parametrize(
        "method",
        [
            _compute_mst_errors,
            _compute_fnn_errors,
        ],
    )
    def test_error_func_flatten(self, method):
        """3x3 is equal to 9x1 when flattened, so errors should be equal as well."""
        images = np.random.random(size=(10, 3, 3))
        labels = np.arange(10)

        assert method(images, labels) == method(images.reshape((10, -1)), labels)

    @pytest.mark.parametrize(
        ("method", "output"),
        [
            (divergence_mst, {"divergence": 0.9899996399870395, "errors": 5}),
            (divergence_fnn, {"divergence": 1.0, "errors": 0}),
        ],
    )
    @pytest.mark.optional
    def test_divergence_mock_data(self, method, output):
        """Unit testing of Divergence with mock data."""
        rng = np.random.default_rng(3)
        labels = np.concatenate([rng.choice(10, 500), np.arange(10).repeat(50)])
        covariates = np.ones((1000, 28, 28)) * labels[:, np.newaxis, np.newaxis]
        covariates[:, 13:16, 13:16] += 1
        covariates[-200:, 13:16, 13:16] += rng.choice(5)
        covariates /= covariates.max()
        inds = np.array([x % 2 == 0 for x in labels])
        rev_inds = np.invert(inds)
        even = covariates[inds, :, :]
        odd = covariates[rev_inds, :, :]
        even = even.reshape((even.shape[0], -1))
        odd = odd.reshape((odd.shape[0], -1))
        result = method(even, odd)
        assert result == output
