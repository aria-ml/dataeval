import numpy as np
import pytest

from daml.datasets import DamlDataset
from daml.metrics.divergence import HP_FNN, HP_MST, DivergenceOutput


class TestDpDivergence:
    @pytest.mark.parametrize(
        "input, output",
        [
            (
                HP_MST,
                DivergenceOutput(
                    dpdivergence=0.8377897755491117,
                    error=81.0,
                ),
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
        dataset_a = DamlDataset(even)
        dataset_b = DamlDataset(odd)
        result = metric.evaluate(dataset_a=dataset_a, dataset_b=dataset_b)
        assert result == output


class TestEncodedDpDivergence:
    @pytest.mark.parametrize(
        "input, output",
        [
            (
                HP_MST,
                DivergenceOutput(
                    dpdivergence=0.96875,
                    error=1,
                ),
            ),
            (
                HP_FNN,
                DivergenceOutput(
                    dpdivergence=1.0,
                    error=0.0,
                ),
            ),
        ],
    )
    def test_dpd_with_autoencoder(self, input, output):
        # Initialize a dataset of 32 images of size 32x32x3, containing all 1's
        dataset_a = DamlDataset(np.ones(shape=(32, 32)) + np.identity(32))
        dataset_b = DamlDataset(13 * np.ones(shape=(32, 32)) + np.identity(32))

        # Initialize the autoencoder-based outlier detector from alibi-detect
        metric = input(encode=True)
        metric.fit_dataset(dataset_a, epochs=10)
        result = metric.evaluate(dataset_a, dataset_b)
        assert result == output
