import numpy as np
import pytest

from daml.datasets import DamlDataset
from daml.metrics.divergence import HP_FNN, HP_MST, DivergenceOutput

np.random.seed(0)


class TestDpDivergence:
    @pytest.mark.parametrize(
        "metric_class, output",
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
    def test_dp_divergence(self, metric_class, output):
        """Unit testing of Dp Divergence

        TBD
        """
        path = "tests/datasets/mnist.npz"
        with np.load(path, allow_pickle=True) as fp:
            covariates, labels = fp["x_train"][:1000], fp["y_train"][:1000]

        inds = np.array([x % 2 == 0 for x in labels])
        rev_inds = np.invert(inds)
        even = covariates[inds, :, :]
        odd = covariates[rev_inds, :, :]
        even = even.reshape((even.shape[0], -1))
        odd = odd.reshape((odd.shape[0], -1))
        metric = metric_class()
        dataset_a = DamlDataset(even)
        dataset_b = DamlDataset(odd)
        result = metric.evaluate(dataset_a=dataset_a, dataset_b=dataset_b)
        assert result == output


@pytest.mark.skip
class TestEncodedDpDivergence:
    @pytest.mark.parametrize(
        "metric_class, output",
        [
            pytest.param(
                HP_MST,
                DivergenceOutput(
                    dpdivergence=0.8,
                    error=10,
                ),
                # marks=pytest.mark.functional,
            ),
            (
                HP_FNN,
                DivergenceOutput(
                    dpdivergence=0.8,
                    error=10,
                ),
            ),
        ],
    )
    def test_ae_new_label(self, metric_class, output):
        """
        Checks the dp divergence between trained labels and a new distribution

        WIP Does not work, do not run
        """
        # Separates mnist data into even and odd indices as two datasets
        path = "tests/datasets/mnist.npz"
        with np.load(path, allow_pickle=True) as fp:
            covariates, labels = fp["x_train"][:3000], fp["y_train"][:3000]
        # Take two dissimilar numbers
        ones = covariates[labels == 1]
        fours = covariates[labels == 4]
        train = np.concatenate([ones, fours])
        np.random.shuffle(train)
        test = covariates[labels == 8]

        # Add new axis to make NHWC from NHW
        # train = train[..., np.newaxis]
        train = np.stack([train, train, train, train, train], axis=-1)
        # test = test[..., np.newaxis]
        test = np.stack([test, test, test, test, test], axis=-1)
        print(train.shape, test.shape)
        dataset_a = DamlDataset(train)
        dataset_b = DamlDataset(test)
        # Initialize the autoencoder-based outlier detector from alibi-detect
        metric = metric_class(encode=True)
        metric.fit_dataset(dataset_a, epochs=25)
        # Evaluate using model embeddings
        result = metric.evaluate(dataset_a, dataset_b)
        assert result.dpdivergence >= output.dpdivergence
        assert result.error <= output.error

    @pytest.mark.parametrize(
        "metric_class, output",
        [
            (HP_MST, DivergenceOutput(dpdivergence=0.8, error=10)),
            (HP_FNN, DivergenceOutput(dpdivergence=0.8, error=10)),
        ],
    )
    def test_ae_interclass(self, metric_class, output):
        """This test fails, do not run"""
        path = "tests/datasets/mnist.npz"
        with np.load(path, allow_pickle=True) as fp:
            covariates, labels = fp["x_train"][:3000], fp["y_train"][:3000]
        # Take two dissimilar numbers
        ones = covariates[labels == 1]
        eights = covariates[labels == 8]
        training = np.concatenate([ones, eights])
        np.random.shuffle(training)
        # Add new axis to make NHWC from NHW
        ones = ones[..., np.newaxis]
        eights = eights[..., np.newaxis]
        training = training[..., np.newaxis]
        # Convert to DamlDataset
        dataset_a = DamlDataset(ones)
        dataset_b = DamlDataset(eights)
        train_dataset = DamlDataset(training)
        # Initialize the autoencoder-based outlier detector from alibi-detect
        metric = metric_class(encode=True)
        metric.fit_dataset(train_dataset, epochs=25)
        # Evaluate using model embeddings
        result = metric.evaluate(dataset_a, dataset_b)
        assert result.dpdivergence >= output.dpdivergence
        assert result.error <= output.error


class TestDivergenceOutput:
    def test_divergenceoutput_eq(self):
        assert DivergenceOutput(1.0, 1.0) == DivergenceOutput(1.0, 1.0)

    def test_divergenceoutput_ne(self):
        assert DivergenceOutput(1.0, 1.0) != DivergenceOutput(0.9, 0.9)

    def test_divergenceoutput_ne_type(self):
        assert DivergenceOutput(1.0, 1.0) != (1.0, 1.0)
