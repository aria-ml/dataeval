import warnings

import numpy as np
import pytest

from daml._internal.metrics.parity import Parity


class MockDistributionDataset:
    """
    Mock dataset with labels that obey a label distribution supplied at __init__
    """

    # TODO: move defs to init

    # TODO: move defs to init
    def __init__(self, label_dist):
        for label_curr in label_dist:
            if not isinstance(label_curr, (int, np.integer)):
                raise Exception(f"Expected integer in the distribution of labels, got \
                                {label_curr} with type {type(label_curr)}")

        self.image = np.array([0, 0, 0])
        self.image = np.array([0, 0, 0])
        self.length = np.sum(label_dist)
        self.labels = np.zeros(self.length, dtype=int)

        idx = 0
        for label, label_interval in enumerate(label_dist):
            for j in range(label_interval):
                self.labels[idx] = label
                idx += 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (self.image, self.labels[idx])


class TestLabelIndependenceUnit:
    def test_fails_with_imbalanced_nclasses(self):
        f_exp = [1]
        f_obs = [0, 1]
        f_obs = [0, 1]

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with pytest.raises(Exception), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Parity(labels_expected, labels_observed)

    def test_fails_with_unaccounted_for_zero(self):
        f_exp = [1, 0]
        f_obs = [0, 1]
        f_exp = [1, 0]
        f_obs = [0, 1]

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with pytest.raises(Exception), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Parity(labels_expected, labels_observed)

    def test_warns_with_not_enough_frequency(self):
        f_exp = [1, 1]
        f_obs = [1, 4]
        f_exp = [1, 1]
        f_obs = [1, 4]

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with pytest.warns():
            Parity(labels_expected, labels_observed)

    def test_warns_with_not_enough_frequency_rescaled_exp(self):
        f_exp = [10, 10000]
        f_obs = [100, 400]

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with pytest.warns():
            Parity(labels_expected, labels_observed)

    def test_passes_with_enough_frequency(self):
        f_exp = [10, 10]
        f_obs = [10, 40]
        f_exp = [10, 10]
        f_obs = [10, 40]

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Parity(labels_expected, labels_observed)

    def test_passes_with_ncls(self):
        f_exp = [1]
        f_obs = [0, 1]
        f_obs = [0, 1]

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Parity(labels_expected, labels_observed, num_classes=2)

    def test_fails_with_empty_exp_dataset(self):
        f_exp = np.array([], dtype=int)
        f_obs = [0, 1]
        f_exp = np.array([], dtype=int)
        f_obs = [0, 1]

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with pytest.raises(ValueError), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Parity(labels_expected, labels_observed)

    def test_fails_with_empty_obs_dataset(self):
        f_exp = [0, 1]
        f_obs = np.array([], dtype=int)
        f_exp = [0, 1]
        f_obs = np.array([], dtype=int)

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with pytest.raises(ValueError), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Parity(labels_expected, labels_observed)


class TestLabelIndependenceFunctional:
    def test_scipy_example_data(self):
        """
        Verify that using daml tools, we can reconstruct the example at:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html
        """

        f_exp = np.array([44, 24, 29, 3])  # / 100 * 189
        f_exp = np.array([44, 24, 29, 3])  # / 100 * 189
        f_obs = np.array([43, 52, 54, 40])

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        lsi = Parity(labels_expected, labels_observed)

        chisquared, p = lsi.evaluate()

        assert np.isclose(chisquared, 228.23515947653874)
        assert np.isclose(p, 3.3295585338846486e-49)

    def test_5050_data(self):
        """
        Test that a 50/50 distribution of labels gives the expected chi-squared and p-value
        """
        f_exp = np.array([100, 100])
        f_obs = np.array([100, 100])
        f_exp = np.array([100, 100])
        f_obs = np.array([100, 100])

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        lsi = Parity(labels_expected, labels_observed)

        chisquared, p = lsi.evaluate()

        assert np.isclose(chisquared, 0)
        assert np.isclose(p, 1)
        assert np.isclose(p, 1)

        assert np.isclose(p, 1)
