import warnings

import numpy as np
import pytest

from dataeval._internal.metrics.parity import parity, parity_metadata


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
            parity(labels_expected, labels_observed)

    def test_fails_with_unaccounted_for_zero(self):
        f_exp = [1, 0]
        f_obs = [0, 1]
        f_exp = [1, 0]
        f_obs = [0, 1]

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with pytest.raises(Exception), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parity(labels_expected, labels_observed)

    def test_warns_with_not_enough_frequency(self):
        f_exp = [1, 1]
        f_obs = [1, 4]
        f_exp = [1, 1]
        f_obs = [1, 4]

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with pytest.warns():
            parity(labels_expected, labels_observed)

    def test_warns_with_not_enough_frequency_rescaled_exp(self):
        f_exp = [10, 10000]
        f_obs = [100, 400]

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with pytest.warns():
            parity(labels_expected, labels_observed)

    def test_passes_with_enough_frequency(self):
        f_exp = [10, 10]
        f_obs = [10, 40]
        f_exp = [10, 10]
        f_obs = [10, 40]

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            parity(labels_expected, labels_observed)

    def test_passes_with_ncls(self):
        f_exp = [1]
        f_obs = [0, 1]
        f_obs = [0, 1]

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parity(labels_expected, labels_observed, num_classes=2)

    def test_fails_with_empty_exp_dataset(self):
        f_exp = np.array([], dtype=int)
        f_obs = [0, 1]
        f_exp = np.array([], dtype=int)
        f_obs = [0, 1]

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with pytest.raises(ValueError), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parity(labels_expected, labels_observed)

    def test_fails_with_empty_obs_dataset(self):
        f_exp = [0, 1]
        f_obs = np.array([], dtype=int)
        f_exp = [0, 1]
        f_obs = np.array([], dtype=int)

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with pytest.raises(ValueError), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parity(labels_expected, labels_observed)


class TestLabelIndependenceFunctional:
    def test_scipy_example_data(self):
        """
        Verify that using dataeval tools, we can reconstruct the example at:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html
        """

        f_exp = np.array([44, 24, 29, 3])  # / 100 * 189
        f_exp = np.array([44, 24, 29, 3])  # / 100 * 189
        f_obs = np.array([43, 52, 54, 40])

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        chisquared, p = parity(labels_expected, labels_observed)

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

        chisquared, p = parity(labels_expected, labels_observed)

        assert np.isclose(chisquared, 0)
        assert np.isclose(p, 1)
        assert np.isclose(p, 1)

        assert np.isclose(p, 1)


class TestMDParityUnit:
    def test_warns_with_not_enough_frequency(self):
        factors = {
            "class": np.array(
                [
                    0,
                    1,
                ]
            ),
            "factor1": np.array([10, 20]),
        }

        with pytest.warns():
            parity_metadata(factors)

    def test_passes_with_enough_frequency(self):
        factors = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.array(["foo"] * 10),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            parity_metadata(factors)

    def test_cant_quantize_strings(self):
        factors = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.concatenate((["a"] * 5, ["b"] * 5)),
        }
        continuous_bincounts = {"factor1": 2}

        with pytest.raises(TypeError):
            parity_metadata(factors, continuous_bincounts)

    def test_bad_factor_ref(self):
        factors = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.concatenate((["a"] * 5, ["b"] * 5)),
        }
        continuous_bincounts = {"something_else": 2}

        with pytest.raises(Exception):
            parity_metadata(factors, continuous_bincounts)

    def test_uneven_factor_lengths(self):
        factors = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.array(["a"] * 10),
            "factor2": np.array(["a"] * 11),
        }

        with pytest.raises(ValueError):
            parity_metadata(factors)


class TestMDParityFunctional:
    def test_correlated_factors(self):
        """
        In this dataset, class and factor1 are perfectly correlated.
        This tests that the p-value is less than 0.05, which
        corresponds to class and factor1 being highly correlated.
        """
        factors = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.concatenate(([10] * 5, [20] * 5)),
        }

        _, p = parity_metadata(factors)

        # Checks that factor1 is highly correlated with class
        assert p[0] < 0.05

    def test_uncorrelated_factors(self):
        """
        This verifies that if the factor is homogeneous for the whole dataset,
        that chi2 and p correspond to factor1 being uncorrelated with class.
        """
        factors = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.array(["foo"] * 10),
        }

        chi, p = parity_metadata(factors)

        # Checks that factor1 is uncorrelated with class
        assert np.isclose(chi[0], 0)
        assert np.isclose(p[0], 1)

    def test_quantized_factors(self):
        """
        This discretizes 'factor1' into having two values.
        This verifies that the '11' and '10' values get grouped together.
        """
        continuous_dataset = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.concatenate(([10] * 2, [11] * 3, [20] * 5)),
        }
        continuous_bincounts = {"factor1": 2}

        chi1, p1 = parity_metadata(continuous_dataset, continuous_bincounts)

        discrete_dataset = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor2": np.concatenate(([10] * 5, [20] * 5)),
        }

        chi2, p2 = parity_metadata(discrete_dataset)

        # Checks that the test on the quantization continuous_dataset is
        # equivalent to the test on the discrete dataset discrete_dataset
        assert p1[0] == p2[0]
        assert chi1[0] == chi2[0]

    def test_overquantized_factors(self):
        """
        This quantizes factor1 to have only one value, so that the discretized
        factor1 is the same over the entire dataset.
        """
        factors = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.concatenate(([10] * 5, [20] * 5)),
        }
        continuous_bincounts = {"factor1": 1}

        chi, p = parity_metadata(factors, continuous_bincounts)

        # Checks if factor1 and class are perfectly uncorrelated
        assert np.isclose(chi[0], 0)
        assert np.isclose(p[0], 1)

    def test_underquantized_has_low_freqs(self):
        """
        This quantizes factor1 such that there are large regions with bins
        that contain a small number of points.
        """
        factors = {
            "class": np.concatenate(([0] * 5, [1] * 5)),
            "factor1": np.concatenate(([10] * 4, [15], [20] * 5)),
        }
        continuous_bincounts = {"factor1": 100}

        # Looks for a warning that there are (class,factor1) pairs with too low frequency
        with pytest.warns():
            parity_metadata(factors, continuous_bincounts)
