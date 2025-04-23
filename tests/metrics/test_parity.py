import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest

from dataeval.data._metadata import Metadata
from dataeval.metrics.bias._parity import label_parity, parity
from tests.conftest import preprocess


class MockDistributionDataset:
    """
    Mock dataset with labels that obey a label distribution supplied at __init__
    """

    # TODO: move defs to init
    def __init__(self, label_dist):
        for label_curr in label_dist:
            if not isinstance(label_curr, (int, np.integer)):
                raise Exception(
                    f"Expected integer in the distribution of labels, got \
                                {label_curr} with type {type(label_curr)}"
                )

        self.image = np.array([0, 0, 0])
        self.image = np.array([0, 0, 0])
        self.length = np.sum(label_dist)
        self.labels = np.zeros(self.length, dtype=np.intp)

        idx = 0
        for label, label_interval in enumerate(label_dist):
            for j in range(label_interval):
                self.labels[idx] = label
                idx += 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (self.image, self.labels[idx])


@pytest.mark.required
class TestLabelIndependenceUnit:
    def test_fails_with_imbalanced_nclasses(self):
        f_exp = [1]
        f_obs = [0, 1]
        f_obs = [0, 1]

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with pytest.raises(Exception), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            label_parity(labels_expected, labels_observed)

    def test_fails_with_unaccounted_for_zero(self):
        f_exp = [1, 0]
        f_obs = [0, 1]
        f_exp = [1, 0]
        f_obs = [0, 1]

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with pytest.raises(Exception), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            label_parity(labels_expected, labels_observed)

    def test_warns_with_not_enough_frequency(self):
        f_exp = [1, 1]
        f_obs = [1, 4]
        f_exp = [1, 1]
        f_obs = [1, 4]

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with pytest.warns():
            label_parity(labels_expected, labels_observed)

    def test_warns_with_not_enough_frequency_rescaled_exp(self):
        f_exp = [10, 10000]
        f_obs = [100, 400]

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with pytest.warns():
            label_parity(labels_expected, labels_observed)

    def test_passes_with_enough_frequency(self):
        f_exp = [10, 10]
        f_obs = [10, 40]
        f_exp = [10, 10]
        f_obs = [10, 40]

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            label_parity(labels_expected, labels_observed)

    def test_passes_with_ncls(self):
        f_exp = [1]
        f_obs = [0, 1]
        f_obs = [0, 1]

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            label_parity(labels_expected, labels_observed, num_classes=2)

    def test_fails_with_empty_exp_dataset(self):
        f_exp = np.array([], dtype=np.intp)
        f_obs = [0, 1]
        f_exp = np.array([], dtype=np.intp)
        f_obs = [0, 1]

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with pytest.raises(ValueError), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            label_parity(labels_expected, labels_observed)

    def test_fails_with_empty_obs_dataset(self):
        f_exp = [0, 1]
        f_obs = np.array([], dtype=np.intp)
        f_exp = [0, 1]
        f_obs = np.array([], dtype=np.intp)

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        with pytest.raises(ValueError), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            label_parity(labels_expected, labels_observed)

    def test_empty_metadata(self):
        mock_metadata = MagicMock(spec=Metadata)
        mock_metadata.discrete_factor_names = []
        mock_metadata.continuous_factor_names = []
        with pytest.raises(ValueError):
            parity(mock_metadata)


@pytest.mark.optional
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

        result = label_parity(labels_expected, labels_observed)

        assert np.isclose(result.score, 228.23515947653874)
        assert np.isclose(result.p_value, 3.3295585338846486e-49)

    def test_5050_data(self):
        """
        Test that a 50/50 distribution of labels gives the expected chi-squared and :term:`p-value<P-Value>`
        """
        f_exp = np.array([100, 100])
        f_obs = np.array([100, 100])
        f_exp = np.array([100, 100])
        f_obs = np.array([100, 100])

        labels_expected = MockDistributionDataset(f_exp).labels
        labels_observed = MockDistributionDataset(f_obs).labels

        result = label_parity(labels_expected, labels_observed)

        assert np.isclose(result.score, 0)
        assert np.isclose(result.p_value, 1)


class TestMDParityUnit:
    def test_warns_with_not_enough_frequency(self):
        labels = [0, 1]
        factors = {"factor1": [10, 20]}
        metadata = preprocess(factors, labels)
        with pytest.warns(UserWarning):
            parity(metadata)

    def test_passes_with_enough_frequency(self):
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["foo"] * 10}
        metadata = preprocess(factors, labels)
        parity(metadata)

    @pytest.mark.requires_all
    def test_to_dataframe(self):
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["foo"] * 10}
        metadata = preprocess(factors, labels)
        df = parity(metadata).to_dataframe()
        assert df is not None


class TestMDParityFunctional:
    def test_correlated_factors(self):
        """
        In this dataset, class and factor1 are perfectly correlated.
        This tests that the p-value<P-Value>` is less than 0.05, which
        corresponds to class and factor1 being highly correlated.
        """
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["a"] * 5 + ["b"] * 5}
        metadata = preprocess(factors, labels)
        result = parity(metadata)

        # Checks that factor1 is highly correlated with class
        assert result.p_value[0] < 0.05

    def test_uncorrelated_factors(self):
        """
        This verifies that if the factor is homogeneous for the whole dataset,
        that chi2 and p correspond to factor1 being uncorrelated with class.
        """
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": ["foo"] * 10}
        metadata = preprocess(factors, labels)
        result = parity(metadata)

        # Checks that factor1 is uncorrelated with class
        assert np.isclose(result.score[0], 0)
        assert np.isclose(result.p_value[0], 1)

    def test_quantized_factors(self):
        """
        This discretizes 'factor1' into having two values.
        This verifies that the '11' and '10' values get grouped together.
        """
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": [10] * 2 + [11] * 3 + [20] * 5}
        continuous_bincounts = {"factor1": 2}
        metadata = preprocess(factors, labels, continuous_bincounts)
        result1 = parity(metadata)

        discrete_dataset = {"factor2": [10] * 5 + [20] * 5}
        metadata = preprocess(discrete_dataset, labels)
        result2 = parity(metadata)

        # Checks that the test on the quantization continuous_dataset is
        # equivalent to the test on the discrete dataset discrete_dataset
        assert result1.score[0] == result2.score[0]
        assert result1.p_value[0] == result2.p_value[0]

    def test_overquantized_factors(self):
        """
        This quantizes factor1 to have only one value, so that the discretized
        factor1 is the same over the entire dataset.
        """
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": [10] * 2 + [11] * 3 + [20] * 5}
        continuous_bincounts = {"factor1": 1}
        metadata = preprocess(factors, labels, continuous_bincounts)
        result = parity(metadata)

        # Checks if factor1 and class are perfectly uncorrelated
        assert np.isclose(result.score[0], 0)
        assert np.isclose(result.p_value[0], 1)

    def test_underquantized_has_low_freqs(self):
        """
        This quantizes factor1 such that there are large regions with bins
        that contain a small number of points.
        """
        labels = [0] * 5 + [1] * 5
        factors = {"factor1": list(np.arange(10))}
        continuous_bincounts = {"factor1": 10}
        metadata = preprocess(factors, labels, continuous_bincounts)

        # Looks for a warning that there are (class,factor1) pairs with too low frequency
        with pytest.warns(UserWarning):
            parity(metadata)

    def test_underquantized_has_repeated_low_freqs(self):
        """
        This quantizes factor1 such that there are large regions with bins
        that contain a small number of points.
        """
        labels = [0] * 5 + [1] * 5 + [0] * 5 + [1] * 5
        factors = {"factor1": list(np.arange(10)) + list(np.arange(10))}
        continuous_bincounts = {"factor1": 10}
        metadata = preprocess(factors, labels, continuous_bincounts)

        # Looks for a warning that there are (class,factor1) pairs with too low frequency
        with pytest.warns(UserWarning):
            parity(metadata)
