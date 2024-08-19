import warnings
from typing import Optional, Tuple

import numpy as np
import scipy


class Parity:
    """
    Class for evaluating statistics of observed and expected class labels, including:

    - Chi Squared test for statistical independence between expected and observed labels
    """

    def _normalize_expected_dist(self, expected_dist: np.ndarray, observed_dist: np.ndarray) -> np.ndarray:
        exp_sum = np.sum(expected_dist)
        obs_sum = np.sum(observed_dist)

        if exp_sum == 0:
            raise ValueError(
                f"Expected label distribution {expected_dist} is all zeros. "
                "Ensure that Parity.expected_dist is set to a list "
                "with at least one nonzero element"
            )

        # Renormalize expected distribution to have the same total number of labels as the observed dataset
        if exp_sum != obs_sum:
            expected_dist = expected_dist * obs_sum / exp_sum

        return expected_dist

    def _calculate_label_dist(self, labels: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Calculate the class frequencies associated with a dataset

        Parameters
        ----------
        labels : np.ndarray
            List of class labels in a dataset
        num_classes: int
            The number of unique classes in the datasets

        Returns
        -------
        label_dist : np.ndarray
            Array representing label distributions
        """
        label_dist = np.bincount(labels, minlength=num_classes)
        return label_dist

    def _validate_class_balance(self, expected_dist: np.ndarray, observed_dist: np.ndarray):
        """
        Check if the numbers of unique classes in the datasets are unequal

        Parameters
        ----------
        expected_dist : np.ndarray
            Array representing expected label distributions
        observed_dist : np.ndarray
            Array representing observed label distributions

        Raises
        ------
        ValueError
            When exp_ld and obs_ld do not have the same number of classes
        """
        exp_n_cls = len(expected_dist)
        obs_n_cls = len(observed_dist)
        if exp_n_cls != obs_n_cls:
            raise ValueError(
                f"Found {obs_n_cls} unique classes in observed label distribution, "
                f"but found {exp_n_cls} unique classes in expected label distribution,"
                "This can happen when some class ids have zero instances in one dataset but "
                "not in the other. When initializing Parity, "
                "try setting the num_classes parameter to the known number of unique class ids, "
                "so that classes with zero instances are still included in the distributions."
            )

    def _validate_dist(self, label_dist: np.ndarray, label_name: str):
        """
        Verifies that the given label distribution has labels and checks if
        any labels have frequencies less than 5.

        Parameters
        ----------
        label_dist : np.ndarray
            Array representing label distributions

        Raises
        ------
        ValueError
            If label_dist is empty
        Warning
            If any elements of label_dist are less than 5
        """
        if not len(label_dist):
            raise ValueError(f"No labels found in the {label_name} dataset")
        if np.any(label_dist < 5):
            warnings.warn(
                f"Labels {np.where(label_dist<5)[0]} in {label_name}"
                " dataset have frequencies less than 5. This may lead"
                " to invalid chi-squared evaluation."
            )
            warnings.warn(
                f"Labels {np.where(label_dist<5)[0]} in {label_name}"
                " dataset have frequencies less than 5. This may lead"
                " to invalid chi-squared evaluation."
            )

    def evaluate(
        self, expected_labels: np.ndarray, observed_labels: np.ndarray, num_classes: Optional[int] = None
    ) -> Tuple[np.float64, np.float64]:
        """
        Perform a one-way chi-squared test between observation frequencies and expected frequencies that
        tests the null hypothesis that the observed data has the expected frequencies.

        This function acts as an interface to the scipy.stats.chisquare method, which is documented at
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html

        Parameters
        ----------
        expected_labels : np.ndarray
            List of class labels in the expected dataset
        observed_labels : np.ndarray
            List of class labels in the observed dataset
        num_classes : Optional[int]
            The number of unique classes in the datasets. If this is not specified, it will
            be inferred from the set of unique labels in expected_labels and observed_labels

        Returns
        -------
        np.float64
            chi-squared value of the test
        np.float64
            p-value of the test

        Raises
        ------
        ValueError
            If x is empty
        """
        # Calculate
        if not num_classes:
            num_classes = 0

        observed_dist = self._calculate_label_dist(observed_labels, num_classes)
        expected_dist = self._calculate_label_dist(expected_labels, num_classes)

        # Validate
        self._validate_dist(observed_dist, "observed")

        # Normalize
        expected_dist = self._normalize_expected_dist(expected_dist, observed_dist)

        # Validate normalized expected distribution
        self._validate_dist(expected_dist, f"expected for {np.sum(observed_dist)} observations")
        self._validate_class_balance(expected_dist, observed_dist)

        cs_result = scipy.stats.chisquare(f_obs=observed_dist, f_exp=expected_dist)

        chisquared = cs_result.statistic
        p_value = cs_result.pvalue
        return chisquared, p_value
