import warnings
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
from scipy.stats import chisquare

from dataeval._internal.interop import ArrayLike, to_numpy
from dataeval._internal.metrics.base import EvaluateMixin
from dataeval._internal.metrics.functional import format_discretize_factors, normalize_expected_dist, parity


def validate_dist(label_dist: np.ndarray, label_name: str):
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


class Parity(EvaluateMixin):
    """
    Class for evaluating statistics of observed and expected class labels, including:

    - Chi Squared test for statistical independence between expected and observed labels
    """

    def evaluate(
        self,
        expected_labels: ArrayLike,
        observed_labels: ArrayLike,
        num_classes: Optional[int] = None,
    ) -> Tuple[np.float64, np.float64]:
        """
        Perform a one-way chi-squared test between observation frequencies and expected frequencies that
        tests the null hypothesis that the observed data has the expected frequencies.

        This function acts as an interface to the scipy.stats.chisquare method, which is documented at
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html

        Parameters
        ----------
        expected_labels : ArrayLike
            List of class labels in the expected dataset
        observed_labels : ArrayLike
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

        # Calculate the class frequencies associated with the datasets
        observed_dist = np.bincount(to_numpy(observed_labels), minlength=num_classes)
        expected_dist = np.bincount(to_numpy(expected_labels), minlength=num_classes)

        # Validate
        validate_dist(observed_dist, "observed")

        # Normalize
        expected_dist = normalize_expected_dist(expected_dist, observed_dist)

        # Validate normalized expected distribution
        validate_dist(expected_dist, f"expected for {np.sum(observed_dist)} observations")

        if len(observed_dist) != len(expected_dist):
            raise ValueError(
                f"Found {len(observed_dist)} unique classes in observed label distribution, "
                f"but found {len(expected_dist)} unique classes in expected label distribution. "
                "This can happen when some class ids have zero instances in one dataset but "
                "not in the other. When initializing Parity, try setting the num_classes "
                "parameter to the known number of unique class ids, so that classes with "
                "zero instances are still included in the distributions."
            )

        chisquared, p_value = chisquare(f_obs=observed_dist, f_exp=expected_dist)
        return chisquared, p_value


class MetadataParity(EvaluateMixin):
    def evaluate(
        self,
        data_factors: Mapping[str, ArrayLike],
        continuous_factor_bincounts: Optional[Dict[str, int]] = None,
    ) -> dict[str, np.ndarray]:
        """
        Evaluates the statistical independence of metadata factors from class labels.
        This performs a chi-square test, which provides a score and a p-value for
        statistical independence between each pair of a metadata factor and a class label.
        A high score with a low p-value suggests that a metadata factor is strongly
        correlated with a class label.

        Parameters
        ----------
        data_factors: Mapping[str, ArrayLike]
            The dataset factors, which are per-image attributes including class label and metadata.
            Each key of dataset_factors is a factor, whose value is the per-image factor values.
        continuous_factor_bincounts : Optional[Dict[str, int]], default None
            The factors in data_factors that have continuous values and the array of bin counts to
            discretize values into. All factors are treated as having discrete values unless they
            are specified as keys in this dictionary. Each element of this array must occur as a key
            in data_factors.

        Returns
        -------
        Dict[str, np.ndarray]
            chi_square: np.ndarray
                Array of length (num_factors) whose (i)th element corresponds to
                the chi-square score for the relationship between factor i
                and the class labels in the dataset.
            p_values: np.ndarray
                Array of length (num_factors) whose (i)th element corresponds to
                the p-value for the chi-square test for the relationship between
                factor i and the class labels in the dataset.
        """
        data_factors_np = {k: to_numpy(v) for k, v in data_factors.items()}
        continuous_factor_bincounts = continuous_factor_bincounts if continuous_factor_bincounts else {}

        metadata_factors, labels = format_discretize_factors(data_factors_np, continuous_factor_bincounts)
        chi_square, p_values = parity(metadata_factors, labels)

        return {"chi_squares": chi_square, "p_values": p_values}
