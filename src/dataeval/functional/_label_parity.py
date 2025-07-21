from __future__ import annotations

__all__ = []

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chisquare

from dataeval.typing import ArrayLike
from dataeval.utils._array import as_numpy


def _normalize_expected_dist(expected_dist: NDArray[Any], observed_dist: NDArray[Any]) -> NDArray[Any]:
    """
    Normalize the expected label distribution to match the total number of labels in the observed distribution.

    This function adjusts the expected distribution so that its sum equals the sum of the observed distribution.
    If the expected distribution is all zeros, an error is raised.

    Parameters
    ----------
    expected_dist : NDArray
        The expected label distribution. This array represents the anticipated distribution of labels.
    observed_dist : NDArray
        The observed label distribution. This array represents the actual distribution of labels in the dataset.

    Returns
    -------
    NDArray
        The normalized expected distribution, scaled to have the same sum as the observed distribution.

    Raises
    ------
    ValueError
        If the expected distribution is all zeros.

    Note
    ----
    The function ensures that the total number of labels in the expected distribution matches the total
    number of labels in the observed distribution by scaling the expected distribution.
    """

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


def _validate_dist(label_dist: NDArray[Any], label_name: str) -> None:
    """
    Verifies that the given label distribution has labels and checks if
    any labels have frequencies less than 5.

    Parameters
    ----------
    label_dist : NDArray
        Array representing label distributions
    label_name : str
        String representing label name

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
            f"Labels {np.where(label_dist < 5)[0]} in {label_name}"
            " dataset have frequencies less than 5. This may lead"
            " to invalid chi-squared evaluation.",
            UserWarning,
        )


def label_parity(
    expected_labels: ArrayLike,
    observed_labels: ArrayLike,
    *,
    num_classes: int | None = None,
) -> tuple[float, float]:
    """
    Calculate the chi-square statistic to assess the :term:`parity<Parity>` \
    between expected and observed label distributions.

    This function computes the frequency distribution of classes in both expected and observed labels, normalizes
    the expected distribution to match the total number of observed labels, and then calculates the chi-square
    statistic to determine if there is a significant difference between the two distributions.

    Parameters
    ----------
    expected_labels : ArrayLike
        List of class labels in the expected dataset
    observed_labels : ArrayLike
        List of class labels in the observed dataset
    num_classes : int or None, default None
        The number of unique classes in the datasets. If not provided, the function will infer it
        from the set of unique labels in expected_labels and observed_labels

    Returns
    -------
    LabelParityOutput
        chi-squared score and :term`P-Value` of the test

    Raises
    ------
    ValueError
        If expected label distribution is empty, is all zeros, or if there is a mismatch in the number
        of unique classes between the observed and expected distributions.


    Note
    ----
    - Providing ``num_classes`` can be helpful if there are classes with zero instances in one of the distributions.
    - The function first validates the observed distribution and normalizes the expected distribution so that it
      has the same total number of labels as the observed distribution.
    - It then performs a :term:`Chi-Square Test of Independence` to determine if there is a statistically significant
      difference between the observed and expected label distributions.
    - This function acts as an interface to the scipy.stats.chisquare method, which is documented at
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html


    Examples
    --------
    Randomly creating some label distributions using ``np.random.default_rng``

    >>> rng = np.random.default_rng(175)
    >>> expected_labels = rng.choice([0, 1, 2, 3, 4], (100))
    >>> observed_labels = rng.choice([2, 3, 0, 4, 1], (100))
    >>> label_parity(expected_labels, observed_labels)
    (14.007374204742625, 0.0072715574616218)
    """

    # Calculate
    if not num_classes:
        num_classes = 0

    # Calculate the class frequencies associated with the datasets
    observed_dist = np.bincount(as_numpy(observed_labels), minlength=num_classes)
    expected_dist = np.bincount(as_numpy(expected_labels), minlength=num_classes)

    # Validate
    _validate_dist(observed_dist, "observed")

    # Normalize
    expected_dist = _normalize_expected_dist(expected_dist, observed_dist)

    # Validate normalized expected distribution
    _validate_dist(expected_dist, f"expected for {np.sum(observed_dist)} observations")

    if len(observed_dist) != len(expected_dist):
        raise ValueError(
            f"Found {len(observed_dist)} unique classes in observed label distribution, "
            f"but found {len(expected_dist)} unique classes in expected label distribution. "
            "This can happen when some class ids have zero instances in one dataset but "
            "not in the other. When initializing Parity, try setting the num_classes "
            "parameter to the known number of unique class ids, so that classes with "
            "zero instances are still included in the distributions."
        )

    cs, p = chisquare(f_obs=observed_dist, f_exp=expected_dist)
    return cs, p
