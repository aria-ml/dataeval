from __future__ import annotations

__all__ = ["ParityOutput", "parity", "label_parity"]

import warnings
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chisquare
from scipy.stats.contingency import chi2_contingency, crosstab

from dataeval.interop import as_numpy, to_numpy
from dataeval.metrics.bias.metadata_preprocessing import MetadataOutput
from dataeval.output import Output, set_metadata

TData = TypeVar("TData", np.float64, NDArray[np.float64])


@dataclass(frozen=True)
class ParityOutput(Generic[TData], Output):
    """
    Output class for :func:`parity` and :func:`label_parity` :term:`bias<Bias>` metrics

    Attributes
    ----------
    score : np.float64 | NDArray[np.float64]
        chi-squared score(s) of the test
    p_value : np.float64 | NDArray[np.float64]
        p-value(s) of the test
    metadata_names : list[str] | None
        Names of each metadata factor
    """

    score: TData
    p_value: TData
    metadata_names: list[str] | None


def normalize_expected_dist(expected_dist: NDArray[Any], observed_dist: NDArray[Any]) -> NDArray[Any]:
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


def validate_dist(label_dist: NDArray[Any], label_name: str) -> None:
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
            f"Labels {np.where(label_dist<5)[0]} in {label_name}"
            " dataset have frequencies less than 5. This may lead"
            " to invalid chi-squared evaluation.",
            UserWarning,
        )


@set_metadata
def label_parity(
    expected_labels: ArrayLike,
    observed_labels: ArrayLike,
    num_classes: int | None = None,
) -> ParityOutput[np.float64]:
    """
    Calculate the chi-square statistic to assess the :term:`parity<Parity>` between expected and
    observed label distributions.

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
    ParityOutput[np.float64]
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

    >>> expected_labels = np_random_gen.choice([0, 1, 2, 3, 4], (100))
    >>> observed_labels = np_random_gen.choice([2, 3, 0, 4, 1], (100))
    >>> label_parity(expected_labels, observed_labels)
    ParityOutput(score=14.007374204742625, p_value=0.0072715574616218, metadata_names=None)
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

    cs, p = chisquare(f_obs=observed_dist, f_exp=expected_dist)
    return ParityOutput(cs, p, None)


@set_metadata
def parity(metadata: MetadataOutput) -> ParityOutput[NDArray[np.float64]]:
    """
    Calculate chi-square statistics to assess the linear relationship between multiple factors
    and class labels.

    This function computes the chi-square statistic for each metadata factor to determine if there is
    a significant relationship between the factor values and class labels. The chi-square statistic is
    only valid for linear relationships. If non-linear relationships exist, use `balance`.

    Parameters
    ----------
    metadata : MetadataOutput
        Output after running `metadata_preprocessing`

    Returns
    -------
    ParityOutput[NDArray[np.float64]]
        Arrays of length (num_factors) whose (i)th element corresponds to the
        chi-square score and :term:`p-value<P-Value>` for the relationship between factor i and
        the class labels in the dataset.

    Raises
    ------
    Warning
        If any cell in the contingency matrix has a value between 0 and 5, a warning is issued because this can
        lead to inaccurate chi-square calculations. It is recommended to ensure that each label co-occurs with
        factor values either 0 times or at least 5 times.

    Note
    ----
    - A high score with a low p-value suggests that a metadata factor is strongly correlated with a class label.
    - The function creates a contingency matrix for each factor, where each entry represents the frequency of a
      specific factor value co-occurring with a particular class label.
    - Rows containing only zeros in the contingency matrix are removed before performing the chi-square test
      to prevent errors in the calculation.

    See Also
    --------
    balance

    Examples
    --------
    Randomly creating some "continuous" and categorical variables using ``np.random.default_rng``

    >>> labels = np_random_gen.choice([0, 1, 2], (100))
    >>> metadata_dict = [
    ...     {
    ...         "age": list(np_random_gen.choice([25, 30, 35, 45], (100))),
    ...         "income": list(np_random_gen.choice([50000, 65000, 80000], (100))),
    ...         "gender": list(np_random_gen.choice(["M", "F"], (100))),
    ...     }
    ... ]
    >>> continuous_factor_bincounts = {"age": 4, "income": 3}
    >>> metadata = metadata_preprocessing(metadata_dict, labels, continuous_factor_bincounts)
    >>> parity(metadata)
    ParityOutput(score=array([7.35731943, 5.46711299, 0.51506212]), p_value=array([0.28906231, 0.24263543, 0.77295762]), metadata_names=['age', 'income', 'gender'])
    """  # noqa: E501
    chi_scores = np.zeros(metadata.discrete_data.shape[1])
    p_values = np.zeros_like(chi_scores)
    not_enough_data = {}
    for i, col_data in enumerate(metadata.discrete_data.T):
        # Builds a contingency matrix where entry at index (r,c) represents
        # the frequency of current_factor_name achieving value unique_factor_values[r]
        # at a data point with class c.
        results = crosstab(col_data, metadata.class_labels)
        contingency_matrix = as_numpy(results.count)  # type: ignore

        # Determines if any frequencies are too low
        counts = np.nonzero(contingency_matrix < 5)
        unique_factor_values = np.unique(col_data)
        current_factor_name = metadata.discrete_factor_names[i]
        for int_factor, int_class in zip(counts[0], counts[1]):
            if contingency_matrix[int_factor, int_class] > 0:
                factor_category = unique_factor_values[int_factor]
                if current_factor_name not in not_enough_data:
                    not_enough_data[current_factor_name] = {}
                if factor_category not in not_enough_data[current_factor_name]:
                    not_enough_data[current_factor_name][factor_category] = []
                not_enough_data[current_factor_name][factor_category].append(
                    (metadata.class_names[int_class], int(contingency_matrix[int_factor, int_class]))
                )

        # This deletes rows containing only zeros,
        # because scipy.stats.chi2_contingency fails when there are rows containing only zeros.
        rowsums = np.sum(contingency_matrix, axis=1)
        rowmask = np.nonzero(rowsums)[0]
        contingency_matrix = contingency_matrix[rowmask]

        chi2, p, _, _ = chi2_contingency(contingency_matrix)

        chi_scores[i] = chi2
        p_values[i] = p

    if not_enough_data:
        factor_msg = []
        for factor, fact_dict in not_enough_data.items():
            stacked_msg = []
            for key, value in fact_dict.items():
                msg = []
                for item in value:
                    msg.append(f"label {item[0]}: {item[1]} occurrences")
                flat_msg = "\n\t\t".join(msg)
                stacked_msg.append(f"value {key} - {flat_msg}\n\t")
            factor_msg.append(factor + " - " + "".join(stacked_msg))

        message = "\n".join(factor_msg)

        warnings.warn(
            f"The following factors did not meet the recommended 5 occurrences for each value-label combination. \n\
            Recommend rerunning parity after adjusting the following factor-value-label combinations: \n{message}",
            UserWarning,
        )

    return ParityOutput(chi_scores, p_values, metadata.discrete_factor_names)
