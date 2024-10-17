from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Generic, Mapping, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2_contingency, chisquare

from dataeval._internal.interop import to_numpy
from dataeval._internal.output import OutputMetadata, set_metadata

TData = TypeVar("TData", np.float64, NDArray[np.float64])


@dataclass(frozen=True)
class ParityOutput(Generic[TData], OutputMetadata):
    """
    Output class for :func:`parity` and :func:`label_parity` bias metrics

    Attributes
    ----------
    score : np.float64 | NDArray[np.float64]
        chi-squared score(s) of the test
    p_value : np.float64 | NDArray[np.float64]
        p-value(s) of the test
    """

    score: TData
    p_value: TData


def digitize_factor_bins(continuous_values: NDArray, bins: int, factor_name: str) -> NDArray:
    """
    Digitizes a list of values into a given number of bins.

    Parameters
    ----------
    continuous_values: NDArray
        The values to be digitized.
    bins: int
        The number of bins for the discrete values that continuous_values will be digitized into.
    factor_name: str
        The name of the factor to be digitized.

    Returns
    -------
    NDArray
        The digitized values
    """

    if not np.all([np.issubdtype(type(n), np.number) for n in continuous_values]):
        raise TypeError(
            f"Encountered a non-numeric value for factor {factor_name}, but the factor"
            " was specified to be continuous. Ensure all occurrences of this factor are numeric types,"
            f" or do not specify {factor_name} as a continuous factor."
        )

    _, bin_edges = np.histogram(continuous_values, bins=bins)
    bin_edges[-1] = np.inf
    bin_edges[0] = -np.inf
    return np.digitize(continuous_values, bin_edges)


def format_discretize_factors(
    data_factors: Mapping[str, NDArray], continuous_factor_bincounts: Mapping[str, int]
) -> dict[str, NDArray]:
    """
    Sets up the internal list of metadata factors.

    Parameters
    ----------
    data_factors: Dict[str, NDArray]
        The dataset factors, which are per-image attributes including class label and metadata.
        Each key of dataset_factors is a factor, whose value is the per-image factor values.
    continuous_factor_bincounts : Dict[str, int]
        The factors in data_factors that have continuous values and the array of bin counts to
        discretize values into. All factors are treated as having discrete values unless they
        are specified as keys in this dictionary. Each element of this array must occur as a key
        in data_factors.

    Returns
    -------
    Dict[str, NDArray]
        - Intrinsic per-image metadata information with the formatting that input data_factors uses.
          Each key is a metadata factor, whose value is the discrete per-image factor values.
    """

    invalid_keys = set(continuous_factor_bincounts.keys()) - set(data_factors.keys())
    if invalid_keys:
        raise KeyError(
            f"The continuous factor(s) {invalid_keys} do not exist in data_factors. Delete these "
            "keys from `continuous_factor_names` or add corresponding entries to `data_factors`."
        )

    metadata_factors = {}

    # make sure each factor has the same number of entries
    lengths = []
    for arr in data_factors.values():
        lengths.append(arr.shape)

    if lengths[1:] != lengths[:-1]:
        raise ValueError("The lengths of each entry in the dictionary are not equal." f" Found lengths {lengths}")

    metadata_factors = {
        name: val
        if name not in continuous_factor_bincounts
        else digitize_factor_bins(val, continuous_factor_bincounts[name], name)
        for name, val in data_factors.items()
        if name != "class"
    }

    return metadata_factors


def normalize_expected_dist(expected_dist: NDArray, observed_dist: NDArray) -> NDArray:
    """
    Normalize the expected label distribution to match the total number of labels in the observed distribution.

    This function adjusts the expected distribution so that its sum equals the sum of the observed distribution.
    If the expected distribution is all zeros, an error is raised.

    Parameters
    ----------
    expected_dist : np.ndarray
        The expected label distribution. This array represents the anticipated distribution of labels.
    observed_dist : np.ndarray
        The observed label distribution. This array represents the actual distribution of labels in the dataset.

    Returns
    -------
    np.ndarray
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


def validate_dist(label_dist: NDArray, label_name: str):
    """
    Verifies that the given label distribution has labels and checks if
    any labels have frequencies less than 5.

    Parameters
    ----------
    label_dist : NDArray
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
            " to invalid chi-squared evaluation.",
            UserWarning,
        )


@set_metadata("dataeval.metrics")
def label_parity(
    expected_labels: ArrayLike,
    observed_labels: ArrayLike,
    num_classes: int | None = None,
) -> ParityOutput[np.float64]:
    """
    Calculate the chi-square statistic to assess the parity between expected and observed label distributions.

    This function computes the frequency distribution of classes in both expected and observed labels, normalizes
    the expected distribution to match the total number of observed labels, and then calculates the chi-square
    statistic to determine if there is a significant difference between the two distributions.

    Parameters
    ----------
    expected_labels : ArrayLike
        List of class labels in the expected dataset
    observed_labels : ArrayLike
        List of class labels in the observed dataset
    num_classes : int | None, default None
        The number of unique classes in the datasets. If not provided, the function will infer it
        from the set of unique labels in expected_labels and observed_labels

    Returns
    -------
    ParityOutput[np.float64]
        chi-squared score and p-value of the test

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
    - It then performs a chi-square test to determine if there is a statistically significant difference between
      the observed and expected label distributions.
    - This function acts as an interface to the scipy.stats.chisquare method, which is documented at
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html


    Examples
    --------
    Randomly creating some label distributions using ``np.random.default_rng``

    >>> expected_labels = np_random_gen.choice([0, 1, 2, 3, 4], (100))
    >>> observed_labels = np_random_gen.choice([2, 3, 0, 4, 1], (100))
    >>> label_parity(expected_labels, observed_labels)
    ParityOutput(score=14.007374204742625, p_value=0.0072715574616218)
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
    return ParityOutput(cs, p)


@set_metadata("dataeval.metrics")
def parity(
    class_labels: ArrayLike,
    data_factors: Mapping[str, ArrayLike],
    continuous_factor_bincounts: Mapping[str, int] | None = None,
) -> ParityOutput[NDArray[np.float64]]:
    """
    Calculate chi-square statistics to assess the relationship between multiple factors and class labels.

    This function computes the chi-square statistic for each metadata factor to determine if there is
    a significant relationship between the factor values and class labels. The function handles both categorical
    and discretized continuous factors.

    Parameters
    ----------
    class_labels: ArrayLike
        List of class labels for each image
    data_factors: Mapping[str, ArrayLike]
        The dataset factors, which are per-image metadata attributes.
        Each key of dataset_factors is a factor, whose value is the per-image factor values.
    continuous_factor_bincounts : Mapping[str, int] | None, default None
        A dictionary specifying the number of bins for discretizing the continuous factors.
        The keys should correspond to the names of continuous factors in `data_factors`,
        and the values should be the number of bins to use for discretization.
        If not provided, no discretization is applied.

    Returns
    -------
    ParityOutput[NDArray[np.float64]]
        Arrays of length (num_factors) whose (i)th element corresponds to the
        chi-square score and p-value for the relationship between factor i and
        the class labels in the dataset.

    Raises
    ------
    Warning
        If any cell in the contingency matrix has a value between 0 and 5, a warning is issued because this can
        lead to inaccurate chi-square calculations. It is recommended to ensure that each label co-occurs with
        factor values either 0 times or at least 5 times. Alternatively, continuous-valued factors can be digitized
        into fewer bins.

    Note
    ----
    - Each key of the ``continuous_factor_bincounts`` dictionary must occur as a key in data_factors.
    - A high score with a low p-value suggests that a metadata factor is strongly correlated with a class label.
    - The function creates a contingency matrix for each factor, where each entry represents the frequency of a
      specific factor value co-occurring with a particular class label.
    - Rows containing only zeros in the contingency matrix are removed before performing the chi-square test
      to prevent errors in the calculation.

    Examples
    --------
    Randomly creating some "continuous" and categorical variables using ``np.random.default_rng``

    >>> labels = np_random_gen.choice([0, 1, 2], (100))
    >>> data_factors = {
    ...     "age": np_random_gen.choice([25, 30, 35, 45], (100)),
    ...     "income": np_random_gen.choice([50000, 65000, 80000], (100)),
    ...     "gender": np_random_gen.choice(["M", "F"], (100)),
    ... }
    >>> continuous_factor_bincounts = {"age": 4, "income": 3}
    >>> parity(labels, data_factors, continuous_factor_bincounts)
    ParityOutput(score=array([7.35731943, 5.46711299, 0.51506212]), p_value=array([0.28906231, 0.24263543, 0.77295762]))
    """
    if len(np.shape(class_labels)) > 1:
        raise ValueError(
            f"Got class labels with {len(np.shape(class_labels))}-dimensional",
            f" shape {np.shape(class_labels)}, but expected a 1-dimensional array.",
        )

    data_factors_np = {k: to_numpy(v) for k, v in data_factors.items()}
    continuous_factor_bincounts = continuous_factor_bincounts if continuous_factor_bincounts else {}

    labels = to_numpy(class_labels)
    factors = format_discretize_factors(data_factors_np, continuous_factor_bincounts)

    chi_scores = np.zeros(len(factors))
    p_values = np.zeros(len(factors))
    n_cls = len(np.unique(labels))
    not_enough_data = {}
    for i, (current_factor_name, factor_values) in enumerate(factors.items()):
        unique_factor_values = np.unique(factor_values)
        contingency_matrix = np.zeros((len(unique_factor_values), n_cls))
        # Builds a contingency matrix where entry at index (r,c) represents
        # the frequency of current_factor_name achieving value unique_factor_values[r]
        # at a data point with class c.

        # TODO: Vectorize this nested for loop
        for fi, factor_value in enumerate(unique_factor_values):
            for label in range(n_cls):
                with_both = np.bitwise_and((labels == label), factor_values == factor_value)
                contingency_matrix[fi, label] = np.sum(with_both)
                if 0 < contingency_matrix[fi, label] < 5:
                    if current_factor_name not in not_enough_data:
                        not_enough_data[current_factor_name] = {}
                    if factor_value not in not_enough_data[current_factor_name]:
                        not_enough_data[current_factor_name][factor_value] = []
                    not_enough_data[current_factor_name][factor_value].append(
                        (label, int(contingency_matrix[fi, label]))
                    )

        # This deletes rows containing only zeros,
        # because scipy.stats.chi2_contingency fails when there are rows containing only zeros.
        rowsums = np.sum(contingency_matrix, axis=1)
        rowmask = np.where(rowsums)
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

    return ParityOutput(chi_scores, p_values)
