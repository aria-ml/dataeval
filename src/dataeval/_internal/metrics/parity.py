import warnings
from typing import Dict, Generic, Mapping, NamedTuple, Optional, Tuple, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2_contingency, chisquare

from dataeval._internal.interop import to_numpy

TValue = TypeVar("TValue", np.float64, NDArray[np.float64])


class ParityOutput(Generic[TValue], NamedTuple):
    """
    Attributes
    ----------
    score : np.float64 | NDArray[np.float64]
        chi-squared value(s) of the test
    p_value : np.float64 | NDArray[np.float64]
        p-value(s) of the test
    """

    score: TValue
    p_value: TValue


def digitize_factor_bins(continuous_values: np.ndarray, bins: int, factor_name: str):
    """
    Digitizes a list of values into a given number of bins.

    Parameters
    ----------
    continuous_values: np.ndarray
        The values to be digitized.
    bins: int
        The number of bins for the discrete values that continuous_values will be digitized into.
    factor_name: str
        The name of the factor to be digitized.

    Returns
    -------
    np.ndarray
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
    data_factors: dict[str, np.ndarray], continuous_factor_bincounts: Dict[str, int]
) -> Tuple[dict, np.ndarray]:
    """
    Sets up the internal list of metadata factors.

    Parameters
    ----------
    data_factors: Dict[str, np.ndarray]
        The dataset factors, which are per-image attributes including class label and metadata.
        Each key of dataset_factors is a factor, whose value is the per-image factor values.
    continuous_factor_bincounts : Dict[str, int]
        The factors in data_factors that have continuous values and the array of bin counts to
        discretize values into. All factors are treated as having discrete values unless they
        are specified as keys in this dictionary. Each element of this array must occur as a key
        in data_factors.

    Returns
    -------
    Dict[str, np.ndarray]
        Intrinsic per-image metadata information with the formatting that input data_factors uses.
        Each key is a metadata factor, whose value is the discrete per-image factor values.
    np.ndarray
        Per-image labels, whose ith element is the label for the ith element of the dataset.
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

    labels = data_factors["class"]

    metadata_factors = {
        name: val
        if name not in continuous_factor_bincounts
        else digitize_factor_bins(val, continuous_factor_bincounts[name], name)
        for name, val in data_factors.items()
        if name != "class"
    }

    return metadata_factors, labels


def normalize_expected_dist(expected_dist: np.ndarray, observed_dist: np.ndarray) -> np.ndarray:
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


def parity(
    expected_labels: ArrayLike,
    observed_labels: ArrayLike,
    num_classes: Optional[int] = None,
) -> ParityOutput[np.float64]:
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
    ParityOutput[np.float64]
        chi-squared score and p-value of the test

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

    cs, p = chisquare(f_obs=observed_dist, f_exp=expected_dist)
    return ParityOutput(cs, p)


def parity_metadata(
    data_factors: Mapping[str, ArrayLike],
    continuous_factor_bincounts: Optional[Dict[str, int]] = None,
) -> ParityOutput[NDArray[np.float64]]:
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
    ParityOutput[NDArray[np.float64]]
        Arrays of length (num_factors) whose (i)th element corresponds to the
        chi-square score and p-value for the relationship between factor i and
        the class labels in the dataset.
    """
    data_factors_np = {k: to_numpy(v) for k, v in data_factors.items()}
    continuous_factor_bincounts = continuous_factor_bincounts if continuous_factor_bincounts else {}

    factors, labels = format_discretize_factors(data_factors_np, continuous_factor_bincounts)

    chi_scores = np.zeros(len(factors))
    p_values = np.zeros(len(factors))
    n_cls = len(np.unique(labels))
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
                    warnings.warn(
                        f"Factor {current_factor_name} value {factor_value} co-occurs "
                        f"only {contingency_matrix[fi, label]} times with label {label}. "
                        "This can cause inaccurate chi_square calculation. Recommend"
                        "ensuring each label occurs either 0 times or at least 5 times. "
                        "Alternatively, digitize any continuous-valued factors "
                        "into fewer bins."
                    )

        # This deletes rows containing only zeros,
        # because scipy.stats.chi2_contingency fails when there are rows containing only zeros.
        rowsums = np.sum(contingency_matrix, axis=1)
        rowmask = np.where(rowsums)
        contingency_matrix = contingency_matrix[rowmask]

        chi2, p, _, _ = chi2_contingency(contingency_matrix)

        chi_scores[i] = chi2
        p_values[i] = p

    return ParityOutput(chi_scores, p_values)
