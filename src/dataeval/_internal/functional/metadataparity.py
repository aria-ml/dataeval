import warnings
from typing import Dict, Tuple

import numpy as np
import scipy


def validate_dict(d: Dict) -> None:
    """
    Verify that dict-of-arrays (proxy for dataframe) contains arrays of equal
    length.  Future iterations could include type checking, conversion from
    string to numeric types, etc.

    Parameters
    ----------
    d: Dict
        dictionary of {variable_name: values}
    """
    # assert that length of all arrays are equal -- could expand to other properties
    lengths = []
    for arr in d.values():
        lengths.append(arr.shape)

    if lengths[1:] != lengths[:-1]:
        raise ValueError("The lengths of each entry in the dictionary are not equal." f" Found lengths {lengths}")


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
    data_factors: dict[str, np.ndarray], continuous_factor_names: np.ndarray, continuous_factor_bincounts: np.ndarray
) -> Tuple[dict, np.ndarray]:
    """
    Sets up the internal list of metadata factors.

    Parameters
    ----------
    data_factors: Dict[str, np.ndarray]
        The dataset factors, which are per-image attributes including class label and metadata.
        Each key of dataset_factors is a factor, whose value is the per-image factor values.
    continuous_factor_names : np.ndarray
        The factors in data_factors that have continuous values.
        All factors are treated as having discrete values unless they
        are specified in this array. Each element of this array must occur as a key in data_factors.
    continuous_factor_bincounts : np.ndarray
        Array of the bin counts to discretize values into for each factor in continuous_factor_names.

    Returns
    -------
    Dict[str, np.ndarray]
        Intrinsic per-image metadata information with the formatting that input data_factors uses.
        Each key is a metadata factor, whose value is the discrete per-image factor values.
    np.ndarray
        Per-image labels, whose ith element is the label for the ith element of the dataset.
    """

    if len(continuous_factor_bincounts) != len(continuous_factor_names):
        raise ValueError(
            f"continuous_factor_bincounts has length {len(continuous_factor_bincounts)}, "
            f"but continuous_factor_names has length {len(continuous_factor_names)}. "
            "Each element of continuous_factor_names must have a corresponding element "
            "in continuous_factor_bincounts. Alternatively, leave continuous_factor_bincounts empty "
            "to use a default digitization of 10 bins."
        )

    # TODO: add unit test for this
    for key in continuous_factor_names:
        if key not in data_factors:
            raise KeyError(
                f"The continuous factor name {key} "
                f"does not exist in data_factors. Delete {key} from "
                f"continuous_factor_names or add an entry with key {key} to "
                "data_factors."
            )

    metadata_factors = {}

    # make sure each factor has the same number of entries
    validate_dict(data_factors)

    labels = data_factors["class"]

    # Each continuous factor is discretized into some number of bins.
    # This matches the number of bins for a factor with the factor
    num_bins = dict(zip(continuous_factor_names, continuous_factor_bincounts))

    metadata_factors = {
        name: val if name not in continuous_factor_names else digitize_factor_bins(val, num_bins[name], name)
        for name, val in data_factors.items()
        if name != "class"
    }

    return metadata_factors, labels


def compute_parity(factors: dict[str, np.ndarray], labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates the statistical independence of metadata factors from class labels.
    This performs a chi-square test, which provides a score and a p-value for
    statistical independence between each pair of a metadata factor and a class label.
    A high score with a low p-value suggests that a metadata factor is strongly
    correlated with a class label.

    Parameters
    ----------
    factors: Dict[str, np.ndarray]
        Intrinsic per-image metadata information.
        factors['key'][i] is the value of the metadata factor 'key' at the ith element of the dataset.
    labels: np.ndarray
        Dataset labels.
        Labels[i] is the label for the ith element of the dataset.

    Returns
    -------
    np.ndarray
        Array of length (num_factors) whose (i)th element corresponds to
        the chi-square score for the relationship between factor i
        and the class labels in the dataset.
    np.ndarray
        Array of length (num_factors) whose (i)th element corresponds to
        the p-value value for the chi-square test for the relationship between
        factor i and the class labels in the dataset.
    """

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

        chi2, p, _, _ = scipy.stats.chi2_contingency(contingency_matrix)

        chi_scores[i] = chi2
        p_values[i] = p
    return chi_scores, p_values
