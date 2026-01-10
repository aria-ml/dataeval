"""Core metadata insight functions."""

__all__ = []

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_selection import mutual_info_classif

from dataeval.config import get_max_processes, get_seed
from dataeval.protocols import SequenceLike

_logger = logging.getLogger(__name__)

_NATS2BITS = 1.442695
"""
_NATS2BITS is the reciprocal of natural log of 2. If you have an information/entropy-type quantity measured in nats,
which is what many library functions return, multiply it by _NATS2BITS to get it in bits.
"""


def _calc_median_deviations(reference: NDArray[Any], test: NDArray[Any]) -> NDArray[Any]:
    """
    Calculates deviations of the test data from the median of the reference data

    Parameters
    ----------
    reference : NDArray
        Reference values of shape (samples, factors)
    test : NDArray
        Incoming values where each sample's factors will be compared to the median of
        the reference set corresponding factors

    Returns
    -------
    NDArray
        Scaled positive and negative deviations of the test data from the reference.

    Notes
    -----
    All return values are in the range [0, pos_inf]
    """

    # Take median over samples (rows)
    ref_median = np.median(reference, axis=0)  # (F, )

    # Shift reference and test distributions by reference
    ref_dev = reference - ref_median  # (S, F) - F
    test_dev = test - ref_median  # (S_t, F) - F

    # Separate positive and negative distributions
    # Fills with nans to keep shape in both 1-D and N-D matrices
    pdev = np.where(ref_dev > 0, ref_dev, np.nan)  # (S, F)
    ndev = np.where(ref_dev < 0, ref_dev, np.nan)  # (S, F)

    # Calculate middle of positive and negative distributions per feature
    pscale = np.nanmedian(pdev, axis=0)  # (F, )
    nscale = np.abs(np.nanmedian(ndev, axis=0))  # (F, )

    # Replace 0's for division. Negatives should not happen
    pscale = np.where(pscale > 0, pscale, 1.0)  # (F, )
    nscale = np.where(nscale > 0, nscale, 1.0)  # (F, )

    # Scales positive values by positive scale and negative values by negative
    return np.abs(np.where(test_dev >= 0, test_dev / pscale, test_dev / nscale))  # (S_t, F)


def factor_deviation(
    reference_factors: Mapping[str, NDArray[Any]],
    test_factors: Mapping[str, NDArray[Any]],
    indices: SequenceLike[int],
) -> Sequence[Mapping[str, float]]:
    """
    Determine greatest deviation in metadata features per sample.

    Parameters
    ----------
    reference_factors : dict[str, NDArray]
        A dictionary mapping factor names to arrays of reference values.
        - Keys: factor names (str)
        - Values: 1D arrays of shape (n_reference,) containing reference data
        All arrays must have the same length.
    test_factors : dict[str, NDArray]
        A dictionary mapping factor names to arrays of test values.
        - Keys: factor names (str) - must match keys in reference_factors
        - Values: 1D arrays of shape (n_test,) containing test data to be evaluated
        All arrays must have the same length.
    indices : SequenceLike[int]
        Array of test sample indices. Indices must not exceed the number of test samples.

    Returns
    -------
    Sequence[Mapping[str, float]]
        A sequence of maps, one per specified test sample index (in the order provided),
        where each dictionary maps all factor names to their deviation values for that sample.
        Within each dictionary, factors are sorted by deviation value (descending order).
        Returns empty list if no indices are provided.

    Notes
    -----
    1. At least 3 reference samples are needed for meaningful deviation calculation
    2. Deviations are calculated as scaled distance from reference median
    3. Each dictionary contains all factors for a single test sample
    4. The order of dictionaries in the result matches the order of indices in the input

    Examples
    --------
    >>> reference_factors = {
    ...     "time": np.array([1.0, 2.0, 3.0]),
    ...     "altitude": np.array([100, 110, 105]),
    ... }
    >>> test_factors = {
    ...     "time": np.array([5.0, 12.0, 4.0]),
    ...     "altitude": np.array([108, 112, 500]),
    ... }
    >>> indices = [1, 2]  # Second and third test sample
    >>> factor_deviation(reference_factors, test_factors, indices)
    [{'time': 10.0, 'altitude': 1.4}, {'altitude': 79.0, 'time': 2.0}]
    """

    # Early return if no samples
    if not indices:
        return []

    if not reference_factors:
        raise ValueError("reference_factors dictionary cannot be empty")

    if not test_factors:
        raise ValueError("test_factors dictionary cannot be empty")

    # Validate that both dictionaries have the same keys
    if set(reference_factors.keys()) != set(test_factors.keys()):
        raise ValueError("reference_factors and test_factors must have the same keys")

    # Get factor names and validate all arrays have same length
    factor_names = list(reference_factors.keys())
    ref_arrays = [reference_factors[name] for name in factor_names]
    test_arrays = [test_factors[name] for name in factor_names]

    n_ref = len(ref_arrays[0])
    n_test = len(test_arrays[0])

    if max(indices) > n_test:
        raise ValueError(f"Invalid data dimensions: test={n_test}, indices={indices}")

    if not all(len(arr) == n_ref for arr in ref_arrays):
        raise ValueError("All reference factor arrays must have the same length")

    if not all(len(arr) == n_test for arr in test_arrays):
        raise ValueError("All test factor arrays must have the same length")

    if n_ref < 3:
        _logger.warning(f"At least 3 reference metadata samples are needed, got {n_ref}")
        return [{} for _ in indices]

    # Convert indices to array
    indices_array = np.asarray(indices)

    # Stack all factor arrays
    ref_data = np.column_stack(ref_arrays)  # (n_ref, n_factors)
    tst_data = np.column_stack(test_arrays)  # (n_test, n_factors)

    # Calculates deviations of all samples in test data
    # from the median values of the reference data
    deviations = _calc_median_deviations(ref_data, tst_data)  # (n_test, n_factors)

    # Get deviations for selected indices only
    selected_deviations = deviations[indices_array]  # (n_indices, n_factors)

    # Create list of dictionaries, one per index with all factors sorted by deviation
    results = []
    for sample_devs in selected_deviations:
        # Create dict with factors and their deviations
        factor_dict = {factor_name: float(dev_value) for factor_name, dev_value in zip(factor_names, sample_devs)}
        # Sort dictionary by deviation value (descending order)
        sorted_dict = dict(sorted(factor_dict.items(), key=lambda item: item[1], reverse=True))
        results.append(sorted_dict)

    return results


def factor_predictors(
    factors: Mapping[str, NDArray[Any]],
    indices: SequenceLike[int],
    discrete_features: list[bool] | None = None,
) -> Mapping[str, float]:
    """
    Computes mutual information between metadata factors and flagged sample indices.

    Given a set of metadata factors per sample and indices of flagged samples, this function
    calculates the mutual information between each factor and the flagged status.
    In other words, it finds which metadata factors most likely correlate to a
    flagged sample (e.g., outliers, OOD samples, or other anomalies).

    Parameters
    ----------
    factors : dict[str, NDArray]
        A dictionary mapping factor names to arrays of values. All arrays must have the same length.
        - Keys: factor names (str)
        - Values: Arrays of shape (n_samples,) or (n_samples, n_features_per_factor)
    indices : SequenceLike[int]
        Sequence of sample indices that are flagged for analysis.
        Indices must not exceed the number of samples in factor arrays.
    discrete_features : list[bool] | None
        List indicating whether each factor is discrete (True) or continuous (False).
        Length must match the number of factors. If None, all factors are treated as continuous.

    Returns
    -------
    Mapping[str, float]
        A map with keys corresponding to factor names, and values indicating the strength of association
        between each named factor and the flagged status, as mutual information measured in bits.
        Returns dict with 0.0 values for all factors if no indices are provided.

    Notes
    -----
    A high mutual information between a factor and flagged samples is an indication of correlation,
    but not causation. Additional analysis should be done to determine how to handle factors
    with a high mutual information.

    Examples
    --------
    >>> factors = {
    ...     "time": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    ...     "altitude": np.array([100, 110, 105, 108, 112]),
    ... }
    >>> indices = [2, 3, 4]  # Flag last three samples
    >>> factor_predictors(factors, indices)
    {'time': 0.8415720833333329, 'altitude': 0.0}
    """

    if not factors:
        raise ValueError("factors dictionary cannot be empty")

    factor_names = list(factors.keys())
    arrays = list(factors.values())

    # Validate all arrays have same length
    n_samples = len(arrays[0])
    if not all(len(arr) == n_samples for arr in arrays):
        raise ValueError("All factor arrays must have the same length")

    # Convert indices to boolean mask
    sample_mask = np.zeros(n_samples, dtype=bool)
    sample_mask[np.asarray(indices)] = True

    # No metadata correlated with flagged samples, return 0.0 for all factors
    if not any(sample_mask):
        return dict.fromkeys(factor_names, 0.0)

    # Stack all factor arrays
    data = np.column_stack(arrays)  # (n_samples, n_factors)

    # Calculate mean, std of each factor over all samples
    scaled_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0, ddof=1)  # (n_samples, n_factors)

    # Default to all continuous if not specified
    if discrete_features is None:
        discrete_features = [False] * len(factor_names)

    if len(discrete_features) != len(factor_names):
        raise ValueError(
            f"discrete_features length ({len(discrete_features)}) must match number of factors ({len(factor_names)})"
        )

    mutual_info_values = (
        mutual_info_classif(
            X=scaled_data,
            y=sample_mask,
            discrete_features=discrete_features,  # type: ignore - sklearn function not typed
            random_state=get_seed(),
            n_jobs=get_max_processes(),  # type: ignore
        )
        * _NATS2BITS
    )

    return {k: mutual_info_values[i] for i, k in enumerate(factor_names)}
