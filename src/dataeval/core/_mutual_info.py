from __future__ import annotations

__all__ = []

import logging
from collections.abc import Iterable
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from dataeval.config import EPSILON, get_max_processes, get_seed
from dataeval.core._bin import get_counts, is_continuous
from dataeval.types import Array1D, Array2D
from dataeval.utils._array import as_numpy, opt_as_numpy

_logger = logging.getLogger(__name__)


class MutualInfoResult(TypedDict):
    """
    Type definition for mutual information function output.

    Attributes
    ----------
    class_to_factor : NDArray[np.float64]
        1D array of length (num_factors+1) containing normalized mutual information between
        class labels and each factor (including class label itself at index 0).
    interfactor : NDArray[np.float64]
        (num_factors) x (num_factors) symmetric matrix of normalized mutual information
        between metadata factors only (excluding class labels).
    """

    class_to_factor: NDArray[np.float64]
    interfactor: NDArray[np.float64]


def _validate_num_neighbors(num_neighbors: int) -> int:
    if not isinstance(num_neighbors, int | float):
        raise TypeError(
            f"Variable {num_neighbors} is not real-valued numeric type."
            "num_neighbors should be an int, greater than 0 and less than"
            "the number of samples in the dataset"
        )
    if num_neighbors < 1:
        raise ValueError(
            f"Invalid value for {num_neighbors}."
            "Choose a value greater than 0 and less than number of samples"
            "in the dataset."
        )
    if isinstance(num_neighbors, float):
        num_neighbors = int(num_neighbors)
        _logger.warning(f"Variable {num_neighbors} is currently type float and will be truncated to type int.")

    return num_neighbors


def _merge_labels_and_factors(
    class_labels: NDArray[np.intp], factor_data: NDArray[np.intp], discrete_features: Iterable[bool] | None
) -> tuple[NDArray[np.intp], list[bool]]:
    discrete_features = [False] + (
        [not is_continuous(d) for d in factor_data.T] if discrete_features is None else list(discrete_features)
    )

    # Use numeric data for MI
    data = np.hstack((class_labels[:, np.newaxis], factor_data))
    # Present discrete features composed of distinct values as continuous for `mutual_info_classif`
    for i in range(len(discrete_features)):
        if len(data) == len(np.unique(data[:, i])):
            discrete_features[i] = False

    return data, discrete_features


def mutual_info(
    class_labels: Array1D[int],
    factor_data: Array2D[int | float],
    discrete_features: Array1D[bool] | None = None,
    num_neighbors: int = 5,
) -> MutualInfoResult:
    """
    Mutual information between factors (class label, metadata, label/image properties).

    Parameters
    ----------
    class_labels : Array1D[int]
        Target class labels as integer indices. Can be a 1D list, or array-like object.
    factor_data : Array2D[int | float]
        Factor values after binning or digitization. Can be a 2D list, or array-like object.
    discrete_features : Array1D[bool] | None = None
        Boolean array defining whether or not the feature set is discretized. Can be a 1D list, or array-like object.
    num_neighbors : int = 5
        Number of points to consider as neighbors.

    Returns
    -------
    MutualInfoResult
        TypedDict containing:
        - class_to_factor: 1D array of MI between class labels and each factor
        - interfactor: (num_factors) x (num_factors) matrix of MI between factors only

    Notes
    -----
    We use `mutual_info_classif` from sklearn since class label is categorical.
    `mutual_info_classif` outputs are consistent up to O(1e-4) and depend on a random
    seed. MI is computed differently for categorical and continuous variables.

    Example
    -------
    Return balance (mutual information) of factors with class_labels

    >>> class_labels, binned_data = generate_random_class_labels_and_binned_data(
    ...     labels=["doctor", "artist", "teacher"],
    ...     factors={"age": [25, 30, 35, 45], "income": [50000, 65000, 80000], "gender": ["M", "F"]},
    ...     length=100,
    ...     random_seed=175,
    ... )

    >>> result = mutual_info(class_labels=class_labels, factor_data=binned_data)
    >>> result["class_to_factor"]
    array([1.017, 0.034, 0.   , 0.028])
    >>> result["interfactor"]
    array([[1.   , 0.015, 0.038],
           [0.015, 1.   , 0.008],
           [0.038, 0.008, 1.   ]])

    See Also
    --------
    sklearn.feature_selection.mutual_info_classif
    sklearn.feature_selection.mutual_info_regression
    sklearn.metrics.mutual_info_score
    """
    _logger.info("Starting mutual_info calculation with num_neighbors=%d", num_neighbors)

    class_labels_np = as_numpy(class_labels, dtype=np.intp, required_ndim=1)
    factor_data_np = as_numpy(factor_data, required_ndim=2)
    discrete_feat_np = opt_as_numpy(discrete_features, dtype=np.bool_, required_ndim=1)

    _logger.debug("Input shapes: class_labels=%s, factor_data=%s", class_labels_np.shape, factor_data_np.shape)

    num_neighbors = _validate_num_neighbors(num_neighbors)
    data, discrete_list = _merge_labels_and_factors(class_labels_np, factor_data_np, discrete_feat_np)
    num_factors = len(discrete_list)

    _logger.debug("Computing MI for %d factors (%d discrete)", num_factors, sum(discrete_list))

    # initialize output matrix
    mi = np.full((num_factors, num_factors), np.nan, dtype=np.float32)

    for idx, is_discrete in enumerate(discrete_list):
        mi[idx, :] = (mutual_info_classif if is_discrete else mutual_info_regression)(
            data,
            data[:, idx],
            discrete_features=discrete_list,  # type: ignore - sklearn function not typed
            n_neighbors=num_neighbors,
            random_state=get_seed(),
            n_jobs=get_max_processes(),  # type: ignore - added in 1.5
        )

    # Normalization via entropy
    bin_cnts = get_counts(data)
    ent_factor = entropy(bin_cnts, axis=0)
    norm_factor = 0.5 * np.add.outer(ent_factor, ent_factor) + EPSILON
    full_matrix = 0.5 * (mi + mi.T) / norm_factor

    _logger.info(
        "Mutual info calculation complete: %d factors, mean class_to_factor MI=%.4f",
        num_factors - 1,
        np.mean(full_matrix[0, 1:]),
    )

    return MutualInfoResult(
        class_to_factor=full_matrix[0, :],
        interfactor=full_matrix[1:, 1:],
    )


def mutual_info_classwise(
    class_labels: Array1D[int],
    factor_data: Array2D[int],
    discrete_features: Array1D[bool] | None = None,
    num_neighbors: int = 5,
) -> NDArray[np.float64]:
    """
    Mutual information (MI) between factors (class label, metadata, label/image properties).

    Parameters
    ----------
    class_labels : Array1D[int]
        Target class labels as integer indices. Can be a 1D list, or array-like object.
    factor_data : Array2D[int]
        Factor values after binning or digitization. Can be a 1D list, or array-like object.
    discrete_features : Array1D[bool] | None = None
        Boolean array or iterable defining whether or not the feature set is discretized.
        Can be a 1D list, or array-like object.
    num_neighbors : int = 5
        Number of points to consider as neighbors.

    Returns
    -------
    NDArray[np.float64]
        (num_factors+1) x (num_factors+1) estimate of mutual information \
            between num_factors metadata factors and class label. Symmetry is enforced.

    Notes
    -----
    We use `mutual_info_classif` from sklearn since class label is categorical.
    `mutual_info_classif` outputs are consistent up to O(1e-4) and depend on a random
    seed. MI is computed differently for categorical and continuous variables.

    Example
    -------
    Return balance (mutual information) of factors with class_labels

    >>> class_labels, binned_data = generate_random_class_labels_and_binned_data(
    ...     labels=["doctor", "artist", "teacher"],
    ...     factors={
    ...         "age": [25, 30, 35, 45],
    ...         "income": [50000, 65000, 80000],
    ...         "gender": ["M", "F"]},
    ...     length=100,
    ...     random_seed=175)

    Return classwise balance (mutual information) of factors with individual class_labels

    >>> mutual_info_classwise(class_labels=class_labels, factor_data=binned_data)
    array([[7.818e-01, 1.388e-02, 1.803e-03, 7.282e-04],
           [7.084e-01, 2.934e-02, 1.744e-02, 3.996e-03],
           [7.295e-01, 1.157e-02, 2.799e-02, 9.451e-04]])

    See Also
    --------
    sklearn.feature_selection.mutual_info_classif
    sklearn.feature_selection.mutual_info_regression
    sklearn.metrics.mutual_info_score
    """
    _logger.info("Starting mutual_info_classwise calculation with num_neighbors=%d", num_neighbors)

    class_labels_np = as_numpy(class_labels, dtype=np.intp, required_ndim=1)
    factor_data_np = as_numpy(factor_data, dtype=np.intp, required_ndim=2)
    discrete_feat_np = opt_as_numpy(discrete_features, dtype=np.bool_, required_ndim=1)

    num_neighbors = _validate_num_neighbors(num_neighbors)
    data, discrete_list = _merge_labels_and_factors(class_labels_np, factor_data_np, discrete_feat_np)
    num_factors = len(discrete_list)
    u_classes = np.unique(class_labels_np)
    num_classes = len(u_classes)

    _logger.debug("Computing classwise MI for %d classes and %d factors", num_classes, num_factors)

    # initialize output matrix
    classwise_mi = np.full((num_classes, num_factors), np.nan, dtype=np.float32)

    # classwise targets
    tgt_bin = data[:, 0][:, None] == u_classes

    # classification MI for discrete/categorical features
    for idx in range(num_classes):
        classwise_mi[idx, :] = mutual_info_classif(
            data,
            tgt_bin[:, idx],
            discrete_features=discrete_list,  # type: ignore - sklearn function not typed
            n_neighbors=num_neighbors,
            random_state=get_seed(),
            n_jobs=get_max_processes(),  # type: ignore - added in 1.5
        )

    # Classwise normalization via entropy
    bin_cnts = get_counts(data)
    ent_factor = entropy(bin_cnts, axis=0)
    classwise_bin_cnts = get_counts(tgt_bin)
    ent_tgt_bin = entropy(classwise_bin_cnts, axis=0)
    norm_factor = 0.5 * np.add.outer(ent_tgt_bin, ent_factor) + EPSILON
    result = classwise_mi / norm_factor

    _logger.info("Mutual info classwise calculation complete: %d classes x %d factors", num_classes, num_factors)

    return result
