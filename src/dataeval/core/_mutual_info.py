__all__ = []

import logging
from collections.abc import Iterable
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from dataeval.config import get_max_processes, get_seed
from dataeval.core._bin import is_continuous
from dataeval.types import Array1D, Array2D
from dataeval.utils.arrays import as_numpy, opt_as_numpy

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
    discrete_features = [True] + (
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
    Mutual information between factors (class label, metadata, label/image properties),
    transformed to lie in [0, 1].

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

        - class_to_factor: NDArray[np.float64] - 1D array of MI between class labels and each factor
        - interfactor: NDArray[np.float64] - (num_factors) x (num_factors) matrix of MI between factors only

    Notes
    -----
    We use `mutual_info_classif` from sklearn since class label is categorical.
    `mutual_info_classif` outputs are consistent up to O(1e-4) and depend on a random
    seed. MI is computed differently for categorical and continuous variables. With
    continuous variables, since there is no upper limit to the entropy of a continuous
    distribution, normalization by entropy becomes problematic.  So instead we transform
    mutual information into a balance metric using the Linfoot transformation.

    References
    ----------
    [1] `Linfoot, E.H. (1957). "An Informational Measure of Correlation." Information and
    Control, 1(1), 85-89. <https://www.sciencedirect.com/science/article/pii/S001999585790116X>`_

    Example
    -------
    Return balance (mutual information) of factors with class_labels

    >>> rng = np.random.default_rng(175)
    >>> class_labels = rng.choice([0, 1, 2], size=100)
    >>> factor_data = np.column_stack(
    ...     [
    ...         rng.choice([25, 35, 45, 55], size=100),  # age
    ...         rng.choice([50000, 65000, 80000], size=100),  # income
    ...         rng.choice([0, 1], size=100),  # gender
    ...     ]
    ... )
    >>> result = mutual_info(class_labels=class_labels, factor_data=factor_data)
    >>> result["class_to_factor"]
    array([1.   , 0.034, 0.026, 0.004])
    >>> result["interfactor"]
    array([[1.   , 0.017, 0.056],
           [0.017, 1.   , 0.01 ],
           [0.056, 0.01 , 1.   ]])

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

    # pre-compute normalization factor and use it for discrete-discrete continuous-discrete cases.
    norm_factor = np.zeros(len(discrete_list))
    for i in range(len(discrete_list)):
        if not discrete_list[i]:
            # Ensure that bogus entropies from a continuous variable will not be chosen ever.
            norm_factor[i] = np.inf
        else:
            _, counts = np.unique(data[:, i], return_counts=True)
            probs = counts / counts.sum()
            norm_factor[i] = entropy(probs)  # natural log by default; use base=2 for bits

    for idx, is_discrete in enumerate(discrete_list):
        mi[idx, :] = (mutual_info_classif if is_discrete else mutual_info_regression)(
            data,
            data[:, idx],
            discrete_features=discrete_list,  # type: ignore - sklearn function not typed
            n_neighbors=num_neighbors,
            random_state=get_seed(),
            n_jobs=get_max_processes(),  # type: ignore - added in 1.5
        )

        pass
        # Normalization via entropy, pre-computed above
        for j in range(data.shape[1]):
            if discrete_list[j] or is_discrete:
                if norm_factor[j] == 0 or norm_factor[idx] == 0:
                    mi[idx, j] = 0.0
                else:
                    mi[idx, j] /= min(norm_factor[j], norm_factor[idx])
            else:
                mi[idx, j] = 1.0 - np.exp(-2.0 * float(mi[idx, j]))  # Linfoot transformation, mi in nats

    full_matrix = 0.5 * (mi + mi.T).astype(np.float64)

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
    Mutual information (MI) between factors (class label, metadata, label/image properties),
    transformed to lie in [0, 1].

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
        (num_factors+1) x (num_factors+1) estimate of mutual information
        between num_factors metadata factors and class label. Symmetry is enforced.

    Notes
    -----
    We use `mutual_info_classif` from sklearn since class label is categorical.
    `mutual_info_classif` outputs are consistent up to O(1e-4) and depend on a random
    seed. MI is computed differently for categorical and continuous variables. We
    return a transformation of MI onto the interval [0, 1].

    Example
    -------
    Return classwise balance (mutual information) of factors with individual class_labels

    >>> rng = np.random.default_rng(175)
    >>> class_labels = rng.choice([0, 1, 2], size=100)
    >>> factor_data = np.column_stack(
    ...     [
    ...         rng.choice([25, 35, 45, 55], size=100),  # age
    ...         rng.choice([50000, 65000, 80000], size=100),  # income
    ...         rng.choice([0, 1], size=100),  # gender
    ...     ]
    ... )
    >>> mutual_info_classwise(class_labels=class_labels, factor_data=factor_data)
    array([[1.000e+00, 2.077e-02, 2.296e-03, 7.317e-04],
           [1.000e+00, 4.893e-02, 2.451e-02, 4.362e-03],
           [1.000e+00, 1.868e-02, 3.820e-02, 1.006e-03]])

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

    # classwise targets (binary indicators)
    tgt_bin = data[:, 0][:, None] == u_classes

    # Entropy of each binary class indicator
    ent_tgt = np.zeros(num_classes)
    for idx in range(num_classes):
        _, counts = np.unique(tgt_bin[:, idx], return_counts=True)
        probs = counts / counts.sum()
        ent_tgt[idx] = entropy(probs)

    # Entropy of each factor (inf for continuous)
    ent_factor = np.zeros(num_factors)
    for j in range(num_factors):
        if not discrete_list[j]:
            ent_factor[j] = np.inf
        else:
            _, counts = np.unique(data[:, j], return_counts=True)
            probs = counts / counts.sum()
            ent_factor[j] = entropy(probs)

    # Compute MI
    classwise_mi = np.full((num_classes, num_factors), np.nan, dtype=np.float32)
    for idx in range(num_classes):
        classwise_mi[idx, :] = mutual_info_classif(
            data,
            tgt_bin[:, idx],
            discrete_features=discrete_list,  # type: ignore - sklearn function not typed
            n_neighbors=num_neighbors,
            random_state=get_seed(),
            n_jobs=get_max_processes(),  # type: ignore - added in 1.5
        )

    # Normalize: vectorized with 0/0 handling
    min_ent = np.minimum.outer(ent_tgt, ent_factor)
    zero_mask = (ent_tgt[:, None] == 0) | (ent_factor[None, :] == 0)
    min_ent[zero_mask] = 1.0  # avoid division warning
    classwise_mi /= min_ent
    classwise_mi[zero_mask] = 0.0

    _logger.info("Mutual info classwise calculation complete: %d classes x %d factors", num_classes, num_factors)

    return classwise_mi.astype(np.float64)
