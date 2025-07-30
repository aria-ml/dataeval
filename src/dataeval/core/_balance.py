from __future__ import annotations

__all__ = []

import warnings
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from dataeval.config import EPSILON, get_max_processes, get_seed
from dataeval.core._bin import get_counts, is_continuous


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
        warnings.warn(f"Variable {num_neighbors} is currently type float and will be truncated to type int.")

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


def balance(
    class_labels: NDArray[np.intp],
    factor_data: NDArray[np.intp],
    discrete_features: Iterable[bool] | None = None,
    num_neighbors: int = 5,
) -> NDArray[np.float64]:
    num_neighbors = _validate_num_neighbors(num_neighbors)
    data, discrete_features = _merge_labels_and_factors(class_labels, factor_data, discrete_features)
    num_factors = len(discrete_features)

    # initialize output matrix
    mi = np.full((num_factors, num_factors), np.nan, dtype=np.float32)

    for idx, is_discrete in enumerate(discrete_features):
        mi[idx, :] = (mutual_info_classif if is_discrete else mutual_info_regression)(
            data,
            data[:, idx],
            discrete_features=discrete_features,  # type: ignore - sklearn function not typed
            n_neighbors=num_neighbors,
            random_state=get_seed(),
            n_jobs=get_max_processes(),  # type: ignore - added in 1.5
        )

    # Normalization via entropy
    bin_cnts = get_counts(data)
    ent_factor = entropy(bin_cnts, axis=0)
    norm_factor = 0.5 * np.add.outer(ent_factor, ent_factor) + EPSILON
    return 0.5 * (mi + mi.T) / norm_factor


def balance_classwise(
    class_labels: NDArray[np.intp],
    factor_data: NDArray[np.intp],
    discrete_features: Iterable[bool] | None = None,
    num_neighbors: int = 5,
) -> NDArray[np.float64]:
    num_neighbors = _validate_num_neighbors(num_neighbors)
    data, discrete_features = _merge_labels_and_factors(class_labels, factor_data, discrete_features)
    num_factors = len(discrete_features)
    u_classes = np.unique(class_labels)
    num_classes = len(u_classes)

    # initialize output matrix
    classwise_mi = np.full((num_classes, num_factors), np.nan, dtype=np.float32)

    # classwise targets
    tgt_bin = data[:, 0][:, None] == u_classes

    # classification MI for discrete/categorical features
    for idx in range(num_classes):
        classwise_mi[idx, :] = mutual_info_classif(
            data,
            tgt_bin[:, idx],
            discrete_features=discrete_features,  # type: ignore - sklearn function not typed
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
    return classwise_mi / norm_factor
