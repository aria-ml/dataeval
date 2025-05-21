from __future__ import annotations

__all__ = []

import warnings

import numpy as np
import scipy as sp
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from dataeval.config import EPSILON, get_seed
from dataeval.data import Metadata
from dataeval.outputs import BalanceOutput
from dataeval.outputs._base import set_metadata
from dataeval.utils._bin import get_counts


def _validate_num_neighbors(num_neighbors: int) -> int:
    if not isinstance(num_neighbors, (int, float)):
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


@set_metadata
def balance(
    metadata: Metadata,
    num_neighbors: int = 5,
) -> BalanceOutput:
    """
    Mutual information (MI) between factors (class label, metadata, label/image properties).

    Parameters
    ----------
    metadata : Metadata
        Preprocessed metadata
    num_neighbors : int, default 5
        Number of points to consider as neighbors

    Returns
    -------
    BalanceOutput
        (num_factors+1) x (num_factors+1) estimate of mutual information \
            between num_factors metadata factors and class label. Symmetry is enforced.

    Note
    ----
    We use `mutual_info_classif` from sklearn since class label is categorical.
    `mutual_info_classif` outputs are consistent up to O(1e-4) and depend on a random
    seed. MI is computed differently for categorical and continuous variables.

    Example
    -------
    Return balance (mutual information) of factors with class_labels

    >>> bal = balance(metadata)
    >>> bal.balance
    array([1.   , 0.134, 0.   , 0.   ])

    Return intra/interfactor balance (mutual information)

    >>> bal.factors
    array([[1.   , 0.017, 0.015],
           [0.017, 0.445, 0.245],
           [0.015, 0.245, 1.063]])

    Return classwise balance (mutual information) of factors with individual class_labels

    >>> bal.classwise
    array([[1.   , 0.134, 0.   , 0.   ],
           [1.   , 0.134, 0.   , 0.   ]])


    See Also
    --------
    sklearn.feature_selection.mutual_info_classif
    sklearn.feature_selection.mutual_info_regression
    sklearn.metrics.mutual_info_score
    """
    if not metadata.factor_names:
        raise ValueError("No factors found in provided metadata.")

    num_neighbors = _validate_num_neighbors(num_neighbors)

    data = metadata.discretized_data
    factor_types = {"class_label": "categorical"} | {k: v.factor_type for k, v in metadata.factor_info.items()}
    is_discrete = [factor_type != "continuous" for factor_type in factor_types.values()]
    num_factors = len(factor_types)

    mi = np.full((num_factors, num_factors), np.nan, dtype=np.float32)
    data = np.hstack((metadata.class_labels[:, np.newaxis], data))

    for idx, factor_type in enumerate(factor_types.values()):
        if factor_type != "continuous":
            mi[idx, :] = mutual_info_classif(
                data,
                data[:, idx],
                discrete_features=is_discrete,  # type: ignore - sklearn function not typed
                n_neighbors=num_neighbors,
                random_state=get_seed(),
            )
        else:
            mi[idx, :] = mutual_info_regression(
                data,
                data[:, idx],
                discrete_features=is_discrete,  # type: ignore - sklearn function not typed
                n_neighbors=num_neighbors,
                random_state=get_seed(),
            )

    # Normalization via entropy
    bin_cnts = get_counts(data)
    ent_factor = sp.stats.entropy(bin_cnts, axis=0)
    norm_factor = 0.5 * np.add.outer(ent_factor, ent_factor) + EPSILON

    # in principle MI should be symmetric, but it is not in practice.
    nmi = 0.5 * (mi + mi.T) / norm_factor
    balance = nmi[0]
    factors = nmi[1:, 1:]

    # assume class is a factor
    num_classes = len(metadata.class_names)
    classwise_mi = np.full((num_classes, num_factors), np.nan, dtype=np.float32)

    # classwise targets
    classes = np.unique(metadata.class_labels)
    tgt_bin = data[:, 0][:, None] == classes

    # classification MI for discrete/categorical features
    for idx in range(num_classes):
        # units: nat
        classwise_mi[idx, :] = mutual_info_classif(
            data,
            tgt_bin[:, idx],
            discrete_features=is_discrete,  # type: ignore - sklearn function not typed
            n_neighbors=num_neighbors,
            random_state=get_seed(),
        )

    # Classwise normalization via entropy
    classwise_bin_cnts = get_counts(tgt_bin)
    ent_tgt_bin = sp.stats.entropy(classwise_bin_cnts, axis=0)
    norm_factor = 0.5 * np.add.outer(ent_tgt_bin, ent_factor) + EPSILON
    classwise = classwise_mi / norm_factor

    # Grabbing factor names for plotting function
    factor_names = ["class_label"] + list(metadata.factor_names)

    return BalanceOutput(balance, factors, classwise, factor_names, metadata.class_names)
