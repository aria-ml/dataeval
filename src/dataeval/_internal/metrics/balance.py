from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from dataeval._internal.metrics.utils import entropy, preprocess_metadata
from dataeval._internal.output import OutputMetadata, set_metadata


@dataclass(frozen=True)
class BalanceOutput(OutputMetadata):
    """
    Output class for :func:`balance` bias metric

    Attributes
    ----------
    balance : NDArray[np.float64]
        Estimate of mutual information between metadata factors and class label
    factors : NDArray[np.float64]
        Estimate of inter/intra-factor mutual information
    classwise : NDArray[np.float64]
        Estimate of mutual information between metadata factors and individual class labels
    """

    balance: NDArray[np.float64]
    factors: NDArray[np.float64]
    classwise: NDArray[np.float64]


def validate_num_neighbors(num_neighbors: int) -> int:
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


@set_metadata("dataeval.metrics")
def balance(class_labels: ArrayLike, metadata: Mapping[str, ArrayLike], num_neighbors: int = 5) -> BalanceOutput:
    """
    Mutual information (MI) between factors (class label, metadata, label/image properties)

    Parameters
    ----------
    class_labels: ArrayLike
        List of class labels for each image
    metadata: Mapping[str, ArrayLike]
        Dict of lists of metadata factors for each image
    num_neighbors: int, default 5
        Number of nearest neighbors to use for computing MI between discrete
        and continuous variables.

    Returns
    -------
    BalanceOutput
        (num_factors+1) x (num_factors+1) estimate of mutual information
        between num_factors metadata factors and class label. Symmetry is enforced.

    Note
    ----
    We use `mutual_info_classif` from sklearn since class label is categorical.
    `mutual_info_classif` outputs are consistent up to O(1e-4) and depend on a random
    seed. MI is computed differently for categorical and continuous variables, and
    we attempt to infer whether a variable is categorical by the fraction of unique
    values in the dataset.

    Example
    -------
    Return balance (mutual information) of factors with class_labels

    >>> bal = balance(class_labels, metadata)
    >>> bal.balance
    array([0.99999822, 0.13363788, 0.04505382, 0.02994455])

    Return intra/interfactor balance (mutual information)

    >>> bal.factors
    array([[0.99999843, 0.04133555, 0.09725766],
           [0.04133555, 0.08433558, 0.1301489 ],
           [0.09725766, 0.1301489 , 0.99999856]])

    Return classwise balance (mutual information) of factors with individual class_labels

    >>> bal.classwise
    array([[0.99999822, 0.13363788, 0.        , 0.        ],
           [0.99999822, 0.13363788, 0.        , 0.        ]])

    See Also
    --------
    sklearn.feature_selection.mutual_info_classif
    sklearn.feature_selection.mutual_info_regression
    sklearn.metrics.mutual_info_score
    """
    num_neighbors = validate_num_neighbors(num_neighbors)
    data, names, is_categorical = preprocess_metadata(class_labels, metadata)
    num_factors = len(names)
    mi = np.empty((num_factors, num_factors))
    mi[:] = np.nan

    for idx in range(num_factors):
        tgt = data[:, idx].astype(int)

        if is_categorical[idx]:
            mi[idx, :] = mutual_info_classif(
                data,
                tgt,
                discrete_features=is_categorical,  # type: ignore
                n_neighbors=num_neighbors,
                random_state=0,
            )
        else:
            mi[idx, :] = mutual_info_regression(
                data,
                tgt,
                discrete_features=is_categorical,  # type: ignore
                n_neighbors=num_neighbors,
                random_state=0,
            )

    ent_all = entropy(data, names, is_categorical, normalized=False)
    norm_factor = 0.5 * np.add.outer(ent_all, ent_all) + 1e-6
    # in principle MI should be symmetric, but it is not in practice.
    nmi = 0.5 * (mi + mi.T) / norm_factor
    balance = nmi[0]
    factors = nmi[1:, 1:]

    # unique class labels
    class_idx = names.index("class_label")
    class_data = data[:, class_idx].astype(int)
    u_cls = np.unique(class_data)
    num_classes = len(u_cls)

    # assume class is a factor
    classwise_mi = np.empty((num_classes, num_factors))
    classwise_mi[:] = np.nan

    # categorical variables, excluding class label
    cat_mask = np.concatenate((is_categorical[:class_idx], is_categorical[(class_idx + 1) :]), axis=0).astype(int)

    tgt_bin = np.stack([class_data == cls for cls in u_cls]).T.astype(int)
    ent_tgt_bin = entropy(
        tgt_bin, names=[str(idx) for idx in range(num_classes)], is_categorical=[True for idx in range(num_classes)]
    )

    # classification MI for discrete/categorical features
    for idx in range(num_classes):
        # tgt = class_data == cls
        # units: nat
        classwise_mi[idx, :] = mutual_info_classif(
            data,
            tgt_bin[:, idx],
            discrete_features=cat_mask,  # type: ignore
            n_neighbors=num_neighbors,
            random_state=0,
        )

    norm_factor = 0.5 * np.add.outer(ent_tgt_bin, ent_all) + 1e-6
    classwise = classwise_mi / norm_factor

    return BalanceOutput(balance, factors, classwise)
