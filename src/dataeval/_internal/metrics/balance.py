import warnings
from typing import Dict, List, Sequence

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from dataeval._internal.metrics.base import EvaluateMixin
from dataeval._internal.metrics.functional import entropy, preprocess_metadata


class BaseBalanceMetric(EvaluateMixin):
    """
    Base class for balance (mutual information) metrics.  Contains input
    validation for balance metrics.
    """

    def __init__(self, num_neighbors: int):
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

        self.num_neighbors = num_neighbors


class Balance(BaseBalanceMetric):
    """
    Metadata balance measures distributional correlation between metadata
    factors and class label to identify opportunities for shortcut learning or
    sampling bias in the dataset.

    Parameters
    ----------
    num_neighbors: int
        number of nearest neighbors used for the computation of

    Notes
    -----
    We use mutual_info_classif from sklearn since class label is categorical
    mutual_info_classif outputs are consistent up to O(1e-4) and depend on
        a random seed.
    MI is computed differently for categorical and continuous variables,
        and we attempt to infer whether a variable is categorical by the
        fraction of unique values in the dataset.

    See Also
    --------
    sklearn.feature_selection.mutual_info_classif
    sklearn.feature_selection.mutual_info_regression
    sklearn.metrics.mutual_info_score
    """

    def __init__(self, num_neighbors: int = 5):
        super().__init__(num_neighbors=num_neighbors)

    def evaluate(self, class_labels: Sequence[int], metadata: List[Dict]) -> np.ndarray:
        """
        Mutual information (MI) between factors (class label, metadata, label/image properties)

        Parameters
        ----------
        class_labels: Sequence[int]
            List of class labels for each image
        metadata: List[Dict]
            List of metadata factors for each image

        Returns
        -------
        NDArray
            (num_factors+1) x (num_factors+1) estimate of mutual information
            between num_factors metadata factors and class label. Symmetry is enforced.

        See Also
        --------
        sklearn.feature_selection.mutual_info_classif
        sklearn.feature_selection.mutual_info_regression
        sklearn.metrics.mutual_info_score
        """
        data, names, is_categorical = preprocess_metadata(class_labels, metadata)
        num_factors = len(names)
        mi = np.empty((num_factors, num_factors))
        mi[:] = np.nan

        for idx in range(num_factors):
            tgt = data[:, idx]

            if is_categorical[idx]:
                # categorical target
                mi[idx, :] = mutual_info_classif(
                    data,
                    tgt,
                    discrete_features=is_categorical,  # type: ignore
                    n_neighbors=self.num_neighbors,
                )
            else:
                # continuous variables
                mi[idx, :] = mutual_info_regression(
                    data,
                    tgt,
                    discrete_features=is_categorical,  # type: ignore
                    n_neighbors=self.num_neighbors,
                )

        ent_all = entropy(data, names, is_categorical, normalized=False)
        norm_factor = 0.5 * np.add.outer(ent_all, ent_all) + 1e-6
        # in principle MI should be symmetric, but it is not in practice.
        nmi = 0.5 * (mi + mi.T) / norm_factor

        return nmi


class BalanceClasswise(BaseBalanceMetric):
    """
    Computes mutual information (analogous to correlation) between metadata
        factors (class label, metadata, label/image properties) with individual
        class labels.

    Parameters
    ----------
    num_neighbors: int
        Number of nearest neighbors to use for computing MI between discrete
        and continuous variables.
    """

    def __init__(self, num_neighbors: int = 5):
        super().__init__(num_neighbors)

    def evaluate(self, class_labels: Sequence[int], metadata: List[Dict]) -> np.ndarray:
        """
        Compute mutual information between metadata factors (class label, metadata,
        label/image properties) with individual class labels.

        Parameters
        ----------
        class_labels: Sequence[int]
            List of class labels for each image
        metadata: List[Dict]
            List of metadata factors for each image

        Notes
        -----
        We use mutual_info_classif from sklearn since class label is categorical
        mutual_info_classif outputs are consistent up to O(1e-4) and depend on
            a random seed
        MI is computed differently for categorical and continuous variables,
            so we have to specify with is_categorical.

        Returns
        -------
        NDArray
            (num_classes x num_factors) estimate of mutual information between
            num_factors metadata factors and individual class labels.

        See Also
        --------
        sklearn.feature_selection.mutual_info_classif
        sklearn.feature_selection.mutual_info_regression
        sklearn.metrics.mutual_info_score
        compute_mutual_information
        """
        data, names, is_categorical = preprocess_metadata(class_labels, metadata)
        num_factors = len(names)
        # unique class labels
        class_idx = names.index("class_label")
        class_data = data[:, class_idx]
        u_cls = np.unique(class_data)
        num_classes = len(u_cls)

        data_no_class = np.concatenate((data[:, :class_idx], data[:, (class_idx + 1) :]), axis=1)

        # assume class is a factor
        mi = np.empty((num_classes, num_factors - 1))
        mi[:] = np.nan

        # categorical variables, excluding class label
        cat_mask = np.concatenate((is_categorical[:class_idx], is_categorical[(class_idx + 1) :]), axis=0).astype(int)

        # classification MI for discrete/categorical features
        for idx, cls in enumerate(u_cls):
            tgt = class_data == cls
            # units: nat
            mi[idx, :] = mutual_info_classif(
                data_no_class,
                tgt,
                discrete_features=cat_mask,  # type: ignore
                n_neighbors=self.num_neighbors,
            )

        # let this recompute for all features including class label
        ent_all = entropy(data, names, is_categorical)
        ent_tgt = ent_all[class_idx]
        ent_all = np.concatenate((ent_all[:class_idx], ent_all[(class_idx + 1) :]), axis=0)
        norm_factor = 0.5 * np.add.outer(ent_tgt, ent_all) + 1e-6
        nmi = mi / norm_factor
        return nmi
