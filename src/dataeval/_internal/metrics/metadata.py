import warnings
from typing import Dict, List

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from torchmetrics import Metric

from dataeval._internal.functional.metadata import _entropy, _get_counts, _get_num_bins, _infer_categorical


def str_to_int(d: Dict) -> Dict:
    """
    Map columns of dict that are not numeric (e.g. string) to numeric values
    that mutual information and diversity functions can accommodate.  Each
    unique string receives a unique integer value.

    Parameters
    ----------
    d: Dict
        Dictionary of ndarray feature values or descriptors.

    Returns
    -------
    Dict
        Dictionary with same keys and non-numeric values mapped to numeric values.
    """
    for key, val in d.items():
        val = val.numpy() if torch.is_tensor(val) else val
        val = np.array(val) if isinstance(val, list) else val
        # if not numeric
        if not np.issubdtype(val.dtype, np.number):
            _, mapped_vals = np.unique(val, return_inverse=True)
            d[key] = mapped_vals
    return d


def list_to_dict(list_of_dicts: List[Dict]) -> Dict:
    """
    Converts list of dicts to dict of ndarrays

    Parameters
    ----------
    list_of_dicts: List[Dict]
        list of dictionaries, typically of metadata factors

    Returns
    -------
    Dict[np.ndarray]
        dictionary whose columns are np.ndarray
    """
    return {k: np.array([dic[k] for dic in list_of_dicts]) for k in list_of_dicts[0]}


class BaseBiasMetric(Metric):
    """
    Base class for bias metrics with common functionality for consuming
    metadata---subclasses torchmetrics.Metric

    Attributes
    ---------
    data: np.ndarray
        Array of metadata factors; string variables are converted to integers
    names: List[str]
        List of the names of metadata factor variables
    is_categorical: List
        List of boolean flags for categorical features.  Mutual information is
        computed differently for categorical/discrete and continuous variables
    num_factors: int
        Number of metadata factors in the dataset
    num_samples: int
        Number of samples in the dataset
    """

    def __init__(self):
        super().__init__()
        self.names = []
        self.data = np.empty(0)
        self.is_categorical = []

        # torchmetric 'compute' function operates on these states
        self.add_state("metadata", default=[], dist_reduce_fx="cat")
        self.add_state("class_label", default=[], dist_reduce_fx="cat")

        self.num_factors = 0
        self.num_samples = 0

    def update(self, class_label: ArrayLike, metadata: List[Dict]):
        self.metadata.extend(metadata)
        self.class_label.append(class_label)

    def _collect_data(self):
        metadata_dict = {"class_label": np.concatenate(self.class_label).astype(int)}
        metadata_dict = {**metadata_dict, **list_to_dict(self.metadata)}

        # convert string variables to int
        metadata_dict = str_to_int(metadata_dict)
        self.data = np.stack(list(metadata_dict.values()), axis=-1)
        self.names = list(metadata_dict.keys())

        self.is_categorical = [_infer_categorical(metadata_dict[var], 0.25)[0] for var in self.names]

        # class_label is also in self.names
        self.num_factors = len(self.names)
        self.num_samples = len(self.metadata)


class BaseBalanceMetric(BaseBiasMetric):
    """
    Base class for balance (mutual information) metrics.  Contains input
    validation for balance metrics.
    """

    def __init__(self, num_neighbors: int):
        super().__init__()
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

    Attributes
    ---------
    data: np.ndarray
        Array of metadata factors; string variables are converted to integers
    names: List[str]
        List of the names of metadata factor variables
    is_categorical: List
        List of boolean flags for categorical features.  Mutual information is
        computed differently for categorical/discrete and continuous variables
    num_factors: int
        Number of metadata factors in the dataset
    num_samples: int
        Number of samples in the dataset

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

    def compute(self) -> NDArray:
        """
        Mutual information (MI) between factors (class label, metadata, label/image properties)

        Parameters
        ----------
            num_neighbors: int
                Number of nearest neighbors to use for computing MI between discrete
                and continuous variables.

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
        self._collect_data()
        mi = np.empty((self.num_factors, self.num_factors))
        mi[:] = np.nan

        for idx, tgt_var in enumerate(self.names):
            tgt = self.data[:, idx]

            if self.is_categorical[idx]:
                # categorical target
                mi[idx, :] = mutual_info_classif(
                    self.data,
                    tgt,
                    discrete_features=self.is_categorical,  # type: ignore
                    n_neighbors=self.num_neighbors,
                )
            else:
                # continuous variables
                mi[idx, :] = mutual_info_regression(
                    self.data,
                    tgt,
                    discrete_features=self.is_categorical,  # type: ignore
                    n_neighbors=self.num_neighbors,
                )

        ent_all = _entropy(self.data, self.names, self.is_categorical, normalized=False)
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

    Attributes
    ----------
    num_neighbors: int
        Number of nearest neighbors to use for computing MI between discrete
        and continuous variables.
    data: np.ndarray
        Array of metadata factors; string variables are converted to integers
    names: List[str]
        List of the names of metadata factor variables
    is_categorical: List
        List of boolean flags for categorical features.  Mutual information is
        computed differently for categorical/discrete and continuous variables
    num_factors: int
        Number of metadata factors in the dataset
    num_samples: int
        Number of samples in the dataset
    """

    def __init__(self, num_neighbors: int = 5):
        super().__init__(num_neighbors)

    def compute(self) -> NDArray:
        """
        Compute mutual information between metadata factors (class label, metadata,
        label/image properties) with individual class labels.

        Parameters
        ----------
        num_neighbors: int
            Number of nearest neighbors to use for computing MI between discrete
            and continuous variables.

        Notes
        -----
        We use mutual_info_classif from sklearn since class label is categorical
        mutual_info_classif outputs are consistent up to O(1e-4) and depend on
            a random seed
        MI is computed differently for categorical and continuous variables,
            so we have to specify with self.is_categorical.

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

        self._collect_data()
        # unique class labels
        class_idx = self.names.index("class_label")
        class_data = self.data[:, class_idx]
        u_cls = np.unique(class_data)
        num_classes = len(u_cls)

        data_no_class = np.concatenate((self.data[:, :class_idx], self.data[:, (class_idx + 1) :]), axis=1)

        # assume class is a factor
        mi = np.empty((num_classes, self.num_factors - 1))
        mi[:] = np.nan

        # categorical variables, excluding class label
        cat_mask = np.concatenate(
            (self.is_categorical[:class_idx], self.is_categorical[(class_idx + 1) :]), axis=0
        ).astype(int)

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
        ent_all = _entropy(self.data, self.names, self.is_categorical)
        ent_tgt = ent_all[class_idx]
        ent_all = np.concatenate((ent_all[:class_idx], ent_all[(class_idx + 1) :]), axis=0)
        norm_factor = 0.5 * np.add.outer(ent_tgt, ent_all) + 1e-6
        nmi = mi / norm_factor
        return nmi


class BaseDiversityMetric(BaseBiasMetric):
    """
    Base class for Diversity and ClasswiseDiversity metrics.

    Parameters
    ----------
    metric: str
        string variable indicating which diversity index should be used.
        Permissible values include "simpson" and "shannon"

    Attributes
    ----------
    metric: str
        string variable indicating which diversity index should be used.
        Permissible values include "simpson" and "shannon"
    data: np.ndarray
        Array of metadata factors; string variables are converted to integers
    names: List[str]
        List of the names of metadata factor variables
    is_categorical: List
        List of boolean flags for categorical features.  Mutual information is
        computed differently for categorical/discrete and continuous variables
    num_factors: int
        Number of metadata factors in the dataset
    num_samples: int
        Number of samples in the dataset
    """

    def __init__(self, metric: str):
        super().__init__()
        allowed_metrics = ["simpson", "shannon"]
        if metric.lower() not in allowed_metrics:
            raise ValueError(f"metric '{metric}' should be one of {allowed_metrics}")
        self.metric = metric

    def _diversity_simpson(self, subset_mask: np.ndarray = np.empty(shape=0)) -> np.ndarray:
        """
        Compute diversity for discrete/categorical variables and, through standard
        histogram binning, for continuous variables.

        We define diversity as a normalized form of the inverse Simpson diversity
        index.

        diversity = 1 implies that samples are evenly distributed across a particular factor
        diversity = 1/num_categories implies that all samples belong to one category/bin

        Parameters
        ----------
        subset_mask: Optional[np.ndarray[bool]]
            Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts

        Notes
        -----
        For continuous variables, histogram bins are chosen automatically.  See
            numpy.histogram for details.
        The expression is undefined for q=1, but it approaches the Shannon entropy
            in the limit.
        If there is only one category, the diversity index takes a value of 1 =
            1/N = 1/1.  Entropy will take a value of 0.

        Returns
        -------
        np.ndarray
            Diversity index per column of X

        See Also
        --------
        numpy.histogram
        """

        # hist_counts,_ = _get_counts(subset_mask)
        hist_counts, _ = _get_counts(self.data, self.names, self.is_categorical, subset_mask)
        # normalize by global counts, not classwise counts
        num_bins = _get_num_bins(self.data, self.names, self.is_categorical)

        ev_index = np.empty(self.num_factors)
        # loop over columns for convenience
        for col, cnts in enumerate(hist_counts.values()):
            # relative frequencies
            p_i = cnts / cnts.sum()
            # inverse Simpson index normalized by (number of bins)
            ev_index[col] = 1 / np.sum(p_i**2) / num_bins[col]

        return ev_index

    def _diversity_shannon(self, subset_mask: np.ndarray = np.empty(shape=0)) -> np.ndarray:
        """
        Compute diversity for discrete/categorical variables and, through standard
        histogram binning, for continuous variables.

        We define diversity as a normalized form of the Shannon entropy.

        diversity = 1 implies that samples are evenly distributed across a particular factor
        diversity = 0 implies that all samples belong to one category/bin

        Parameters
        ----------
        subset_mask: Optional[np.ndarray[bool]]
            Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts

        Notes
        -----
        - For continuous variables, histogram bins are chosen automatically.  See
        numpy.histogram for details.

        Returns
        -------
        diversity_index: np.ndarray
            Diversity index per column of X

        See Also
        --------
        numpy.histogram
        """

        # entropy computed using global auto bins so that we can properly normalize
        ent_unnormalized = _entropy(
            self.data, self.names, self.is_categorical, normalized=False, subset_mask=subset_mask
        )
        # normalize by global counts rather than classwise counts
        num_bins = _get_num_bins(self.data, self.names, is_categorical=self.is_categorical, subset_mask=subset_mask)
        return ent_unnormalized / np.log(num_bins)


class DiversityClasswise(BaseDiversityMetric):
    """
    Classwise diversity index: evenness of the distribution of metadata factors
    per class.

    Parameters
    ----------
    metric: str
        string variable indicating which diversity index should be used.
        Permissible values include "simpson" and "shannon"

    Attributes
    ----------
    metric: str
        string variable indicating which diversity index should be used.
        Permissible values include "simpson" and "shannon"
    data: np.ndarray
        Array of metadata factors; string variables are converted to integers
    names: List[str]
        List of the names of metadata factor variables
    is_categorical: List
        List of boolean flags for categorical features.  Mutual information is
        computed differently for categorical/discrete and continuous variables
    num_factors: int
        Number of metadata factors in the dataset
    num_samples: int
        Number of samples in the dataset

    """

    def __init__(self, metric="simpson"):
        super().__init__(metric=metric)

    def compute(self):
        """
        Compute diversity for discrete/categorical variables and, through standard
        histogram binning, for continuous variables.

        We define diversity as a normalized form of the inverse Simpson diversity
        index.

        diversity = 1 implies that samples are evenly distributed across a particular factor
        diversity = 1/num_categories implies that all samples belong to one category/bin

        Notes
        -----
        For continuous variables, histogram bins are chosen automatically.  See
            numpy.histogram for details.
        The expression is undefined for q=1, but it approaches the Shannon entropy
            in the limit.
        If there is only one category, the diversity index takes a value of 1 =
            1/N = 1/1.  Entropy will take a value of 0.

        Returns
        -------
        np.ndarray
            Diversity index [n_class x n_factor]

        See Also
        --------
        diversity_simpson
        diversity_shannon
        numpy.histogram
        """
        self._collect_data()

        class_idx = self.names.index("class_label")
        class_labels = self.data[:, class_idx]

        u_classes = np.unique(class_labels)
        num_factors = len(self.names)
        diversity = np.empty((len(u_classes), num_factors))
        diversity[:] = np.nan
        for idx, cls in enumerate(u_classes):
            subset_mask = class_labels == cls
            if self.metric == "simpson":
                diversity[idx, :] = self._diversity_simpson(subset_mask)
            elif self.metric == "shannon":
                diversity[idx, :] = self._diversity_shannon(subset_mask)
        div_no_class = np.concatenate((diversity[:, :class_idx], diversity[:, (class_idx + 1) :]), axis=1)
        return div_no_class


class Diversity(BaseDiversityMetric):
    """
    Diversity index: evenness of the distribution of metadata factors to
    identify imbalance or undersampled data categories.

    Parameters
    ----------
    metric: str
        string variable indicating which diversity index should be used.
        Permissible values include "simpson" and "shannon"

    Attributes
    ----------
    metric: str
        string variable indicating which diversity index should be used.
        Permissible values include "simpson" and "shannon"
    data: np.ndarray
        Array of metadata factors; string variables are converted to integers
    names: List[str]
        List of the names of metadata factor variables
    is_categorical: List
        List of boolean flags for categorical features.  Mutual information is
        computed differently for categorical/discrete and continuous variables
    num_factors: int
        Number of metadata factors in the dataset
    num_samples: int
        Number of samples in the dataset
    """

    def __init__(self, metric="simpson"):
        super().__init__(metric=metric)

    def compute(self):
        """
        Compute diversity for discrete/categorical variables and, through standard
        histogram binning, for continuous variables.

        diversity = 1 implies that samples are evenly distributed across a particular factor
        diversity = 0 implies that all samples belong to one category/bin

        Parameters
        ----------
        metric: str
            The type of diversity index to return,  currently ["simpson",
            "shannon"]

        Notes
        -----
        - For continuous variables, histogram bins are chosen automatically.  See
        numpy.histogram for details.

        Returns
        -------
        diversity_index: np.ndarray
            Diversity index per column of self.data or each factor in self.names

        See Also
        --------
        numpy.histogram

        """
        self._collect_data()
        if self.metric.lower() == "simpson":
            return self._diversity_simpson()
        elif self.metric.lower() == "shannon":
            return self._diversity_shannon()
