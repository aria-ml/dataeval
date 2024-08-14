from torchmetrics import Metric
from typing import Dict, List

import torch

import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# from maite.protocols import ArrayLike


class BaseBiasMetric(Metric):
    """
    Class BiasMetric:
    def __init__(self, bias_metrics = List["Coverage", "Parity", etc])
        self.bias_metrics = bias_metrics

    def update(self, labels, metadata):
        self.labels += labels
        self.metadata += metadata

    def compute(self):
        results = {}
        for bias_metric in self.bias_metrics:
        result = bias_metric.forward(self.labels, self.metadata)
        results[bias_metric.str()] = result
        return results
    """


    def __init__(self):
        super().__init__()
        # self.bias_metrics = bias_metrics
        self.names = [] #list(metadata_dict.keys())
        self.data = np.empty(0) # np.stack(list(metadata_dict.values()), axis=-1)
        # verify again that insertion order is the same between data and categorical labels
        self.is_categorical = [] # {k: metadata_type[k] for k in self.names}

        self.add_state("metadata", default=[], dist_reduce_fx="cat")
        self.add_state("class_label", default=[], dist_reduce_fx="cat")

        self.entropy = np.empty(shape=0)
        self.hist_counts = {}
        self.hist_bins = {}

        self.num_factors = 0 # len(self.names)
        self.num_samples = 0 # self.data.shape[0]

    def update(self, class_label, metadata):
        self.metadata.extend(metadata)
        self.class_label.append(class_label)

    # def compute(self):
    #     self._collect_data()
    #     # return empty to satisfy linter
    #     return np.empty(0)

    def _collect_data(self):
        metadata_dict = {"class_label": torch.cat(self.class_label).numpy()}
        metadata_dict = {**metadata_dict, **list_to_dict(self.metadata)}

        # convert string variables to int
        metadata_dict = str_to_int(metadata_dict)
        self.data = np.stack(list(metadata_dict.values()), axis=-1)
        self.names = list(metadata_dict.keys())

        self.is_categorical = [infer_categorical(metadata_dict[var], 0.25)[0] for var in self.names]

        self.num_factors = len(self.names)
        self.num_samples = len(self.metadata)

    def _get_counts(self, subset_mask: np.ndarray = np.empty(shape=0)) -> Dict:
        """
        Initialize dictionary of histogram counts --- treat categorical values
        as histogram bins.

        It is necessary to store global histogram bins in order to properly
        normalize classwise diveristy index.

        Parameters
        ----------
        subset_mask: Optional[np.ndarray[bool]]
            Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts

        Returns
        -------
        counts: Dict
            histogram counts per metadata factor in `factors`.  Each
            factor will have a different number of bins.  Counts get reused
            across metrics, so hist_counts are cached but only if computed
            globally, i.e. without masked samples.
        """
        if len(self.hist_counts) > 0 and len(subset_mask) == 0:
            # global precomputed hist_counts
            return self.hist_counts

        hist_counts, hist_bins = {}, {}
        # np.where needed to satisfy linter
        mask = np.where(subset_mask if len(subset_mask) > 0 else np.ones(self.data.shape[0], dtype=bool))

        # loop over columns for convenience
        for cdx, fn in enumerate(self.names):
            # linter doesn't like double indexing
            col_data = self.data[mask, cdx].squeeze()
            if self.is_categorical[cdx]:
                # if discrete, use unique values as bins
                bins, cnts = np.unique(col_data, return_counts=True)
            else:
                bins = self.hist_bins.get(fn, "auto")
                cnts, bins = np.histogram(col_data, bins=bins, density=True)

            hist_counts[fn] = cnts
            hist_bins[fn] = bins

        if len(subset_mask) == 0:
            # save if using all values
            self.hist_counts = hist_counts
            self.hist_bins = hist_bins
        return hist_counts

    def _get_num_bins(self, subset_mask: np.ndarray = np.empty(shape=0)) -> np.ndarray:
        """
        Number of bins or unique values for each metadata factor, used to
        normalize entropy/diversity.

        Parameters
        ----------
        subset_mask: Optional[np.ndarray[bool]]
            Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts
        """
        # likely cached
        hist_counts = self._get_counts(subset_mask)
        num_bins = np.empty(len(hist_counts))
        for idx, cnts in enumerate(hist_counts.values()):
            num_bins[idx] = len(cnts)
        if len(subset_mask) == 0:
            self.num_bins = num_bins
        return num_bins

    def _entropy(self, normalized: bool = False, subset_mask: np.ndarray = np.empty(shape=0)) -> np.ndarray:
        """
        Compute entropy for discrete/categorical variables and, through standard
        histogram binning, for continuous variables.


        Parameters
        ----------
        normalized: bool
            Flag that determines whether or not to normalize entropy by log(num_bins)
        subset_mask: Optional[np.ndarray[bool]]
            Boolean mask of samples to bin (e.g. when computing per class).  True -> include in histogram counts


        Notes
        -----
        For continuous variables, histogram bins are chosen automatically.  See
        numpy.histogram for details.

        Returns
        -------
        ent: np.ndarray[float]
            Entropy estimate per column of X

        See Also
        --------
        numpy.histogram
        scipy.stats.entropy
        """
        if len(self.entropy) > 0 and len(subset_mask) == 0:
            return self.entropy

        if len(subset_mask) > 0 and len(self.hist_bins) == 0:
            # if we haven't populated global bins
            _ = self._get_counts()

        ev_index = np.empty(self.num_factors)
        # loop over columns for convenience
        hist_counts = self._get_counts(subset_mask)
        for col, cnts in enumerate(hist_counts.values()):
            # entropy in nats, normalizes counts
            ev_index[col] = entropy(cnts)
            if normalized:
                if len(cnts) == 1:
                    # log(0)
                    ev_index[col] = 0
                else:
                    ev_index[col] /= np.log(len(cnts))

        if len(subset_mask) == 0:
            self.entropy = ev_index
        return ev_index

class Balance(BaseBiasMetric):
    def __init__(self):
        super().__init__()

    def compute(self, num_neighbors: int = 5) -> NDArray:
        """
        Mutual information (MI) between factors (class label, metadata, label/image properties)

        Parameters
        ----------
            num_neighbors: int
                Number of nearest neighbors to use for computing MI between discrete
                and continuous variables.

            Computes MI from class properties

        Notes
        -----
        - We use mutual_info_classif from sklearn since class label is categorical
        - mutual_info_classif outputs are consistent up to O(1e-4) and depend on
            a random seed
        - MI is computed differently for categorical and continuous variables,
            so we have to specify---see self.is_categorical

        Returns
        -------
        NDArray

        See Also
        --------
        sklearn.feature_selection.mutual_info_classif
        sklearn.feature_selection.mutual_info_regression
        sklearn.metrics.mutual_info_score
        """
        self._collect_data()
        # initialize
        mi = np.empty((self.num_factors, self.num_factors))
        mi[:] = np.nan

        # cat_mask = np.stack(list(self.is_categorical.values()), -1)
        for idx, tgt_var in enumerate(self.names):
            tgt = self.data[:, idx]

            if self.is_categorical[idx]:
                # categorical target
                mi[idx, :] = mutual_info_classif(self.data, tgt, discrete_features=self.is_categorical, n_neighbors=num_neighbors)
            else:
                # continuous variables
                mi[idx, :] = mutual_info_regression(
                    self.data, tgt, discrete_features=self.is_categorical, n_neighbors=num_neighbors
                )

        ent_all = self._entropy(normalized=False)
        norm_factor = 0.5 * np.add.outer(ent_all, ent_all) + 1e-6
        # in principle MI should be symmetric, but it is not in practice.
        nmi = 0.5 * (mi + mi.T) / norm_factor

        return nmi

class BalanceClasswise(BaseBiasMetric):
    def __init__(self):
        super().__init__()

    def compute(self, num_neighbors: int = 5) -> NDArray:
        """
        Compute mutual information of metadata factors (class label, metadata,
        label/image properties) with individual classes.

        Parameters
        ----------
            num_neighbors: int
                Number of nearest neighbors to use for computing MI between discrete
                and continuous variables.

            Factor data and type (categorical, discrete, continuous) are managed
            by the class.

        Notes
        -----
        - we use mutual_info_classif from sklearn since class label is categorical
        - mutual_info_classif outputs are consistent up to O(1e-4) and depend on
            a random seed
        - MI is computed differently for categorical and continuous variables,
            so we have to specify with self.is_categorical.

        Returns
        -------
        np.ndarray

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
        cat_mask = np.concatenate((self.is_categorical[:class_idx], self.is_categorical[(class_idx + 1) :]), axis=0).astype(int)

        # classification MI for discrete/categorical features
        for idx, cls in enumerate(u_cls):
            tgt = class_data == cls
            # units: nat
            mi[idx, :] = mutual_info_classif(data_no_class, tgt, discrete_features=cat_mask, n_neighbors=num_neighbors)

        # let this recompute for all features including class label
        ent_all = self._entropy()
        ent_tgt = ent_all[class_idx]
        ent_all = np.concatenate((ent_all[:class_idx], ent_all[(class_idx + 1) :]), axis=0)
        norm_factor = 0.5 * np.add.outer(ent_tgt, ent_all) + 1e-6
        nmi = mi / norm_factor
        return nmi


class BaseDiversityMetric(BaseBiasMetric):
    def __init__(self, metric: str = "simpson"):
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
        - For continuous variables, histogram bins are chosen automatically.  See
        numpy.histogram for details.
        - The expression is undefined for q=1, but it approaches the Shannon entropy
        in the limit.
        - If there is only one category, the diversity index takes a value of 1 =
        1/N = 1/1.  Entropy will take a value of 0.

        Returns
        -------
        ev_index: np.ndarray
            Diversity index per column of X

        See Also
        --------
        numpy.histogram

        """

        hist_counts = self._get_counts(subset_mask)
        # normalize by global counts, not classwise counts
        num_bins = self._get_num_bins()

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
        ent_unnormalized = self._entropy(normalized=False, subset_mask=subset_mask)
        # normalize by global counts rather than classwise counts
        num_bins = self._get_num_bins()
        return ent_unnormalized / np.log(num_bins)

class DiversityClasswise(BaseDiversityMetric):

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

        Parameters
        ----------
        class_labels: np.nadarray
            numpy array of class labels
        metric: str["simpson", "shannon"]
            which diversity metric to use

        Notes
        -----
        - For continuous variables, histogram bins are chosen automatically.  See
        numpy.histogram for details.
        - The expression is undefined for q=1, but it approaches the Shannon entropy
        in the limit.
        - If there is only one category, the diversity index takes a value of 1 =
        1/N = 1/1.  Entropy will take a value of 0.

        Returns
        -------
        ev_index: np.ndarray
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
            # filt_factors = {k: v[inds] for k, v in factors.items()}
            if self.metric == "simpson":
                diversity[idx, :] = self._diversity_simpson(subset_mask)
            elif self.metric == "shannon":
                diversity[idx, :] = self._diversity_shannon(subset_mask)
        div_no_class = np.concatenate((diversity[:, :class_idx], diversity[:, (class_idx + 1) :]), axis=1)
        return div_no_class

class Diversity(BaseDiversityMetric):

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
        raise ValueError("The lengths of each entry in the dictionary are not equal."
                             f" Found lengths {lengths}")


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
    d: Dict
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

def list_to_dict(lod : List[Dict]):
    return {k: np.array([dic[k] for dic in lod]) for k in lod[0]}

def infer_categorical(X, threshold: float = 0.5) -> np.ndarray:
    """
    Compute fraction of feature values that are unique --- intended to be used
    for inferring whether variables are categorical.

    Notes:
        - Not tested yet
    """
    if X.ndim == 1:
        X = np.expand_dims(X, axis=1)
    num_samples = X.shape[0]
    pct_unique = np.empty(X.shape[1])
    for col in range(X.shape[1]):  # type: ignore
        uvals = np.unique(X[:, col], axis=0)
        pct_unique[col] = len(uvals) / num_samples
    return pct_unique < threshold
