from typing import Callable, Dict, List, Literal, Optional, Sequence

import numpy as np

from dataeval._internal.metrics.base import EvaluateMixin, MethodsMixin
from dataeval._internal.metrics.functional import diversity_shannon, diversity_simpson, preprocess_metadata

_METHODS = Literal["simpson", "shannon"]
_FUNCTION = Callable[[np.ndarray, List[str], List[bool], Optional[np.ndarray]], np.ndarray]


class BaseDiversityMetric(EvaluateMixin, MethodsMixin[_METHODS, _FUNCTION]):
    """
    Base class for Diversity and ClasswiseDiversity metrics.

    Parameters
    ----------
    method: str
        string variable indicating which diversity index should be used.
        Permissible values include "simpson" and "shannon"
    """

    def __init__(self, method: _METHODS):
        self._set_method(method)

    @classmethod
    def _methods(cls) -> Dict[str, _FUNCTION]:
        return {"simpson": diversity_simpson, "shannon": diversity_shannon}


class DiversityClasswise(BaseDiversityMetric):
    """
    Classwise diversity index: evenness of the distribution of metadata factors
    per class.

    Parameters
    ----------
    method: str
        string variable indicating which diversity index should be used.
        Permissible values include "simpson" and "shannon"

    Attributes
    ----------
    method: str
        string variable indicating which diversity index should be used.
        Permissible values include "simpson" and "shannon"
    """

    def __init__(self, method: _METHODS = "simpson"):
        super().__init__(method=method)

    def evaluate(self, class_labels: Sequence[int], metadata: List[Dict]) -> np.ndarray:
        """
        Compute diversity for discrete/categorical variables and, through standard
        histogram binning, for continuous variables.

        We define diversity as a normalized form of the inverse Simpson diversity
        index.

        diversity = 1 implies that samples are evenly distributed across a particular factor
        diversity = 1/num_categories implies that all samples belong to one category/bin

        Parameters
        ----------
        class_labels: Sequence[int]
            List of class labels for each image
        metadata: List[Dict]
            List of metadata factors for each image

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
        data, names, is_categorical = preprocess_metadata(class_labels, metadata)
        class_idx = names.index("class_label")
        class_lbl = data[:, class_idx]

        u_classes = np.unique(class_lbl)
        num_factors = len(names)
        diversity = np.empty((len(u_classes), num_factors))
        diversity[:] = np.nan
        for idx, cls in enumerate(u_classes):
            subset_mask = class_lbl == cls
            diversity[idx, :] = self._method(data, names, is_categorical, subset_mask)
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
    """

    def __init__(self, method: _METHODS = "simpson"):
        super().__init__(method=method)

    def evaluate(self, class_labels: Sequence[int], metadata: List[Dict]) -> np.ndarray:
        """
        Compute diversity for discrete/categorical variables and, through standard
        histogram binning, for continuous variables.

        diversity = 1 implies that samples are evenly distributed across a particular factor
        diversity = 0 implies that all samples belong to one category/bin

        Parameters
        ----------
        class_labels: Sequence[int]
            List of class labels for each image
        metadata: List[Dict]
            List of metadata factors for each image

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
        data, names, is_categorical = preprocess_metadata(class_labels, metadata)
        return self._method(data, names, is_categorical, None)
