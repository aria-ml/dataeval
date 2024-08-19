from typing import Dict

import maite.protocols.image_classification as ic
import numpy as np
from maite.protocols import ArrayLike

from dataeval._internal.metrics.ber import _METHODS, BER

from .utils import arraylike_to_numpy, extract_to_numpy


class BERArrayLike(BER):
    """
    An estimator for Multi-class Bayes Error Rate using FR or KNN test statistic basis

    Parameters
    ----------
    method : Literal["MST", "KNN"], default "KNN"
        Method to use when estimating the Bayes error rate
    k : int, default 1
        number of nearest neighbors for KNN estimator -- ignored by MST estimator

    See Also
    --------
    `Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4) <https://arxiv.org/abs/1811.06419>`_

    """

    def __init__(self, method: _METHODS = "KNN", k: int = 1):
        super().__init__(method=method, k=k)

    def evaluate(self, data: ArrayLike, labels: ArrayLike) -> Dict[str, float]:
        """
        Parameters
        ----------
        data : ArrayLike
            Images or image embeddings
        labels : ArrayLike
            Label for each image or image embedding

        Returns
        -------
        Dict[str, float]
            ber : float
                The estimated lower bounds of the Bayes Error Rate
            ber_lower : float
                The estimated upper bounds of the Bayes Error Rate

        Raises
        ------
        ValueError
            If unique classes M < 2
        """
        return super().evaluate(data=arraylike_to_numpy(data), labels=arraylike_to_numpy(labels))


class BERDataset(BER):
    """
    An estimator for Multi-class Bayes Error Rate using FR or KNN test statistic basis

    Parameters
    ----------
    method : Literal["MST", "KNN"], default "KNN"
        Method to use when estimating the Bayes error rate
    k : int, default 1
        number of nearest neighbors for KNN estimator -- ignored by MST estimator


    See Also
    --------
    `Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4) <https://arxiv.org/abs/1811.06419>`_

    """

    def __init__(self, method: _METHODS = "KNN", k: int = 1):
        super().__init__(method=method, k=k)

    def evaluate_dataset(self, data: ic.Dataset) -> Dict[str, float]:
        """
        Parameters
        ----------
        data : `maite.protocols.image_classification.Dataset`
            Dataset of images and labels
        labels: None
            Unused (labels are a component of the Dataset)

        Returns
        -------
        Dict[str, float]
            ber : float
                The estimated lower bounds of the Bayes Error Rate
            ber_lower : float
                The estimated upper bounds of the Bayes Error Rate

        Raises
        ------
        ValueError
            If unique classes M < 2
        """
        images, labels = extract_to_numpy(dataset=data)
        return super().evaluate(np.asarray(images), np.asarray(labels))
