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
    data : ArrayLike
        Images or image embeddings
    labels : ArrayLike
        Label for each image or image embedding
    method : Literal["MST", "KNN"], default "KNN"
        Method to use when estimating the Bayes error rate
    k : int, default 1
        number of nearest neighbors for KNN estimator -- ignored by MST estimator


    See Also
    --------
    `Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4) <https://arxiv.org/abs/1811.06419>`_

    """

    def __init__(self, data: ArrayLike, labels: ArrayLike, method: _METHODS = "KNN", k: int = 1):
        super().__init__(data=arraylike_to_numpy(data), labels=arraylike_to_numpy(labels), method=method, k=k)


class BERDataset(BER):
    """
    An estimator for Multi-class Bayes Error Rate using FR or KNN test statistic basis

    Parameters
    ----------
    dataset : `maite.protocols.image_classification.Dataset`
        Dataset of images and labels
    method : Literal["MST", "KNN"], default "KNN"
        Method to use when estimating the Bayes error rate
    k : int, default 1
        number of nearest neighbors for KNN estimator -- ignored by MST estimator


    See Also
    --------
    `Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4) <https://arxiv.org/abs/1811.06419>`_

    """

    def __init__(self, dataset: ic.Dataset, method: _METHODS = "KNN", k: int = 1):
        images, labels = extract_to_numpy(dataset=dataset)
        super().__init__(np.asarray(images), np.asarray(labels), method=method, k=k)
