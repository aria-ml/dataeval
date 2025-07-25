"""
This module contains the implementation of the
FR Test Statistic based estimate and the
KNN based estimate for the :term:`Bayes error rate<Bayes Error Rate (BER)>`

Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4)
https://arxiv.org/abs/1811.06419

"""

from __future__ import annotations

__all__ = []

from typing import Literal

import numpy as np

from dataeval.core._ber import ber_knn, ber_mst
from dataeval.outputs import BEROutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import Array
from dataeval.utils._array import as_numpy, ensure_embeddings
from dataeval.utils._method import get_method

_BER_FN_MAP = {"KNN": ber_knn, "MST": ber_mst}


@set_metadata
def ber(embeddings: Array, labels: Array, k: int = 1, method: Literal["KNN", "MST"] = "KNN") -> BEROutput:
    """
    An estimator for Multi-class :term:`Bayes error rate<Bayes Error Rate (BER)>` \
    using FR or KNN test statistic basis.

    Parameters
    ----------
    embeddings : ArrayLike (N, ... )
        Array of image :term:`embeddings<Embeddings>`
    labels : ArrayLike (N, 1)
        Array of labels for each image
    k : int, default 1
        Number of nearest neighbors for KNN estimator -- ignored by MST estimator
    method : Literal["KNN", "MST"], default "KNN"
        Method to use when estimating the Bayes error rate

    Returns
    -------
    BEROutput
        The upper and lower bounds of the Bayes Error Rate

    References
    ----------
    [1] `Learning to Bound the Multi-class Bayes Error (Th. 3 and Th. 4) <https://arxiv.org/abs/1811.06419>`_

    Examples
    --------
    >>> import sklearn.datasets as dsets
    >>> from dataeval.metrics.estimators import ber

    >>> images, labels = dsets.make_blobs(n_samples=50, centers=2, n_features=2, random_state=0)

    >>> ber(images, labels)
    BEROutput(ber=0.04, ber_lower=0.020416847668728033)
    """
    ber_fn = get_method(_BER_FN_MAP, method)
    X = ensure_embeddings(embeddings, dtype=np.float64)
    y = as_numpy(labels)
    upper, lower = ber_fn(X, y, k)
    return BEROutput(upper, lower)
