"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

__all__ = []

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import cramervonmises_2samp

from dataeval.data._embeddings import Embeddings
from dataeval.detectors.drift._base import BaseDriftUnivariate, UpdateStrategy
from dataeval.typing import Array


class DriftCVM(BaseDriftUnivariate):
    """
    :term:`Drift` detector employing the :term:`Cramér-von Mises (CVM) Drift Detection` test.

    The CVM test detects changes in the distribution of continuous
    univariate data. For multivariate data, a separate CVM test is applied to each
    feature, and the obtained p-values are aggregated via the Bonferroni or
    :term:`False Discovery Rate (FDR)` corrections.

    Parameters
    ----------
    data : Embeddings or Array
        Data used as reference distribution.
    p_val : float or None, default 0.05
        :term:`p-value<P-Value>` used for significance of the statistical test for each feature.
        If the FDR correction method is used, this corresponds to the acceptable
        q-value.
    update_strategy : UpdateStrategy or None, default None
        Reference data can optionally be updated using an UpdateStrategy class. Update
        using the last n instances seen by the detector with LastSeenUpdateStrategy
        or via reservoir sampling with ReservoirSamplingUpdateStrategy.
    correction : "bonferroni" or "fdr", default "bonferroni"
        Correction type for multivariate data. Either 'bonferroni' or 'fdr' (False
        Discovery Rate).
    n_features : int or None, default None
        Number of features used in the univariate drift tests. If not provided, it will
        be inferred from the data.


    Example
    -------
    >>> from dataeval.data import Embeddings

    Use Embeddings to encode images before testing for drift

    >>> train_emb = Embeddings(train_images, model=encoder, batch_size=64)
    >>> drift = DriftCVM(train_emb)

    Test incoming images for drift

    >>> drift.predict(test_images).drifted
    True
    """

    def __init__(
        self,
        data: Embeddings | Array,
        p_val: float = 0.05,
        update_strategy: UpdateStrategy | None = None,
        correction: Literal["bonferroni", "fdr"] = "bonferroni",
        n_features: int | None = None,
    ) -> None:
        super().__init__(
            data=data,
            p_val=p_val,
            update_strategy=update_strategy,
            correction=correction,
            n_features=n_features,
        )

    def _score_fn(self, x: NDArray[np.float32], y: NDArray[np.float32]) -> tuple[np.float32, np.float32]:
        result = cramervonmises_2samp(x, y, method="auto")
        return np.float32(result.statistic), np.float32(result.pvalue)
