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
from scipy.stats import ks_2samp

from dataeval.data._embeddings import Embeddings
from dataeval.detectors.drift._base import BaseDriftUnivariate, UpdateStrategy
from dataeval.typing import Array


class DriftKS(BaseDriftUnivariate):
    """
    :term:`Drift` detector employing the :term:`Kolmogorov-Smirnov (KS) \
    distribution<Kolmogorov-Smirnov (K-S) test>` test.

    The KS test detects changes in the maximum distance between two data
    distributions with Bonferroni or :term:`False Discovery Rate (FDR)` correction
    for multivariate data.

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
    alternative : "two-sided", "less" or "greater", default "two-sided"
        Defines the alternative hypothesis. Options are 'two-sided', 'less' or
        'greater'.
    n_features : int | None, default None
        Number of features used in the univariate drift tests. If not provided, it will
        be inferred from the data.

    Example
    -------
    >>> from dataeval.data import Embeddings

    Use Embeddings to encode images before testing for drift

    >>> train_emb = Embeddings(train_images, model=encoder, batch_size=64)
    >>> drift = DriftKS(train_emb)

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
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        n_features: int | None = None,
    ) -> None:
        super().__init__(
            data=data,
            p_val=p_val,
            update_strategy=update_strategy,
            correction=correction,
            n_features=n_features,
        )

        # Other attributes
        self.alternative = alternative

    def _score_fn(self, x: NDArray[np.float32], y: NDArray[np.float32]) -> tuple[np.float32, np.float32]:
        return ks_2samp(x, y, alternative=self.alternative, method="exact")
