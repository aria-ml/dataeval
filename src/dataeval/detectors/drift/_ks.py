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
    """:term:`Drift` detector employing the :term:`Kolmogorov-Smirnov (KS) \
    distribution<Kolmogorov-Smirnov (K-S) test>` test.

    Detects distributional changes by measuring the maximum distance between
    empirical cumulative distribution functions of reference and test datasets.
    For multivariate data, applies KS test independently to each feature
    and aggregates results using multiple testing correction.

    The Kolmogorov-Smirnov test is particularly sensitive to differences in
    the middle portions of distributions but has reduced power in the tails
    where cumulative distribution functions are constrained near 0 and 1.

    Parameters
    ----------
    data : Embeddings or Array
        Reference dataset used as baseline distribution for drift detection.
        Should represent the expected data distribution.
    p_val : float, default 0.05
        Significance threshold for drift detection, between 0 and 1. 
        Default 0.05 limits false drift alerts to 5% when no drift exists (Type I error rate).
    update_strategy : UpdateStrategy or None, default None
        Strategy for updating reference data when new data arrives.
        When None, reference data remains fixed throughout detection.
    correction : "bonferroni" or "fdr", default "bonferroni"
        Multiple testing correction method for multivariate drift detection.
        "bonferroni" provides conservative family-wise error control by
        dividing significance threshold by number of features.
        "fdr" uses Benjamini-Hochberg procedure for less conservative control.
        Default "bonferroni" minimizes false positive drift detections.
    alternative : "two-sided", "less" or "greater", default "two-sided"
        Alternative hypothesis for the statistical test. "two-sided" detects
        any distributional difference. "less" tests if test distribution is
        stochastically smaller. "greater" tests if test distribution is
        stochastically larger. Default "two-sided" provides most general
        drift detection without directional assumptions.
    n_features : int | None, default None
        Number of features to analyze in univariate tests.
        When None, automatically inferred from the flattened shape of first data sample.

    Example
    -------
    Basic drift detection with image embeddings:

    >>> from dataeval.data import Embeddings
    >>> train_emb = Embeddings(train_images, model=encoder, batch_size=64)
    >>> drift_detector = DriftKS(train_emb)
    
    Test incoming images for distributional drift
    
    >>> result = drift_detector.predict(test_images)
    >>> print(f"Drift detected: {result.drifted}")
    Drift detected: True

    >>> print(f"Mean KS statistic: {result.distance:.4f}")
    Mean KS statistic: 0.8750
    
    Detect if test data has systematically higher values
    
    >>> drift_greater = DriftKS(train_emb, alternative="greater")
    >>> result = drift_greater.predict(test_images)
    
    Using different correction methods

    >>> drift_fdr = DriftKS(train_emb, correction="fdr", p_val=0.1)
    >>> result = drift_fdr.predict(test_images)
    
    Access feature-level results
    
    >>> n_features = result.feature_drift
    >>> print(f"Features showing drift: {n_features.sum()} / {len(n_features)}")
    Features showing drift: 576 / 576
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
