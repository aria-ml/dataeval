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
    """:term:`Drift` detector using the :term:`Cramér-von Mises (CVM) Test`.

    Detects distributional changes in continuous data by comparing empirical
    cumulative distribution functions between reference and test datasets.
    For multivariate data, applies CVM test independently to each feature
    and aggregates results using either the Bonferroni or
    :term:`False Discovery Rate (FDR)` correction.

    The CVM test is particularly effective at detecting subtle
    distributional shifts throughout the entire domain, providing higher
    power than Kolmogorov-Smirnov for many types of drift.

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
    n_features : int or None, default None
        Number of features to analyze in univariate tests.
        When None, automatically inferred from the flattened shape of first data sample.

    Example
    -------
    Basic drift detection with image embeddings

    >>> from dataeval.data import Embeddings
    >>> train_emb = Embeddings(train_images, model=encoder, batch_size=64)
    >>> drift_detector = DriftCVM(train_emb)

    Test incoming images for distributional drift

    >>> result = drift_detector.predict(test_images)
    >>> print(f"Drift detected: {result.drifted}")
    Drift detected: True

    >>> print(f"Mean CVM statistic: {result.distance:.4f}")
    Mean CVM statistic: 24.1325

    Using different correction methods

    >>> drift_fdr = DriftCVM(train_emb, correction="fdr", p_val=0.1)
    >>> result = drift_fdr.predict(test_images)

    Access feature level results

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
