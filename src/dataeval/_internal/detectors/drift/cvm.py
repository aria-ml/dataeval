"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import cramervonmises_2samp

from dataeval._internal.interop import to_numpy

from .base import BaseDriftUnivariate, UpdateStrategy, preprocess_x


class DriftCVM(BaseDriftUnivariate):
    """
    Drift detector employing the Cramér-von Mises (CVM) distribution test.

    The CVM test detects changes in the distribution of continuous
    univariate data. For multivariate data, a separate CVM test is applied to each
    feature, and the obtained p-values are aggregated via the Bonferroni or
    False Discovery Rate (FDR) corrections.

    Parameters
    ----------
    x_ref : ArrayLike
        Data used as reference distribution.
    p_val : float | None, default 0.05
        p-value used for significance of the statistical test for each feature.
        If the FDR correction method is used, this corresponds to the acceptable
        q-value.
    x_ref_preprocessed : bool, default False
        Whether the given reference data ``x_ref`` has been preprocessed yet.
        If ``True``, only the test data ``x`` will be preprocessed at prediction time.
        If ``False``, the reference data will also be preprocessed.
    update_x_ref : UpdateStrategy | None, default None
        Reference data can optionally be updated using an UpdateStrategy class. Update
        using the last n instances seen by the detector with LastSeenUpdateStrategy
        or via reservoir sampling with ReservoirSamplingUpdateStrategy.
    preprocess_fn : Callable | None, default None
        Function to preprocess the data before computing the data drift metrics.
        Typically a dimensionality reduction technique.
    correction : "bonferroni" | "fdr", default "bonferroni"
        Correction type for multivariate data. Either 'bonferroni' or 'fdr' (False
        Discovery Rate).
    n_features : int | None, default None
        Number of features used in the statistical test. No need to pass it if no
        preprocessing takes place. In case of a preprocessing step, this can also
        be inferred automatically but could be more expensive to compute.
    """

    def __init__(
        self,
        x_ref: ArrayLike,
        p_val: float = 0.05,
        x_ref_preprocessed: bool = False,
        update_x_ref: UpdateStrategy | None = None,
        preprocess_fn: Callable[[ArrayLike], ArrayLike] | None = None,
        correction: Literal["bonferroni", "fdr"] = "bonferroni",
        n_features: int | None = None,
    ) -> None:
        super().__init__(
            x_ref=x_ref,
            p_val=p_val,
            x_ref_preprocessed=x_ref_preprocessed,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,
            correction=correction,
            n_features=n_features,
        )

    @preprocess_x
    def score(self, x: ArrayLike) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Performs the two-sample Cramér-von Mises test(s), computing the p-value and
        test statistic per feature.

        Parameters
        ----------
        x : ArrayLike
            Batch of instances.

        Returns
        -------
        tuple[NDArray, NDArray]
            Feature level p-values and CVM statistic
        """
        x_np = to_numpy(x)
        x_np = x_np.reshape(x_np.shape[0], -1)
        x_ref = self.x_ref.reshape(self.x_ref.shape[0], -1)
        p_val = np.zeros(self.n_features, dtype=np.float32)
        dist = np.zeros_like(p_val)
        for f in range(self.n_features):
            result = cramervonmises_2samp(x_ref[:, f], x_np[:, f], method="auto")
            p_val[f], dist[f] = result.pvalue, result.statistic
        return p_val, dist
