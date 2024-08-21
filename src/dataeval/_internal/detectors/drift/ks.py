"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from typing import Callable, Literal, Optional, Tuple

import numpy as np
from scipy.stats import ks_2samp

from dataeval._internal.interop import ArrayLike, to_numpy

from .base import BaseUnivariateDrift, UpdateStrategy, preprocess_x


class DriftKS(BaseUnivariateDrift):
    """
    Kolmogorov-Smirnov (K-S) data drift detector with Bonferroni or False Discovery
    Rate (FDR) correction for multivariate data.

    Parameters
    ----------
    x_ref : np.ndarray
        Data used as reference distribution.
    p_val : float, default 0.05
        p-value used for significance of the statistical test for each feature.
        If the FDR correction method is used, this corresponds to the acceptable
        q-value.
    x_ref_preprocessed : bool, default False
        Whether the given reference data `x_ref` has been preprocessed yet. If
        `x_ref_preprocessed=True`, only the test data `x` will be preprocessed at
        prediction time. If `x_ref_preprocessed=False`, the reference data will also
        be preprocessed.
    update_x_ref : Optional[UpdateStrategy], default None
        Reference data can optionally be updated using an UpdateStrategy class. Update
        using the last n instances seen by the detector with
        :py:class:`dataeval.detectors.LastSeenUpdateStrategy`
        or via reservoir sampling with
        :py:class:`dataeval.detectors.ReservoirSamplingUpdateStrategy`.
    preprocess_fn : Optional[Callable[[np.ndarray], np.ndarray]], default None
        Function to preprocess the data before computing the data drift metrics.
        Typically a dimensionality reduction technique.
    correction : Literal["bonferroni", "fdr"], default "bonferroni"
        Correction type for multivariate data. Either 'bonferroni' or 'fdr' (False
        Discovery Rate).
    alternative : Literal["two-sided", "less", "greater"], default "two-sided"
        Defines the alternative hypothesis. Options are 'two-sided', 'less' or
        'greater'.
    n_features
        Number of features used in the statistical test. No need to pass it if no
        preprocessing takes place. In case of a preprocessing step, this can also
        be inferred automatically but could be more expensive to compute.
    """

    def __init__(
        self,
        x_ref: ArrayLike,
        p_val: float = 0.05,
        x_ref_preprocessed: bool = False,
        update_x_ref: Optional[UpdateStrategy] = None,
        preprocess_fn: Optional[Callable[[ArrayLike], ArrayLike]] = None,
        correction: Literal["bonferroni", "fdr"] = "bonferroni",
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        n_features: Optional[int] = None,
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

        # Other attributes
        self.alternative = alternative

    @preprocess_x
    def score(self, x: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute K-S scores and statistics per feature.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        Feature level p-values and K-S statistics.
        """
        x = to_numpy(x)
        x = x.reshape(x.shape[0], -1)
        x_ref = self.x_ref.reshape(self.x_ref.shape[0], -1)
        p_val = np.zeros(self.n_features, dtype=np.float32)
        dist = np.zeros_like(p_val)
        for f in range(self.n_features):
            dist[f], p_val[f] = ks_2samp(x_ref[:, f], x[:, f], alternative=self.alternative, method="exact")
        return p_val, dist
