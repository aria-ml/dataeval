"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

__all__ = []

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import scipy.stats
from numpy.typing import NDArray

from dataeval.protocols import FeatureExtractor, UpdateStrategy
from dataeval.shift._drift._base import BaseDriftUnivariate


class DriftUnivariate(BaseDriftUnivariate):
    """:term:`Drift` detector using univariate statistical tests.

    Detects distributional changes by comparing empirical distributions of
    reference and test datasets using classical statistical tests. For multivariate
    data, applies the test independently to each feature and aggregates results
    using multiple testing correction.

    Supports five statistical methods with different strengths:

    - **Kolmogorov-Smirnov (ks)**: Measures maximum distance between empirical CDFs.
      General-purpose test, sensitive to middle portions of distributions. Supports
      directional alternatives for detecting systematic shifts.

    - **Cramér-von Mises (cvm)**: Measures integrated squared distance between CDFs.
      More sensitive than KS to subtle distributional differences across the entire
      domain. Higher statistical power for many drift types.

    - **Mann-Whitney U (mwu)**: Nonparametric rank-based test for stochastic ordering.
      Robust to outliers and effective for detecting location (median) shifts.
      Works well with non-normal distributions. Supports directional alternatives.

    - **Anderson-Darling (anderson)**: Tests equality of distributions with emphasis
      on tail differences. More sensitive than KS to heavy-tailed distributions.
      Ideal for detecting drift in extreme values. Two-sided only.

    - **Baumgartner-Weiss-Schindler (bws)**: Modern test emphasizing tail differences
      with higher power than KS. Balanced sensitivity to both tails and center.
      Supports directional alternatives. Requires scipy>=1.12.0.

    **Choosing a Method:**

    - Use **ks** for general-purpose drift detection with directional testing
    - Use **cvm** for higher sensitivity to overall distributional changes
    - Use **mwu** for robust detection of median shifts, especially with outliers
    - Use **anderson** when tail behavior is critical (SLA violations, rare events)
    - Use **bws** for best overall power with tail sensitivity and directional testing

    Parameters
    ----------
    data : Any
        Reference dataset used as baseline distribution for drift detection.
        Can be Array or any type supported by feature_extractor parameter.
        Should represent the expected data distribution.
    method : "ks", "cvm", "mwu", "anderson", or "bws", default "ks"
        Statistical test method to use. See method descriptions above.
        Default "ks" provides a well-established general-purpose test.
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
        Alternative hypothesis for the statistical test.
        Applies to: ks, mwu, bws methods only.

        - "two-sided": detects any distributional difference
        - "less": tests if test distribution is stochastically smaller
        - "greater": tests if test distribution is stochastically larger

        Default "two-sided" provides most general drift detection.
        Ignored for cvm and anderson (only support two-sided).
    n_features : int | None, default None
        Number of features to analyze in univariate tests.
        When None, automatically inferred from the flattened shape of first data sample.
    feature_extractor : FeatureExtractor or None, default None
        Optional feature extraction function to convert input data to arrays.
        When provided, enables drift detection on non-array inputs such as
        datasets, metadata, or raw model outputs. The extractor is applied to
        both reference and test data before drift detection.
        When None, data must already be Array-like.
    config : DriftUnivariate.Config or None, default None
        Optional configuration object with default parameters. Parameters
        specified directly in __init__ will override config defaults.

    Example
    -------
    Basic drift detection with Kolmogorov-Smirnov test

    >>> train_emb = np.ones((100, 128), dtype=np.float32)
    >>> drift_detector = DriftUnivariate(train_emb, method="ks")
    >>> test_emb = np.zeros((20, 128), dtype=np.float32)
    >>> result = drift_detector.predict(test_emb)
    >>> print(f"Drift detected: {result.drifted}")
    Drift detected: True

    Using Mann-Whitney U for robust median shift detection

    >>> drift_mwu = DriftUnivariate(train_emb, method="mwu")
    >>> result = drift_mwu.predict(test_emb)

    Using Anderson-Darling for tail-sensitive detection

    >>> drift_anderson = DriftUnivariate(train_emb, method="anderson")
    >>> result = drift_anderson.predict(test_emb)

    Detect if test data has systematically higher values

    >>> drift_greater = DriftUnivariate(train_emb, method="ks", alternative="greater")
    >>> result = drift_greater.predict(test_emb)

    Using Baumgartner-Weiss-Schindler with high power

    >>> drift_bws = DriftUnivariate(train_emb, method="bws")
    >>> result = drift_bws.predict(test_emb)

    Access feature-level results

    >>> n_features = result.feature_drift
    >>> print(f"Features showing drift: {n_features.sum()} / {len(n_features)}")
    Features showing drift: 128 / 128

    Using configuration:

    >>> config = DriftUnivariate.Config(method="cvm", p_val=0.01, correction="fdr")
    >>> drift = DriftUnivariate(train_emb, config=config)
    """

    @dataclass
    class Config:
        """
        Configuration for DriftUnivariate detector.

        Attributes
        ----------
        method : {"ks", "cvm", "mwu", "anderson", "bws"}, default "ks"
            Statistical test method to use.
        p_val : float, default 0.05
            Significance threshold for drift detection.
        correction : {"bonferroni", "fdr"}, default "bonferroni"
            Multiple testing correction method.
        alternative : {"two-sided", "less", "greater"}, default "two-sided"
            Alternative hypothesis for the statistical test.
        n_features : int or None, default None
            Number of features to analyze.
        """

        method: Literal["ks", "cvm", "mwu", "anderson", "bws"] = "ks"
        p_val: float = 0.05
        correction: Literal["bonferroni", "fdr"] = "bonferroni"
        alternative: Literal["two-sided", "less", "greater"] = "two-sided"
        n_features: int | None = None

    def __init__(
        self,
        data: Any,
        method: Literal["ks", "cvm", "mwu", "anderson", "bws"] | None = None,
        p_val: float | None = None,
        update_strategy: UpdateStrategy | None = None,
        correction: Literal["bonferroni", "fdr"] | None = None,
        alternative: Literal["two-sided", "less", "greater"] | None = None,
        n_features: int | None = None,
        feature_extractor: FeatureExtractor | None = None,
        config: Config | None = None,
    ) -> None:
        # Store config or create default
        self.config: DriftUnivariate.Config = config or DriftUnivariate.Config()

        # Use config defaults if parameters not specified
        method = method if method is not None else self.config.method
        p_val = p_val if p_val is not None else self.config.p_val
        correction = correction if correction is not None else self.config.correction
        alternative = alternative if alternative is not None else self.config.alternative
        n_features = n_features if n_features is not None else self.config.n_features

        super().__init__(
            data=data,
            p_val=p_val,
            update_strategy=update_strategy,
            correction=correction,
            n_features=n_features,
            feature_extractor=feature_extractor,
        )

        # Validate method
        valid_methods = ["ks", "cvm", "mwu", "anderson", "bws"]
        if method not in valid_methods:
            raise ValueError(f"`method` must be one of {valid_methods}, got '{method}'.")

        # Check bws availability
        if method == "bws" and not hasattr(scipy.stats, "bws_test"):
            raise ImportError("The 'bws' method requires scipy>=1.12.0.")

        # Validate alternative
        if alternative not in ["two-sided", "less", "greater"]:
            raise ValueError("`alternative` must be 'two-sided', 'less', or 'greater'.")

        self.method = method
        self.alternative = alternative

    def _score_fn(self, x: NDArray[np.float32], y: NDArray[np.float32]) -> tuple[np.float32, np.float32]:
        """Compute test statistic and p-value for the selected method.

        Parameters
        ----------
        x : NDArray[np.float32]
            Reference data for a single feature.
        y : NDArray[np.float32]
            Test data for a single feature.

        Returns
        -------
        tuple[np.float32, np.float32]
            Test statistic and p-value from the selected statistical test.
        """
        if self.method == "ks":
            return scipy.stats.ks_2samp(x, y, alternative=self.alternative, method="exact")

        if self.method == "cvm":
            result = scipy.stats.cramervonmises_2samp(x, y, method="auto")
            return np.float32(result.statistic), np.float32(result.pvalue)

        if self.method == "mwu":
            result = scipy.stats.mannwhitneyu(x, y, alternative=self.alternative)
            return np.float32(result.statistic), np.float32(result.pvalue)

        if self.method == "anderson":
            result = scipy.stats.anderson_ksamp([x, y])
            return np.float32(result.statistic), np.float32(result.pvalue)  # type: ignore

        if self.method == "bws":
            result = scipy.stats.bws_test(x, y, alternative=self.alternative)
            return np.float32(result.statistic), np.float32(result.pvalue)

        raise ValueError(f"Unknown method: {self.method}")
