"""
Source code derived from Alibi-Detect 0.11.4.

https://github.com/SeldonIO/alibi-detect/tree/v0.11.4.

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

__all__ = []

import math
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import numpy as np
import scipy.stats
from numpy.typing import NDArray
from typing_extensions import Self

from dataeval.protocols import Array, FeatureExtractor, Threshold, UpdateStrategy
from dataeval.shift._drift._base import BaseDrift, DriftAdaptiveMixin, DriftChunkerMixin, DriftOutput
from dataeval.shift._drift._chunk import BaseChunker
from dataeval.types import set_metadata
from dataeval.utils.thresholds import ZScoreThreshold


class DriftUnivariateStats(TypedDict):
    """Per-feature statistics from univariate drift detection.

    Attributes
    ----------
    p_val : float
        Mean p-value across all features, between 0 and 1.
    feature_drift : NDArray[bool]
        Boolean array indicating which features show drift.
        Shape matches the number of features in the input data.
    feature_threshold : float
        Uncorrected p-value threshold used for individual feature testing.
        Typically the original p_val before multivariate correction.
    p_vals : NDArray[np.float32]
        P-values for each feature, all values between 0 and 1.
        Shape matches the number of features in the input data.
    distances : NDArray[np.float32]
        Test statistics for each feature, all values >= 0.
        Shape matches the number of features in the input data.
    """

    p_val: float
    feature_drift: NDArray[np.bool_]
    feature_threshold: float
    p_vals: NDArray[np.float32]
    distances: NDArray[np.float32]


class DriftUnivariate(DriftAdaptiveMixin, DriftChunkerMixin, BaseDrift):
    """:term:`Drift` detector using univariate statistical tests.

    Detects distributional changes by comparing empirical distributions of
    reference and test datasets using classical statistical tests. For multivariate
    data, applies the test independently to each feature and aggregates results
    using multiple testing correction.

    Uses a fit/predict lifecycle: construct with hyperparameters, call
    :meth:`fit` with reference data, then call :meth:`predict` with test data.
    Supports chunked mode when chunking parameters are provided to :meth:`fit`.

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
    method : "ks", "cvm", "mwu", "anderson", or "bws", default "ks"
        Statistical test method to use. See method descriptions above.
        Default "ks" provides a well-established general-purpose test.
    p_val : float, default 0.05
        Significance threshold for drift detection, between 0 and 1.
        Default 0.05 limits false drift alerts to 5% when no drift exists (Type I error rate).
    update_strategy : UpdateStrategy or None, default None
        Strategy for updating reference data when new data arrives.
        When None, reference data remains fixed throughout detection.
        Ignored in chunked mode.
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
    extractor : FeatureExtractor or None, default None
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

    >>> rng = np.random.default_rng(42)
    >>> train_emb = rng.standard_normal((100, 128)).astype(np.float32)
    >>> drift_detector = DriftUnivariate(method="ks").fit(train_emb)
    >>> test_emb = np.zeros((20, 128), dtype=np.float32)
    >>> result = drift_detector.predict(test_emb)
    >>> print(f"Drift detected: {result.drifted}")
    Drift detected: True

    Chunked drift detection with z-score thresholds

    >>> drift_detector = DriftUnivariate(method="ks").fit(train_emb, chunk_size=20)
    >>> result = drift_detector.predict(test_emb)
    >>> print(f"Drift detected: {result.drifted}, chunks: {len(result.details)}")
    Drift detected: True, chunks: 1

    Using configuration:

    >>> config = DriftUnivariate.Config(method="cvm", p_val=0.01, correction="fdr")
    >>> drift = DriftUnivariate(config=config).fit(train_emb)
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
        update_strategy : UpdateStrategy or None, default None
            Strategy for updating reference data over time.
        extractor : FeatureExtractor or None, default None
            Feature extractor for transforming input data before drift detection.
        """

        method: Literal["ks", "cvm", "mwu", "anderson", "bws"] = "ks"
        p_val: float = 0.05
        correction: Literal["bonferroni", "fdr"] = "bonferroni"
        alternative: Literal["two-sided", "less", "greater"] = "two-sided"
        n_features: int | None = None
        update_strategy: UpdateStrategy | None = None
        extractor: FeatureExtractor | None = None

    def __init__(
        self,
        method: Literal["ks", "cvm", "mwu", "anderson", "bws"] | None = None,
        p_val: float | None = None,
        update_strategy: UpdateStrategy | None = None,
        correction: Literal["bonferroni", "fdr"] | None = None,
        alternative: Literal["two-sided", "less", "greater"] | None = None,
        n_features: int | None = None,
        extractor: FeatureExtractor | None = None,
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

        # Initialise base + mixins
        BaseDrift.__init__(self)
        self._init_chunking()
        self._init_adaptive(
            extractor=extractor,
            update_strategy=update_strategy,
        )

        # Validate correction (only used by Univariate)
        if correction not in ["bonferroni", "fdr"]:
            raise ValueError("`correction` must be `bonferroni` or `fdr`.")

        self.p_val = p_val
        self.correction = correction
        self._n_features = n_features

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
        self._metric_name = f"{method}_distance"

    @property
    def n_features(self) -> int:
        """Number of features in the reference data.

        Lazily computes the number of features from the first data sample
        if not provided during initialization. Features correspond to the
        flattened dimensionality of the input data (e.g., pixels for images).

        Returns
        -------
        int
            Number of features (flattened dimensions) in the reference data.
            Always > 0 for valid datasets.

        Notes
        -----
        For image data, this equals C x H x W.
        Computed once and cached for efficiency.
        """
        if self._n_features is None:
            if self._data is None:
                raise RuntimeError("Must call fit() before accessing n_features.")
            if self.extractor is not None:
                first_encoded = self._encode(self._data[:1])
                self._n_features = first_encoded.shape[1]
            else:
                self._n_features = int(math.prod(self._data[0].shape))

        return self._n_features

    def fit(
        self,
        data: Any,
        chunker: BaseChunker | None = None,
        chunk_size: int | None = None,
        chunk_count: int | None = None,
        chunks: list[Any] | None = None,
        chunk_indices: list[list[int]] | None = None,
        threshold: Threshold | None = None,
    ) -> Self:
        """Fit detector with reference data, optionally enabling chunked mode.

        When chunking is enabled, the detector computes per-chunk baseline
        metrics from the reference data and derives threshold bounds. During
        prediction, the test data is split into chunks of the **same size**
        used here, so that per-chunk statistics are comparable to the baseline.

        If ``chunk_count`` is provided, the effective chunk size is computed
        as ``len(data) // chunk_count`` and locked in for prediction.  This
        means the number of chunks produced by ``predict()`` depends on the
        test set size, not the original ``chunk_count``.  Use ``chunk_size``
        directly when you want explicit control over the chunk size used for
        both fitting and prediction.

        Parameters
        ----------
        data : Any
            Reference dataset used as baseline for drift detection.
            Can be Array or any type supported by the configured extractor.
        chunker : ArrayChunker or None, default None
            Explicit chunker instance for chunked mode.
        chunk_size : int or None, default None
            Create fixed-size chunks of this many samples. The same size is
            used during prediction to keep statistics comparable.
        chunk_count : int or None, default None
            Split reference into this many equal chunks. Converted to a
            fixed ``chunk_size`` based on the reference data length.
        chunks : list[ArrayLike] or None, default None
            Pre-split reference data arrays for chunked mode.
        chunk_indices : list[list[int]] or None, default None
            Index groupings for chunking reference data.
        threshold : Threshold or None, default None
            Threshold strategy for chunked mode. Defaults to
            StandardDeviationThreshold (mean +/- 3*std).

        Returns
        -------
        Self
        """
        self._set_adaptive_data(data)

        # Prebuilt chunks → convert to numpy
        if chunks is not None:
            chunks = [np.atleast_2d(np.asarray(c, dtype=np.float32)) for c in chunks]

        self._resolve_fit_chunks(
            self.n,
            chunker=chunker,
            chunk_size=chunk_size,
            chunk_count=chunk_count,
            chunks=chunks,
            chunk_indices=chunk_indices,
            threshold=threshold,
            default_threshold=ZScoreThreshold(lower_limit=0.0),
        )

        self._fitted = True
        return self

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

    def _score_against(
        self, x_ref: NDArray[np.float32], x_test: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Score test data against given reference data (feature-wise).

        Parameters
        ----------
        x_ref : NDArray[np.float32]
            Reference data, shape (n_ref, n_features).
        x_test : NDArray[np.float32]
            Test data, shape (n_test, n_features).

        Returns
        -------
        tuple[NDArray[np.float32], NDArray[np.float32]]
            First array contains p-values per feature.
            Second array contains test statistics per feature.
        """
        n_features = x_ref.shape[1]
        p_val = np.zeros(n_features, dtype=np.float32)
        dist = np.zeros_like(p_val)
        for f in range(n_features):
            dist[f], p_val[f] = self._score_fn(x_ref[:, f], x_test[:, f])
        return p_val, dist

    def score(self, data: Array) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Calculate feature-wise p-values and test statistics.

        Applies the detector's statistical test independently to each feature,
        comparing the distribution of each feature between reference and test data.

        Parameters
        ----------
        data : Array
            Test dataset to compare against reference data.

        Returns
        -------
        tuple[NDArray[np.float32], NDArray[np.float32]]
            First array contains p-values for each feature (all between 0 and 1).
            Second array contains test statistics for each feature (all >= 0).
            Both arrays have shape (n_features,).

        Notes
        -----
        Lower p-values indicate stronger evidence of drift for that feature.
        Higher test statistics indicate greater distributional differences.
        """
        x_np = self._encode(data)
        return self._score_against(self.x_ref, x_np)

    def _apply_correction(self, p_vals: NDArray[np.float32]) -> tuple[bool, float]:
        """
        Apply multiple testing correction to feature-wise p-values.

        Corrects for multiple comparisons across features to control
        false positive rates. Bonferroni correction divides the significance
        threshold by the number of features. FDR correction uses the
        Benjamini-Hochberg procedure for less conservative control.

        Parameters
        ----------
        p_vals : NDArray[np.float32]
            Array of p-values from univariate tests for each feature.
            All values should be between 0 and 1.

        Returns
        -------
        tuple[bool, float]
            Boolean indicating whether drift was detected after correction.
            Float is the effective threshold used for detection.

        Notes
        -----
        Bonferroni correction: threshold = p_val / n_features
        FDR correction: Uses Benjamini-Hochberg step-up procedure
        """
        if self.correction == "bonferroni":
            threshold = self.p_val / self.n_features
            drift_pred = bool((p_vals < threshold).any())
            return drift_pred, threshold
        if self.correction == "fdr":
            n = p_vals.shape[0]
            i = np.arange(n) + np.intp(1)
            p_sorted = np.sort(p_vals)
            q_threshold = self.p_val * i / n
            below_threshold = p_sorted < q_threshold
            try:
                idx_threshold = int(np.where(below_threshold)[0].max())
            except ValueError:  # sorted p-values not below thresholds
                return bool(below_threshold.any()), q_threshold.min()
            return bool(below_threshold.any()), q_threshold[idx_threshold]
        raise ValueError("`correction` needs to be either `bonferroni` or `fdr`.")

    def _compute_chunk_metric(self, chunk_data: NDArray[np.float32]) -> float:
        """Compute mean distance: chunk vs full reference."""
        _, dist = self._score_against(self.x_ref, chunk_data)
        return float(np.mean(dist))

    def _fit_chunked_baseline(
        self,
        chunker: BaseChunker,
        threshold: Threshold | None,
        default_threshold: Threshold | None,
    ) -> None:
        """Compute per-chunk mean distances on reference (chunk vs rest)."""
        x_ref = self.x_ref  # trigger lazy encoding
        n = len(x_ref)
        index_groups = chunker.split(n)

        baseline_values: list[float] = []
        for indices in index_groups:
            mask = np.ones(n, dtype=bool)
            mask[indices] = False
            _, dist = self._score_against(x_ref[mask], x_ref[indices])
            baseline_values.append(float(np.mean(dist)))

        self._resolve_baseline_threshold(np.array(baseline_values, dtype=np.float32), threshold, default_threshold)

    def _fit_prebuilt_baseline(
        self,
        chunks: list[Any],
        threshold: Threshold | None,
        default_threshold: Threshold | None,
    ) -> None:
        """Compute per-chunk mean distances from prebuilt chunks (chunk vs rest)."""
        _ = self.x_ref  # trigger lazy encoding

        baseline_values: list[float] = []
        for i, chunk_data in enumerate(chunks):
            rest_data = np.concatenate([c for j, c in enumerate(chunks) if j != i], axis=0)
            _, dist = self._score_against(rest_data, chunk_data)
            baseline_values.append(float(np.mean(dist)))

        self._resolve_baseline_threshold(np.array(baseline_values, dtype=np.float32), threshold, default_threshold)

    @set_metadata
    def predict(
        self,
        data: Any = None,
        chunks: list[Any] | None = None,
        chunk_indices: list[list[int]] | None = None,
    ) -> DriftOutput:
        """Predict drift and optionally update reference data.

        In non-chunked mode, performs feature-wise drift detection with
        multiple testing correction. In chunked mode, computes per-chunk
        metrics and compares against baseline thresholds.

        Parameters
        ----------
        data : Any, optional
            Test dataset to analyze for drift. Required for non-chunked mode
            and for chunked mode unless pre-built chunks are provided.
        chunks : list[ArrayLike] or None, default None
            Pre-built test data chunks. When provided, each array is treated
            as a separate chunk and ``data`` is ignored.
        chunk_indices : list[list[int]] or None, default None
            Index groupings for chunking ``data``. Each inner list specifies
            which samples from ``data`` belong to a chunk.

        Returns
        -------
        DriftOutput
            Non-chunked mode: ``details`` is a :class:`DriftUnivariateStats` TypedDict.
            Chunked mode: ``details`` is a :class:`polars.DataFrame` with per-chunk results.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict().")

        if self.is_chunked or chunks is not None or chunk_indices is not None:
            if chunks is not None:
                prepared = [np.atleast_2d(np.asarray(c, dtype=np.float32)) for c in chunks]
                return self._predict_chunked(chunks_override=prepared)

            x_test = self._encode(data) if data is not None else None
            return self._predict_chunked(
                x_test=x_test,
                chunk_indices_override=chunk_indices,
            )

        if data is None:
            raise ValueError("data is required for non-chunked prediction.")
        return self._predict_single(data)

    def _predict_single(self, data: Array) -> DriftOutput:
        """Non-chunked prediction with optional reference update."""
        p_vals, dist = self.score(data)

        feature_drift = (p_vals < self.p_val).astype(np.bool_)
        drift_pred, threshold = self._apply_correction(p_vals)

        result = DriftOutput(
            drifted=drift_pred,
            threshold=threshold,
            distance=float(np.mean(dist)),
            metric_name=self._metric_name,
            details=DriftUnivariateStats(
                p_val=float(np.mean(p_vals)),
                feature_drift=feature_drift,
                feature_threshold=self.p_val,
                p_vals=p_vals,
                distances=dist,
            ),
        )

        # Update reference if strategy is set
        self._apply_update_strategy(data)

        return result
