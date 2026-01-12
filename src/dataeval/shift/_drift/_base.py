"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

__all__ = []

import math
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

from dataeval.protocols import Array, FeatureExtractor, UpdateStrategy
from dataeval.types import DictOutput, set_metadata
from dataeval.utils.arrays import flatten_samples

R = TypeVar("R")


@dataclass(frozen=True)
class DriftBaseOutput(DictOutput):
    """Base output class for drift detector classes.

    Provides common fields returned by all drift detection methods, containing
    instance-level drift predictions and summary statistics. Subclasses extend
    this with detector-specific additional fields.

    Attributes
    ----------
    drifted : bool
        Whether drift was detected in the analyzed data. True indicates
        significant drift from reference distribution.
    threshold : float
        Significance threshold used for drift detection, typically between 0 and 1.
        For multivariate methods, this is the corrected threshold after
        Bonferroni or FDR correction.
    p_val : float
        Instance-level p-value from statistical test, between 0 and 1.
        For univariate methods, this is the mean p-value across all features.
    distance : float
        Instance-level test statistic or distance metric, always >= 0.
        For univariate methods, this is the mean distance across all features.
        Higher values indicate greater deviation from reference distribution.
    """

    drifted: bool
    threshold: float
    p_val: float
    distance: float


@dataclass(frozen=True)
class DriftOutput(DriftBaseOutput):
    """Output class for univariate drift detectors.

    Extends :class:`.DriftBaseOutput` with feature-level (per-pixel) drift information.
    Used by Kolmogorov-Smirnov, Cramér-von Mises, and uncertainty-based
    drift detectors that analyze each feature independently.

    Attributes
    ----------
    drifted : bool
        Overall drift prediction after multivariate correction.
    threshold : float
        Corrected threshold after Bonferroni or FDR correction for multiple testing.
    p_val : float
        Mean p-value across all features, between 0 and 1.
        For descriptive purposes only; individual feature p-values are used
        for drift detection decisions. Can appear high even when drifted=True
        if only a subset of features show drift.
    distance : float
        Mean test statistic across all features, always >= 0.
    feature_drift : NDArray[bool]
        Boolean array indicating which features (pixels) show drift.
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

    Notes
    -----
    Feature-level analysis enables identification of specific pixels or regions
    that contribute most to detected drift, useful for interpretability.
    """

    feature_drift: NDArray[np.bool_]
    feature_threshold: float
    p_vals: NDArray[np.float32]
    distances: NDArray[np.float32]


def update_strategy(fn: Callable[..., R]) -> Callable[..., R]:
    """Decorator to update x_ref with x using selected update methodology"""

    @wraps(fn)
    def _(self: BaseDrift, data: Array, *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> R:
        output = fn(self, data, *args, **kwargs)

        # update reference dataset
        if self.update_strategy is not None:
            self._x_ref = self.update_strategy(self.x_ref, self._encode(data))
            self.n += len(data)

        return output

    return _


class BaseDrift:
    """Base class for drift detection algorithms.

    Provides common functionality for drift detectors including reference data
    management, encoding of input data, and statistical correction methods.
    Subclasses implement specific drift detection algorithms.

    Parameters
    ----------
    data : Any
        Reference dataset used as baseline for drift detection.
        Can be image embeddings, raw arrays, or any data type that can be
        converted to arrays via the feature_extractor parameter.
    p_val : float, default 0.05
        Significance threshold for drift detection, between 0 and 1.
        Default 0.05 limits false drift alerts to 5% when no drift exists (Type I error rate).
    update_strategy : UpdateStrategy or None, default None
        Strategy for updating reference data when new data arrives.
        When None, reference data remains fixed throughout detection.
        Default None maintains stable baseline for consistent comparison.
    correction : {"bonferroni", "fdr"}, default "bonferroni"
        Multiple testing correction method for multivariate drift detection.
        "bonferroni" provides conservative family-wise error control.
        "fdr" (False Discovery Rate) offers less conservative control.
        Default "bonferroni" minimizes false positive drift detections.
    feature_extractor : FeatureExtractor or None, default None
        Optional feature extraction function to convert input data to arrays.
        When provided, enables drift detection on non-array inputs such as
        datasets, metadata, or raw model outputs. The extractor is applied to
        both reference and test data before drift detection.
        When None, data must already be Array-like.

    Attributes
    ----------
    p_val : float
        Significance threshold for statistical tests.
    update_strategy : UpdateStrategy or None
        Reference data update strategy.
    correction : {"bonferroni", "fdr"}
        Multiple testing correction method.
    n : int
        Number of samples in the reference dataset.
    feature_extractor : FeatureExtractor or None
        Feature extraction function for converting input data.
    """

    p_val: float
    update_strategy: UpdateStrategy | None
    correction: Literal["bonferroni", "fdr"]
    n: int
    feature_extractor: FeatureExtractor | None

    def __init__(
        self,
        data: Any,
        p_val: float = 0.05,
        update_strategy: UpdateStrategy | None = None,
        correction: Literal["bonferroni", "fdr"] = "bonferroni",
        feature_extractor: FeatureExtractor | None = None,
    ) -> None:
        # Type checking
        if update_strategy is not None and not isinstance(update_strategy, UpdateStrategy):
            raise ValueError("`update_strategy` is not a valid UpdateStrategy class.")
        if correction not in ["bonferroni", "fdr"]:
            raise ValueError("`correction` must be `bonferroni` or `fdr`.")
        if feature_extractor is not None and not isinstance(feature_extractor, FeatureExtractor):
            raise ValueError("`feature_extractor` is not a valid FeatureExtractor.")

        # Validate that data is Array-like (or will be after feature extraction)
        if feature_extractor is None and not isinstance(data, Array):
            raise ValueError(
                "`data` must be Array-like or provide a `feature_extractor` to convert your data to an array."
            )

        self._data = data
        self.p_val = p_val
        self.update_strategy = update_strategy
        self.correction = correction
        self.feature_extractor = feature_extractor
        # Compute length after feature extraction if needed
        if feature_extractor is not None:
            extracted = feature_extractor(data)
            self.n = len(extracted)
        else:
            self.n = len(data)

        self._x_ref: NDArray[np.float32] | None = None

    @property
    def x_ref(self) -> NDArray[np.float32]:
        """Reference data for drift detection.

        Lazily encodes the reference dataset on first access.
        Data is flattened and converted to 32-bit floating point for
        consistent numerical processing across different input types.

        Returns
        -------
        NDArray[np.float32]
            Reference data as flattened 32-bit floating point array.
            Shape is (n_samples, n_features_flattened).

        Notes
        -----
        Data is cached after first access to avoid repeated encoding overhead.
        """
        if self._x_ref is None:
            self._x_ref = self._encode(self._data)
        return self._x_ref

    def _encode(self, data: Any) -> NDArray[np.float32]:
        """
        Encode input data to consistent numpy format.

        Applies feature extraction if configured, then converts to flattened
        32-bit floating point arrays for drift detection.

        Parameters
        ----------
        data : Any
            Input data to encode. Can be Array or any type supported by
            the configured feature_extractor.

        Returns
        -------
        NDArray[np.float32]
            Encoded data as flattened 32-bit floating point array.
        """
        # Apply feature extractor if configured
        if self.feature_extractor is not None:
            data = self.feature_extractor(data)

        return flatten_samples(np.asarray(data, dtype=np.float32))


class BaseDriftUnivariate(BaseDrift):
    """
    Base class for univariate drift detection algorithms.

    Extends BaseDrift with feature-wise drift detection capabilities.
    Applies statistical tests independently to each feature (pixel) and
    uses multiple testing correction to control false discovery rates.

    Parameters
    ----------
    data : Any
        Reference dataset used as baseline for drift detection.
        Can be Array or any type supported by feature_extractor.
    p_val : float, default 0.05
        Significance threshold for drift detection, between 0 and 1.
        Default 0.05 limits false drift alerts to 5% when no drift exists (Type I error rate).
    update_strategy : UpdateStrategy or None, default None
        Strategy for updating reference data when new data arrives.
        When None, reference data remains fixed throughout detection.
        Default None maintains stable baseline for consistent comparison.
    correction : {"bonferroni", "fdr"}, default "bonferroni"
        Multiple testing correction method for controlling false positives
        across multiple features. "bonferroni" divides significance level
        by number of features. "fdr" uses Benjamini-Hochberg procedure.
        Default "bonferroni" provides conservative family-wise error control.
    n_features : int or None, default None
        Number of features to analyze. When None, automatically inferred
        from the first sample's flattened shape. Default None enables
        automatic feature detection for flexible input handling.
    feature_extractor : FeatureExtractor or None, default None
        Optional feature extraction function to convert input data to arrays.
        When provided, enables drift detection on non-array inputs.

    Attributes
    ----------
    p_val : float
        Significance threshold for statistical tests.
    update_strategy : UpdateStrategy or None
        Reference data update strategy.
    correction : {"bonferroni", "fdr"}
        Multiple testing correction method.
    n : int
        Number of samples in the reference dataset.
    feature_extractor : FeatureExtractor or None
        Feature extraction function for converting input data.
    """

    def __init__(
        self,
        data: Any,
        p_val: float = 0.05,
        update_strategy: UpdateStrategy | None = None,
        correction: Literal["bonferroni", "fdr"] = "bonferroni",
        n_features: int | None = None,
        feature_extractor: FeatureExtractor | None = None,
    ) -> None:
        super().__init__(data, p_val, update_strategy, correction, feature_extractor)

        self._n_features = n_features

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
        # lazy process n_features as needed
        if self._n_features is None:
            # Apply feature extractor to first sample if configured
            if self.feature_extractor is not None:
                # Get first element to determine feature dimensionality
                # We need to encode at least one sample to know the output shape
                first_encoded = self._encode(self._data[:1])
                self._n_features = first_encoded.shape[1]  # (1, n_features)
            else:
                self._n_features = int(math.prod(self._data[0].shape))

        return self._n_features

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
        p_val = np.zeros(self.n_features, dtype=np.float32)
        dist = np.zeros_like(p_val)
        for f in range(self.n_features):
            dist[f], p_val[f] = self._score_fn(self.x_ref[:, f], x_np[:, f])
        return p_val, dist

    @abstractmethod
    def _score_fn(self, x: NDArray[np.float32], y: NDArray[np.float32]) -> tuple[np.float32, np.float32]: ...

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

    @set_metadata
    @update_strategy
    def predict(self, data: Array) -> DriftOutput:
        """Predict drift and update reference data using specified strategy.

        Performs feature-wise drift detection, applies multiple testing
        correction, and optionally updates the reference dataset based
        on the configured update strategy.

        Parameters
        ----------
        data : Array
            Test dataset to analyze for drift against reference data.

        Returns
        -------
        DriftOutput
            Complete drift detection results including overall :term:`drift<Drift>` prediction,
            corrected thresholds, feature-level analysis, and summary :term:`statistics<Statistics>`.
        """

        # compute drift scores
        p_vals, dist = self.score(data)

        feature_drift = (p_vals < self.p_val).astype(np.bool_)
        drift_pred, threshold = self._apply_correction(p_vals)
        return DriftOutput(
            drift_pred, threshold, float(np.mean(p_vals)), float(np.mean(dist)), feature_drift, self.p_val, p_vals, dist
        )
