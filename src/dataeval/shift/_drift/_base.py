"""
Source code derived from Alibi-Detect 0.11.4.

https://github.com/SeldonIO/alibi-detect/tree/v0.11.4.

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

__all__ = []

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, Generic, Literal, TypedDict, TypeVar, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from typing_extensions import Self

from dataeval.protocols import Array, FeatureExtractor, Threshold, UpdateStrategy
from dataeval.shift._drift._chunk import BaseChunker, SizeChunker, resolve_chunker
from dataeval.types import DictOutput, Output
from dataeval.utils.arrays import flatten_samples

R = TypeVar("R")


class DriftUnivariateStats(TypedDict):
    """Per-feature statistics from univariate drift detection.

    Attributes
    ----------
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

    feature_drift: NDArray[np.bool_]
    feature_threshold: float
    p_vals: NDArray[np.float32]
    distances: NDArray[np.float32]


class DriftMMDStats(TypedDict):
    """Statistics from MMD permutation test.

    Attributes
    ----------
    distance_threshold : float
        Squared Maximum Mean Discrepancy threshold above which drift is flagged.
        Determined from permutation test at specified significance level.
    """

    distance_threshold: float


class DriftMVDCStats(TypedDict):
    """Statistics from Multivariate Domain Classifier drift detection.

    Attributes
    ----------
    fold_aurocs : NDArray[np.float32]
        Per-fold AUROC values from stratified K-fold cross-validation.
        Shape is (n_folds,). Variance across folds indicates stability
        of the AUROC estimate.
    feature_importances : NDArray[np.float32]
        Mean feature importances (split-based) from LightGBM, averaged
        across CV folds. Shape is (n_features,). Higher values indicate
        features that contribute more to distinguishing reference from
        test data.
    """

    fold_aurocs: NDArray[np.float32]
    feature_importances: NDArray[np.float32]


T = TypeVar("T", DriftUnivariateStats, DriftMMDStats, DriftMVDCStats)


@dataclass(frozen=True)
class DriftOutput(DictOutput, Generic[T]):
    """Output class for drift detector classes.

    Provides common fields returned by all drift detection methods, containing
    instance-level drift predictions and summary statistics. Detector-specific
    statistics are grouped into typed dictionaries.

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
    metric_name : str
        Name of the metric used (e.g., "mmd2", "auroc", "ks_distance").
        Matches :attr:`DriftChunkedOutput.metric_name` for a uniform interface.
    stats : dict
        Additional drift detector specific statistics.
    """

    drifted: bool
    threshold: float
    p_val: float
    distance: float
    metric_name: str
    stats: T


@dataclass(frozen=True)
class ChunkResult(DictOutput):
    """Result for a single chunk in chunked drift detection.

    Attributes
    ----------
    key : str
        Human-readable identifier for this chunk.
    index : int
        Sequential chunk index.
    start_index : int
        Start index in the original array (-1 for non-contiguous chunks).
    end_index : int
        End index (inclusive) in the original array (-1 for non-contiguous chunks).
    value : float
        Raw metric value for this chunk (e.g., mean distance, MMD^2, AUROC).
    upper_threshold : float or None
        Upper threshold bound. Exceeding this indicates drift.
    lower_threshold : float or None
        Lower threshold bound. Falling below this indicates anomaly.
    drifted : bool
        Whether this chunk's metric exceeds the threshold bounds.
    """

    key: str
    index: int
    start_index: int
    end_index: int
    value: float
    upper_threshold: float | None
    lower_threshold: float | None
    drifted: bool


class DriftChunkedOutput(Output[pl.DataFrame]):
    """Output for chunked drift detection across all detector types."""

    def __init__(self, metric_name: str, chunk_results: pl.DataFrame) -> None:
        self._metric_name = metric_name
        self._data = chunk_results.clone()

    def data(self) -> pl.DataFrame:
        """Return a copy of the chunk results data."""
        return self._data.clone()

    @property
    def metric_name(self) -> str:
        """Name of the metric used for drift detection."""
        return self._metric_name

    @property
    def drifted(self) -> bool:
        """Whether any chunk triggered a drift alert."""
        return bool(self._data["drifted"].any())

    @property
    def threshold(self) -> float:
        """Upper threshold used for drift detection.

        Returns the upper threshold bound that chunk metric values are
        compared against. This mirrors :attr:`DriftOutput.threshold` to
        provide a uniform interface across both output types.
        """
        val = self._data["upper_threshold"].cast(pl.Float64).to_list()[0]
        return float(val) if val is not None else 0.0

    @property
    def distance(self) -> float:
        """Mean metric value across all chunks.

        Provides a single summary statistic comparable to
        :attr:`DriftOutput.distance`. Computed as the mean of per-chunk
        metric values.
        """
        vals = self._data["value"].cast(pl.Float64).mean() or 0.0
        return cast(float, vals)

    @property
    def chunk_results(self) -> pl.DataFrame:
        """Per-chunk drift detection results."""
        return self._data

    @property
    def empty(self) -> bool:
        """Check if the results are empty."""
        return self._data is None or self._data.is_empty()

    @property
    def plot_type(self) -> Literal["drift_chunked"]:
        """Return the plot type identifier."""
        return "drift_chunked"

    def __len__(self) -> int:
        """Return the number of chunks in the results."""
        return 0 if self.empty else len(self._data)

    def __str__(self) -> str:
        return str(self._data)


def _make_chunk_result(
    index: int,
    indices: NDArray[np.intp],
    value: float,
    upper_threshold: float | None,
    lower_threshold: float | None,
) -> ChunkResult:
    """Build a ChunkResult by deriving metadata from an index array."""
    start = int(indices.min())
    end = int(indices.max())
    alert = (upper_threshold is not None and value > upper_threshold) or (
        lower_threshold is not None and value < lower_threshold
    )
    return ChunkResult(
        key=f"[{start}:{end}]",
        index=index,
        start_index=start,
        end_index=end,
        value=value,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
        drifted=alert,
    )


def _chunk_results_to_dataframe(chunk_results: list[ChunkResult]) -> pl.DataFrame:
    """Convert a list of ChunkResult objects to a polars DataFrame."""
    return pl.DataFrame([cr.data() for cr in chunk_results])


def update_strategy(fn: Callable[..., R]) -> Callable[..., R]:
    """Update x_ref with x using selected update methodology."""

    @wraps(fn)
    def _(self: "BaseDrift", data: Array, *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> R:
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

    Uses a fit/predict lifecycle: construct with hyperparameters, call
    :meth:`fit` with reference data, then call :meth:`predict` with test data.

    Parameters
    ----------
    p_val : float, default 0.05
        Significance threshold for drift detection, between 0 and 1.
        Default 0.05 limits false drift alerts to 5% when no drift exists (Type I error rate).
    update_strategy : UpdateStrategy or None, default None
        Strategy for updating reference data when new data arrives.
        When None, reference data remains fixed throughout detection.
        Ignored in chunked mode where a stable baseline is required.
    correction : {"bonferroni", "fdr"}, default "bonferroni"
        Multiple testing correction method for multivariate drift detection.
        "bonferroni" provides conservative family-wise error control.
        "fdr" (False Discovery Rate) offers less conservative control.
    extractor : FeatureExtractor or None, default None
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
    extractor : FeatureExtractor or None
        Feature extraction function for converting input data.
    """

    p_val: float
    update_strategy: UpdateStrategy | None
    correction: Literal["bonferroni", "fdr"]
    n: int
    extractor: FeatureExtractor | None

    def __init__(
        self,
        p_val: float = 0.05,
        update_strategy: UpdateStrategy | None = None,
        correction: Literal["bonferroni", "fdr"] = "bonferroni",
        extractor: FeatureExtractor | None = None,
    ) -> None:
        # Type checking
        if update_strategy is not None and not isinstance(update_strategy, UpdateStrategy):
            raise ValueError("`update_strategy` is not a valid UpdateStrategy class.")
        if correction not in ["bonferroni", "fdr"]:
            raise ValueError("`correction` must be `bonferroni` or `fdr`.")
        if extractor is not None and not isinstance(extractor, FeatureExtractor):
            raise ValueError("`extractor` is not a valid FeatureExtractor.")

        self.p_val = p_val
        self.update_strategy = update_strategy
        self.correction = correction
        self.extractor = extractor
        self._data: Any = None
        self.n: int = 0
        self._x_ref: NDArray[np.float32] | None = None
        self._fitted: bool = False
        self._chunker: BaseChunker | None = None
        self._baseline_values: NDArray[np.float32] | None = None
        self._threshold_bounds: tuple[float | None, float | None] = (None, None)

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
        # Validate that data is Array-like (or will be after feature extraction)
        if self.extractor is None and not isinstance(data, Array):
            raise ValueError("`data` must be Array-like or provide an `extractor` to convert your data to an array.")

        self._data = data
        self._x_ref = None  # reset lazy encoding

        # Compute length after feature extraction if needed
        if self.extractor is not None:
            extracted = self.extractor(data)
            self.n = len(extracted)
        else:
            self.n = len(data)

        # Handle prebuilt chunks as a direct path (no chunker stored)
        if chunks is not None:
            prebuilt = [np.atleast_2d(np.asarray(c, dtype=np.float32)) for c in chunks]
            self._chunker = None
            self._fit_prebuilt(prebuilt, threshold)
            self._fitted = True
            return self

        # Resolve chunker from convenience params
        resolved = resolve_chunker(chunker, chunk_size, chunk_count, chunk_indices)

        if resolved is not None:
            self._fit_chunked(resolved, threshold)
            # Normalize to SizeChunker so predict() uses the same chunk
            # size regardless of test set size. This ensures the per-chunk
            # statistics are comparable to the baseline computed during fit.
            fit_chunk_size = len(resolved.split(self.n)[0])
            self._chunker = SizeChunker(fit_chunk_size, incomplete="append")
        else:
            self._chunker = None

        self._fitted = True
        return self

    def _fit_chunked(self, chunker: BaseChunker, threshold: Threshold | None) -> None:
        """Compute chunked baselines. Override in subclasses for specific metrics."""

    def _fit_prebuilt(self, chunks: list[NDArray[np.float32]], threshold: Threshold | None) -> None:
        """Compute chunked baselines from prebuilt data arrays. Override in subclasses."""

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
            if self._data is None:
                raise RuntimeError("Must call fit() before accessing x_ref.")
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
            the configured extractor.

        Returns
        -------
        NDArray[np.float32]
            Encoded data as flattened 32-bit floating point array.
        """
        # Apply feature extractor if configured
        if self.extractor is not None:
            data = self.extractor(data)

        return flatten_samples(np.asarray(data, dtype=np.float32))
