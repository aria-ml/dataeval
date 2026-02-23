"""Base classes and mixins for drift detection."""

__all__ = []

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
import polars as pl
from numpy.typing import NDArray

from dataeval.protocols import Array, FeatureExtractor, Threshold, UpdateStrategy
from dataeval.shift._drift._chunk import BaseChunker, SizeChunker, resolve_chunker
from dataeval.types import DictOutput
from dataeval.utils.arrays import flatten_samples
from dataeval.utils.thresholds import ZScoreThreshold

T = TypeVar("T")


@dataclass(frozen=True)
class DriftOutput(DictOutput, Generic[T]):
    """Output class for drift detector classes.

    Provides common fields returned by all drift detection methods, containing
    instance-level drift predictions and summary statistics.

    For non-chunked mode, ``details`` holds a detector-specific TypedDict
    with test statistics (including ``p_val``). For chunked mode, ``details``
    holds a :class:`polars.DataFrame` with per-chunk results.

    Attributes
    ----------
    drifted : bool
        Whether drift was detected in the analyzed data. True indicates
        significant drift from reference distribution.
    threshold : float
        Significance threshold used for drift detection, typically between 0 and 1.
        For multivariate methods, this is the corrected threshold after
        Bonferroni or FDR correction.
    distance : float
        Instance-level test statistic or distance metric, always >= 0.
        For univariate methods, this is the mean distance across all features.
        Higher values indicate greater deviation from reference distribution.
    metric_name : str
        Name of the metric used (e.g., "mmd2", "auroc", "ks_distance").
    details : T
        Detector-specific statistics (TypedDict) for non-chunked mode,
        or a :class:`polars.DataFrame` of per-chunk results for chunked mode.
    """

    drifted: bool
    threshold: float
    distance: float
    metric_name: str
    details: T


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


class BaseDrift:
    """Lightweight base for all drift detectors.

    Provides common state for the fit/predict lifecycle: a fitted flag
    and cached reference data. Combine with :class:`DriftChunkerMixin`
    for chunking support and :class:`DriftAdaptiveMixin` for feature
    extraction and reference-update capabilities.
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self._x_ref: NDArray[np.float32] | None = None

    @property
    def x_ref(self) -> NDArray[np.float32]:
        """Reference data for drift detection.

        Returns
        -------
        NDArray[np.float32]
            Reference data array.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self._x_ref is None:
            raise RuntimeError("Must call fit() before accessing x_ref.")
        return self._x_ref


class DriftChunkerMixin:
    """Mixin providing chunked drift detection infrastructure.

    Manages chunker state, baseline computation, and the shared
    ``_predict_chunked`` iteration that was previously duplicated
    across every drift detector.

    Host class must provide:

    - ``_compute_chunk_metric(chunk_data) -> float``
    - ``_metric_name: str``
    - ``x_ref`` property (from :class:`BaseDrift` or :class:`DriftAdaptiveMixin`)
    """

    _chunker: BaseChunker | None
    _baseline_values: NDArray[np.float32] | None
    _threshold_bounds: tuple[float | None, float | None]
    _metric_name: str

    def _init_chunking(self) -> None:
        """Initialise chunking state.  Call from ``__init__``."""
        self._chunker = None
        self._baseline_values = None
        self._threshold_bounds = (None, None)

    def _resolve_fit_chunks(
        self,
        n_ref: int,
        *,
        chunker: BaseChunker | None = None,
        chunk_size: int | None = None,
        chunk_count: int | None = None,
        chunks: list[Any] | None = None,
        chunk_indices: list[list[int]] | None = None,
        threshold: Threshold | None = None,
        default_threshold: Threshold | None = None,
    ) -> bool:
        """Resolve chunking parameters and compute baselines.

        Call from the detector's ``fit()`` after setting up reference
        data.  Returns ``True`` when chunked mode is activated.
        """
        # Pre-built chunks path
        if chunks is not None:
            self._chunker = None
            self._fit_prebuilt_baseline(chunks, threshold, default_threshold)
            return True

        # Resolve convenience params → chunker
        resolved = resolve_chunker(chunker, chunk_size, chunk_count, chunk_indices)
        if resolved is not None:
            self._fit_chunked_baseline(resolved, threshold, default_threshold)
            # Normalise to SizeChunker so predict() always uses the same
            # chunk size, keeping per-chunk stats comparable to baseline.
            fit_chunk_size = len(resolved.split(n_ref)[0])
            self._chunker = SizeChunker(fit_chunk_size, incomplete="append")
            return True

        self._chunker = None
        return False

    def _fit_chunked_baseline(
        self,
        chunker: BaseChunker,
        threshold: Threshold | None,
        default_threshold: Threshold | None,
    ) -> None:
        """Compute per-chunk baseline metrics from reference data.

        Default implementation scores each chunk independently via
        :meth:`_compute_chunk_metric`.  Override for chunk-vs-rest
        patterns (e.g. DomainClassifier, Univariate, MMD).
        """
        x_ref: NDArray[np.float32] = self.x_ref  # type: ignore[attr-defined]
        index_groups = chunker.split(len(x_ref))
        baseline = np.array(
            [self._compute_chunk_metric(x_ref[idx]) for idx in index_groups],
            dtype=np.float32,
        )
        self._resolve_baseline_threshold(baseline, threshold, default_threshold)

    def _fit_prebuilt_baseline(
        self,
        chunks: list[Any],
        threshold: Threshold | None,
        default_threshold: Threshold | None,
    ) -> None:
        """Compute per-chunk baseline metrics from pre-built chunks.

        Default implementation scores each chunk independently via
        :meth:`_compute_chunk_metric`.  Override for chunk-vs-rest
        patterns.
        """
        baseline = np.array(
            [self._compute_chunk_metric(c) for c in chunks],
            dtype=np.float32,
        )
        self._resolve_baseline_threshold(baseline, threshold, default_threshold)

    def _resolve_baseline_threshold(
        self,
        baseline_values: NDArray[np.float32],
        threshold: Threshold | None,
        default_threshold: Threshold | None,
    ) -> None:
        """Store baseline values and derive threshold bounds."""
        self._baseline_values = baseline_values
        thresh = threshold if threshold is not None else (default_threshold or ZScoreThreshold())
        self._threshold_bounds = thresh(data=baseline_values)

    def _compute_chunk_metric(self, chunk_data: NDArray[np.float32]) -> float:
        """Compute the drift metric for a single chunk.

        Must be implemented by every detector that mixes in
        :class:`DriftChunkerMixin`.
        """
        raise NotImplementedError

    def _predict_chunked(
        self,
        x_test: NDArray[np.float32] | None = None,
        chunks_override: list[NDArray[np.float32]] | None = None,
        chunk_indices_override: list[list[int]] | None = None,
    ) -> DriftOutput[pl.DataFrame]:
        lower, upper = self._threshold_bounds
        chunk_results: list[ChunkResult] = []

        if chunks_override is not None:
            for i, chunk_data in enumerate(chunks_override):
                value = self._compute_chunk_metric(chunk_data)
                alert = (upper is not None and value > upper) or (lower is not None and value < lower)
                chunk_results.append(
                    ChunkResult(
                        key=f"chunk_{i}",
                        index=i,
                        start_index=-1,
                        end_index=-1,
                        value=value,
                        upper_threshold=upper,
                        lower_threshold=lower,
                        drifted=alert,
                    )
                )
        else:
            if x_test is None:
                raise ValueError("data is required for chunked prediction.")

            if chunk_indices_override is not None:
                index_groups = [np.asarray(idx, dtype=np.intp) for idx in chunk_indices_override]
            elif self._chunker is not None:
                index_groups = self._chunker.split(len(x_test))
            else:
                raise ValueError("No chunking specification provided.")

            for i, indices in enumerate(index_groups):
                value = self._compute_chunk_metric(x_test[indices])
                chunk_results.append(_make_chunk_result(i, indices, value, upper, lower))

        df = _chunk_results_to_dataframe(chunk_results)

        if df.is_empty():
            drifted = False
            threshold = 0.0
            distance = 0.0
        else:
            drifted = bool(df["drifted"].any())
            upper_val = df["upper_threshold"].cast(pl.Float64).to_list()[0]
            threshold = float(upper_val) if upper_val is not None else 0.0
            mean_val = df["value"].cast(pl.Float64).mean()
            distance = float(mean_val) if isinstance(mean_val, (int, float)) else 0.0

        return DriftOutput(
            drifted=drifted,
            threshold=threshold,
            distance=distance,
            metric_name=self._metric_name,
            details=df,
        )

    @property
    def is_chunked(self) -> bool:
        """Whether the detector is operating in chunked mode."""
        return self._chunker is not None


class DriftAdaptiveMixin:
    """Mixin for detectors that support feature extraction and reference updating.

    Provides lazy encoding of raw data via an optional
    :class:`~dataeval.protocols.FeatureExtractor` and post-predict
    reference updating via an :class:`~dataeval.protocols.UpdateStrategy`.

    Used by :class:`DriftUnivariate` and :class:`DriftMMD`.
    """

    extractor: FeatureExtractor | None
    update_strategy: UpdateStrategy | None
    _data: Any
    n: int

    def _init_adaptive(
        self,
        extractor: FeatureExtractor | None = None,
        update_strategy: UpdateStrategy | None = None,
    ) -> None:
        """Validate and store adaptive parameters.  Call from ``__init__``."""
        if update_strategy is not None and not isinstance(update_strategy, UpdateStrategy):
            raise ValueError("`update_strategy` is not a valid UpdateStrategy class.")
        if extractor is not None and not isinstance(extractor, FeatureExtractor):
            raise ValueError("`extractor` is not a valid FeatureExtractor.")

        self.extractor = extractor
        self.update_strategy = update_strategy
        self._data = None
        self.n = 0

    def _set_adaptive_data(self, data: Any) -> None:
        """Store raw data for lazy encoding.  Call from ``fit()``."""
        if self.extractor is None and not isinstance(data, Array):
            raise ValueError("`data` must be Array-like or provide an `extractor` to convert your data to an array.")

        self._data = data
        self._x_ref: NDArray[np.float32] | None = None  # reset lazy cache

        if self.extractor is not None:
            self.n = len(self.extractor(data))
        else:
            self.n = len(data)

    def _encode(self, data: Any) -> NDArray[np.float32]:
        """Encode input data: apply extractor if configured, then flatten."""
        if self.extractor is not None:
            data = self.extractor(data)
        return flatten_samples(np.asarray(data, dtype=np.float32))

    @property
    def x_ref(self) -> NDArray[np.float32]:
        """Reference data, lazily encoded on first access.

        Overrides :attr:`BaseDrift.x_ref` via MRO when this mixin
        appears before :class:`BaseDrift` in the inheritance list.
        """
        if self._x_ref is None:
            if self._data is None:
                raise RuntimeError("Must call fit() before accessing x_ref.")
            self._x_ref = self._encode(self._data)
        return self._x_ref

    def _apply_update_strategy(self, data: Any) -> None:
        """Update reference data after prediction if a strategy is set."""
        if self.update_strategy is not None:
            self._x_ref = self.update_strategy(self.x_ref, self._encode(data))
            self.n += len(data)
