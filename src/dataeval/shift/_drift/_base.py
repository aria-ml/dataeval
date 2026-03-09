"""Base classes and mixins for drift detection."""

__all__ = []

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
import polars as pl
from numpy.typing import NDArray
from typing_extensions import Self

from dataeval.exceptions import NotFittedError
from dataeval.protocols import Array, FeatureExtractor, Threshold, UpdateStrategy
from dataeval.shift._drift._chunk import BaseChunker, SizeChunker, resolve_chunker
from dataeval.types import DictOutput, Evaluator, set_metadata
from dataeval.utils._internal import flatten_samples
from dataeval.utils.thresholds import ZScoreThreshold

TDetails = TypeVar("TDetails", Mapping[str, Any], pl.DataFrame)


@dataclass(frozen=True, repr=False)
class DriftOutput(DictOutput, Generic[TDetails]):
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
    details : TDetails
        Detector-specific statistics (TypedDict) for non-chunked mode,
        or a :class:`polars.DataFrame` of per-chunk results for chunked mode.
    """

    drifted: bool
    threshold: float
    distance: float
    metric_name: str
    details: TDetails


@dataclass(frozen=True, repr=False)
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


class BaseDrift(Evaluator, ABC, Generic[TDetails]):
    """Abstract base for all drift detectors.

    Provides common state for the fit/predict lifecycle: a fitted flag
    and cached reference data. Use :meth:`chunked` to create a chunked
    wrapper for any drift detector.

    Subclasses must implement :meth:`fit` and :meth:`predict`.
    To support chunked mode, also mix in :class:`ChunkableMixin`.
    """

    _metric_name: str

    def __init__(self) -> None:
        super().__init__()
        self._fitted: bool = False
        self._reference_data: NDArray[np.float32] | None = None

    def _repr_extras(self) -> dict[str, Any]:
        """Append fitted status to repr."""
        return {"fitted": self._fitted}

    @property
    def reference_data(self) -> NDArray[np.float32]:
        """Reference data for drift detection.

        Returns
        -------
        NDArray[np.float32]
            Reference data array.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        """
        if self._reference_data is None:
            raise NotFittedError("Must call fit() before accessing reference_data.")
        return self._reference_data

    @abstractmethod
    def fit(self, reference_data: Any) -> Self:
        """Fit the detector on reference data.

        Parameters
        ----------
        reference_data : Any
            Reference dataset for drift comparison.

        Returns
        -------
        Self
        """
        ...

    @abstractmethod
    def predict(self, data: Any, /) -> DriftOutput[TDetails]:
        """Predict whether data has drifted from the reference.

        Parameters
        ----------
        data : Any
            Test dataset to evaluate.

        Returns
        -------
        DriftOutput
        """
        ...

    def _prepare_data(self, data: Any) -> NDArray[np.float32]:
        """Prepare raw input data for drift detection.

        Subclasses may override to add flattening, encoding, etc.
        """
        return np.atleast_2d(np.asarray(data, dtype=np.float32))

    def chunked(
        self,
        chunker: BaseChunker | None = None,
        chunk_size: int | None = None,
        chunk_count: int | None = None,
        threshold: Threshold | None = None,
    ) -> "ChunkedDrift[TDetails]":
        """Create a chunked wrapper around this drift detector.

        Returns a :class:`ChunkedDrift` that splits data into chunks
        during fit and predict, computing per-chunk metrics and comparing
        against baseline thresholds.

        Parameters
        ----------
        chunker : BaseChunker or None, default None
            Explicit chunker instance.
        chunk_size : int or None, default None
            Create fixed-size chunks of this many samples.
        chunk_count : int or None, default None
            Split into this many equal chunks.
        threshold : Threshold or None, default None
            Threshold strategy for determining drift bounds from baseline.
            When None, uses the detector's default threshold.

        Returns
        -------
        ChunkedDrift[TDetails]
            A chunked drift wrapper around this detector.
        """
        return ChunkedDrift(self, chunker=chunker, chunk_size=chunk_size, chunk_count=chunk_count, threshold=threshold)


class ChunkableMixin(ABC):
    """Mixin providing chunked-mode hooks for drift detectors.

    Detectors that support chunked evaluation should inherit from this
    mixin and implement :meth:`_compute_chunk_metric`. Optionally override
    :meth:`_compute_chunk_baselines` for chunk-vs-rest patterns and
    :meth:`_default_chunk_threshold` for detector-specific defaults.

    Used by :class:`ChunkedDrift` during fit and predict.
    """

    @abstractmethod
    def _compute_chunk_metric(self, chunk_data: NDArray[np.float32]) -> float:
        """Compute the drift metric for a single chunk of test data.

        Called by :class:`ChunkedDrift` during prediction to score each
        chunk against the fitted reference data.

        Parameters
        ----------
        chunk_data : NDArray[np.float32]
            A chunk of test data to score.

        Returns
        -------
        float
            Scalar metric value for this chunk.
        """
        ...

    def _compute_chunk_baselines(
        self,
        chunks: list[NDArray[np.float32]],
    ) -> NDArray[np.float32]:
        """Compute baseline metric values from reference data chunks.

        Called by :class:`ChunkedDrift` during fit to establish the
        baseline distribution of chunk metrics.

        Default implementation scores each chunk independently via
        :meth:`_compute_chunk_metric`. Override for chunk-vs-rest
        patterns (e.g., Univariate, MMD, DomainClassifier).

        Parameters
        ----------
        chunks : list[NDArray[np.float32]]
            Reference data split into chunks.

        Returns
        -------
        NDArray[np.float32]
            Baseline metric values, one per chunk.
        """
        return np.array([self._compute_chunk_metric(c) for c in chunks], dtype=np.float32)

    def _default_chunk_threshold(self) -> Threshold:
        """Return the default threshold strategy for chunked mode.

        Override to provide detector-specific defaults. The base
        implementation returns :class:`ZScoreThreshold`.
        """
        return ZScoreThreshold()


class ChunkedDrift(Generic[TDetails]):
    """Chunked drift detection wrapper.

    Wraps a :class:`BaseDrift` detector that also inherits
    :class:`ChunkableMixin` to perform chunked evaluation.
    During :meth:`fit`, splits reference data into chunks and computes
    baseline metric values. During :meth:`predict`, splits test data
    into chunks, scores each against the fitted reference, and compares
    to threshold bounds.

    Typically created via :meth:`BaseDrift.chunked` rather than directly.

    Parameters
    ----------
    detector : BaseDrift
        The underlying drift detector (must also be a :class:`ChunkableMixin`).
    chunker : BaseChunker or None, default None
        Explicit chunker instance.
    chunk_size : int or None, default None
        Create fixed-size chunks.
    chunk_count : int or None, default None
        Split into this many equal chunks.
    threshold : Threshold or None, default None
        Threshold strategy for drift bounds.
    """

    def __init__(
        self,
        detector: BaseDrift[TDetails],
        chunker: BaseChunker | None = None,
        chunk_size: int | None = None,
        chunk_count: int | None = None,
        threshold: Threshold | None = None,
    ) -> None:
        if not isinstance(detector, ChunkableMixin):
            raise TypeError(f"{type(detector).__name__} does not support chunked mode (missing ChunkableMixin).")
        self._detector: BaseDrift[TDetails] = detector
        self._chunkable: ChunkableMixin = detector
        self._threshold_override = threshold
        self._baseline_values: NDArray[np.float32] | None = None
        self._threshold_bounds: tuple[float | None, float | None] = (None, None)

        # Resolve chunker from convenience params
        resolved = resolve_chunker(chunker, chunk_size, chunk_count)
        if resolved is None:
            raise ValueError("Must provide chunker, chunk_size, or chunk_count.")
        self._init_chunker = resolved
        self._chunker: BaseChunker | None = None

    def __repr__(self) -> str:
        fitted = self._baseline_values is not None
        detector_repr = self._detector._repr(extras=False)
        return f"ChunkedDrift({detector_repr}, chunker={self._init_chunker!r}, fitted={fitted})"

    def fit(self, reference_data: Any, /) -> Self:
        """Fit the underlying detector and compute chunked baseline.

        Delegates to the underlying detector's ``fit()`` method, then
        splits the reference data into chunks and computes baseline
        metric values for threshold comparison.

        Parameters
        ----------
        reference_data : Any
            Reference dataset. Passed to the underlying detector's fit().

        Returns
        -------
        Self
        """
        # Fit underlying detector
        self._detector.fit(reference_data)

        # Get reference data and split into chunks
        x_ref = self._detector.reference_data
        n_ref = len(x_ref)
        index_groups = self._init_chunker.split(n_ref)
        chunks = [x_ref[idx] for idx in index_groups]

        # Compute baseline metrics
        baseline = self._chunkable._compute_chunk_baselines(chunks)

        # Derive threshold bounds
        threshold = self._threshold_override or self._chunkable._default_chunk_threshold()
        self._threshold_bounds = threshold(data=baseline)
        self._baseline_values = baseline

        # Normalize to SizeChunker so predict() uses the same chunk size
        fit_chunk_size = len(index_groups[0])
        self._chunker = SizeChunker(fit_chunk_size, incomplete="append")

        return self

    @set_metadata
    def predict(
        self,
        data: Any = None,
        chunks: list[Any] | None = None,
        chunk_indices: list[list[int]] | None = None,
    ) -> DriftOutput[pl.DataFrame]:
        """Predict drift using chunked evaluation.

        Splits test data into chunks, computes per-chunk metrics, and
        compares against baseline thresholds.

        Parameters
        ----------
        data : Any, optional
            Test dataset to analyze. Split into chunks using the fitted
            chunker. Required unless ``chunks`` is provided.
        chunks : list[Any] or None, default None
            Pre-built test data chunks. When provided, each array is
            treated as a separate chunk and ``data`` is ignored.
        chunk_indices : list[list[int]] or None, default None
            Index groupings for chunking ``data``. Each inner list
            specifies which samples from ``data`` belong to a chunk.

        Returns
        -------
        DriftChunkedOutput
            Per-chunk results with a :class:`polars.DataFrame` in ``details``.
        """
        if self._baseline_values is None:
            raise NotFittedError("Must call fit() before predict().")

        lower, upper = self._threshold_bounds
        chunk_results: list[ChunkResult] = []

        if chunks is not None:
            prepared = self._prepare_chunks(chunks)
            for i, chunk_data in enumerate(prepared):
                value = self._chunkable._compute_chunk_metric(chunk_data)
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
            if data is None:
                raise ValueError("data is required for chunked prediction.")

            x_test = self._prepare_data(data)

            if chunk_indices is not None:
                index_groups = [np.asarray(idx, dtype=np.intp) for idx in chunk_indices]
            elif self._chunker is not None:
                index_groups = self._chunker.split(len(x_test))
            else:
                raise ValueError("No chunking specification provided.")

            for i, indices in enumerate(index_groups):
                value = self._chunkable._compute_chunk_metric(x_test[indices])
                chunk_results.append(_make_chunk_result(i, indices, value, upper, lower))

        df = _chunk_results_to_dataframe(chunk_results)

        if df.is_empty():
            drifted = False
            threshold_val = 0.0
            distance = 0.0
        else:
            drifted = bool(df["drifted"].any())
            upper_val = df["upper_threshold"].cast(pl.Float64).to_list()[0]
            threshold_val = float(upper_val) if upper_val is not None else 0.0
            mean_val = df["value"].cast(pl.Float64).mean()
            distance = float(mean_val) if isinstance(mean_val, (int, float)) else 0.0

        return DriftOutput(
            drifted=drifted,
            threshold=threshold_val,
            distance=distance,
            metric_name=self._detector._metric_name,
            details=df,
        )

    def _prepare_data(self, data: Any) -> NDArray[np.float32]:
        """Prepare test data using the underlying detector's preparation."""
        return self._detector._prepare_data(data)

    def _prepare_chunks(self, chunks: list[Any]) -> list[NDArray[np.float32]]:
        """Prepare pre-built chunks using the underlying detector's preparation."""
        return [self._detector._prepare_data(c) for c in chunks]


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
        self._reference_data: NDArray[np.float32] | None = None  # reset lazy cache

        if self.extractor is not None:
            self.n = len(self.extractor(data))
        else:
            self.n = len(data)

    def _encode(self, data: Any) -> NDArray[np.float32]:
        """Encode input data: apply extractor if configured, then flatten."""
        if self.extractor is not None:
            data = self.extractor(data)
        return flatten_samples(np.asarray(data, dtype=np.float32))

    def _prepare_data(self, data: Any) -> NDArray[np.float32]:
        """Prepare data by encoding via extractor and flattening."""
        return self._encode(data)

    @property
    def reference_data(self) -> NDArray[np.float32]:
        """Reference data, lazily encoded on first access.

        Overrides :attr:`BaseDrift.reference_data` via MRO when this mixin
        appears before :class:`BaseDrift` in the inheritance list.
        """
        if self._reference_data is None:
            if self._data is None:
                raise NotFittedError("Must call fit() before accessing reference_data.")
            self._reference_data = self._encode(self._data)
        return self._reference_data

    def _apply_update_strategy(self, data: Any) -> None:
        """Update reference data after prediction if a strategy is set."""
        if self.update_strategy is not None:
            self._reference_data = self.update_strategy(self.reference_data, self._encode(data))
            self.n += len(data)
