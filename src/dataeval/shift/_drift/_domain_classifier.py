"""
Multivariate Domain Classifier for drift detection.

Source code derived from NannyML 0.13.0
https://github.com/NannyML/nannyml/blob/main/nannyml/drift/multivariate/domain_classifier/calculator.py
https://github.com/NannyML/nannyml/blob/main/nannyml/base.py

Licensed under Apache Software License (Apache 2.0)
"""

__all__ = []

import logging
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Self

from dataeval.protocols import Threshold
from dataeval.shift._drift._base import BaseDrift, DriftChunkerMixin, DriftOutput
from dataeval.shift._drift._chunk import BaseChunker
from dataeval.shift._shared._domain_classifier import compute_auroc
from dataeval.types import set_metadata
from dataeval.utils.arrays import flatten_samples
from dataeval.utils.thresholds import ConstantThreshold

logger = logging.getLogger(__name__)


class DriftDomainClassifierStats(TypedDict):
    """Statistics from Multivariate Domain Classifier drift detection.

    Attributes
    ----------
    p_val : float
        AUROC score used as proxy for drift significance.
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

    p_val: float
    fold_aurocs: NDArray[np.float32]
    feature_importances: NDArray[np.float32]


class DriftDomainClassifier(DriftChunkerMixin, BaseDrift):
    """Multivariate Domain Classifier based drift detector.

    Detects drift by training a LightGBM classifier to distinguish between
    reference and test data. If the classifier can discriminate well (high AUROC),
    the distributions differ and drift is detected.

    Uses a fit/predict lifecycle: construct with hyperparameters, call
    :meth:`fit` with reference data, then call :meth:`predict` with test data.

    Supports two modes:

    - **Non-chunked** (default): Computes a single AUROC for the entire test set
      vs reference. Drift is flagged when AUROC exceeds the threshold (default 0.55).
    - **Chunked**: Splits data into chunks, computes AUROC per chunk, and uses
      threshold bounds to flag drift per chunk. Enable by passing chunking
      parameters to :meth:`fit`.

    Parameters
    ----------
    n_folds : int, default 5
        Number of cross-validation (CV) folds.
    threshold : float or tuple[float, float], default 0.55
        For non-chunked mode: float threshold where AUROC > threshold means drift.
        For chunked mode: tuple (lower, upper) bounds on AUROC for identifying drift.
    config : DriftDomainClassifier.Config or None, default None
        Optional configuration object with default parameters. Parameters
        specified directly in __init__ will override config defaults.

    Examples
    --------
    Non-chunked mode:

    >>> ref = np.random.randn(200, 4).astype(np.float32)
    >>> test = np.random.randn(100, 4).astype(np.float32)
    >>> detector = DriftDomainClassifier().fit(ref)
    >>> result = detector.predict(test)
    >>> print(f"Drift: {result.drifted}")
    Drift: ...

    Chunked mode:

    >>> detector = DriftDomainClassifier(threshold=(0.45, 0.65)).fit(ref, chunk_size=100)
    >>> result = detector.predict(test)

    Using configuration:

    >>> config = DriftDomainClassifier.Config(n_folds=10, threshold=(0.4, 0.6))
    >>> detector = DriftDomainClassifier(config=config)
    """

    @dataclass
    class Config:
        """
        Configuration for DriftDomainClassifier detector.

        Attributes
        ----------
        n_folds : int, default 5
            Number of cross-validation folds.
        threshold : float or tuple[float, float], default 0.55
            Threshold for drift detection.
        """

        n_folds: int = 5
        threshold: float | tuple[float, float] = 0.55

    def __init__(
        self,
        n_folds: int | None = None,
        threshold: float | tuple[float, float] | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__()
        self._init_chunking()

        # Store config or create default
        self.config: DriftDomainClassifier.Config = config or DriftDomainClassifier.Config()

        # Use config defaults if parameters not specified
        n_folds = n_folds if n_folds is not None else self.config.n_folds
        threshold = threshold if threshold is not None else self.config.threshold

        self._n_folds = n_folds
        self._threshold_config = threshold
        self._metric_name = "auroc"

    def _make_default_threshold(self) -> Threshold:
        """Convert threshold config (float or tuple) to a ConstantThreshold."""
        tc = self._threshold_config
        if isinstance(tc, tuple):
            t_lower = max(0.0, min(tc))
            t_upper = min(1.0, max(tc))
        else:
            t_lower = max(0.0, tc - 0.1)
            t_upper = min(1.0, tc + 0.1)
        return ConstantThreshold(lower=t_lower, upper=t_upper)

    def fit(
        self,
        x_ref: ArrayLike,
        chunker: BaseChunker | None = None,
        chunk_size: int | None = None,
        chunk_count: int | None = None,
        chunks: list[ArrayLike] | None = None,
        chunk_indices: list[list[int]] | None = None,
    ) -> Self:
        """Fit the domain classifier on the reference data.

        When chunking is enabled, the detector computes per-chunk baseline
        AUROC values from the reference data and derives threshold bounds.
        During prediction, the test data is split into chunks of the
        **same size** used here, so that per-chunk statistics are comparable
        to the baseline.

        If ``chunk_count`` is provided, the effective chunk size is computed
        as ``len(x_ref) // chunk_count`` and locked in for prediction.  Use
        ``chunk_size`` directly when you want explicit control over the
        chunk size used for both fitting and prediction.

        Parameters
        ----------
        x_ref : ArrayLike
            Reference data with dim[n_samples, n_features].
        chunker : ArrayChunker or None, default None
            Explicit chunker instance for chunked mode.
        chunk_size : int or None, default None
            Create fixed-size chunks. The same size is used during
            prediction to keep statistics comparable.
        chunk_count : int or None, default None
            Split into this many equal chunks. Converted to a fixed
            ``chunk_size`` based on the reference data length.
        chunks : list[ArrayLike] or None, default None
            Pre-split reference data for chunked mode.
        chunk_indices : list[list[int]] or None, default None
            Index groupings for chunking reference data.

        Returns
        -------
        Self
        """
        self._x_ref = flatten_samples(np.atleast_2d(np.asarray(x_ref, dtype=np.float32)))
        self.n_features: int = self._x_ref.shape[1]

        # Handle chunking (prebuilt chunks are converted here)
        if chunks is not None:
            chunks = [flatten_samples(np.atleast_2d(np.asarray(c, dtype=np.float32))) for c in chunks]

        self._resolve_fit_chunks(
            len(self._x_ref),
            chunker=chunker,
            chunk_size=chunk_size,
            chunk_count=chunk_count,
            chunks=chunks,
            chunk_indices=chunk_indices,
            default_threshold=self._make_default_threshold(),
        )

        self._fitted = True
        return self

    def _fit_chunked_baseline(
        self,
        chunker: BaseChunker,
        threshold: Threshold | None,
        default_threshold: Threshold | None,
    ) -> None:
        """Compute per-chunk AUROC on reference data (chunk vs rest)."""
        x_ref = self.x_ref
        n = len(x_ref)
        index_groups = chunker.split(n)

        baseline_values: list[float] = []
        for indices in index_groups:
            mask = np.ones(n, dtype=bool)
            mask[indices] = False
            rest_data = x_ref[mask]
            chunk_data = x_ref[indices]
            x = np.concatenate([rest_data, chunk_data], axis=0)
            y = np.concatenate([np.zeros(len(rest_data), dtype=np.intp), np.ones(len(chunk_data), dtype=np.intp)])
            auroc, _, _ = compute_auroc(x, y, self._n_folds)
            baseline_values.append(auroc)

        self._resolve_baseline_threshold(np.array(baseline_values, dtype=np.float32), threshold, default_threshold)

    def _fit_prebuilt_baseline(
        self,
        chunks: list[NDArray[np.float32]],
        threshold: Threshold | None,
        default_threshold: Threshold | None,
    ) -> None:
        """Compute per-chunk AUROC from prebuilt reference chunks (chunk vs rest)."""
        baseline_values: list[float] = []
        for i, chunk_data in enumerate(chunks):
            rest_data = np.concatenate([c for j, c in enumerate(chunks) if j != i], axis=0)
            x = np.concatenate([rest_data, chunk_data], axis=0)
            y = np.concatenate([np.zeros(len(rest_data), dtype=np.intp), np.ones(len(chunk_data), dtype=np.intp)])
            auroc, _, _ = compute_auroc(x, y, self._n_folds)
            baseline_values.append(auroc)

        self._resolve_baseline_threshold(np.array(baseline_values, dtype=np.float32), threshold, default_threshold)

    def _compute_chunk_metric(self, chunk_data: NDArray[np.float32]) -> float:
        """Compute AUROC: chunk (test) vs full reference."""
        x_ref = self.x_ref
        x_combined = np.concatenate([x_ref, chunk_data], axis=0)
        y = np.concatenate([np.zeros(len(x_ref), dtype=np.intp), np.ones(len(chunk_data), dtype=np.intp)])
        auroc, _, _ = compute_auroc(x_combined, y, n_folds=self._n_folds)
        return auroc

    @set_metadata
    def predict(
        self,
        x: ArrayLike | None = None,
        chunks: list[ArrayLike] | None = None,
        chunk_indices: list[list[int]] | None = None,
    ) -> DriftOutput:
        """
        Perform :term:`inference<Inference>` on the test data.

        Parameters
        ----------
        x : ArrayLike or None
            Test (analysis) data with dim[n_samples, n_features].
            Required for non-chunked mode and chunked mode unless
            pre-built chunks are provided.
        chunks : list[ArrayLike] or None, default None
            Pre-built test data chunks.
        chunk_indices : list[list[int]] or None, default None
            Index groupings for chunking test data.

        Returns
        -------
        DriftOutput
            Non-chunked mode: ``details`` is a :class:`DriftDomainClassifierStats` TypedDict.
            Chunked mode: ``details`` is a :class:`polars.DataFrame` with per-chunk results.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict().")

        if self.is_chunked or chunks is not None or chunk_indices is not None:
            if chunks is not None:
                prepared = [flatten_samples(np.atleast_2d(np.asarray(c, dtype=np.float32))) for c in chunks]
                for c in prepared:
                    if c.shape[1] != self.n_features:
                        raise ValueError("Reference and test embeddings have different number of features")
                return self._predict_chunked(chunks_override=prepared)

            x_test = None
            if x is not None:
                x_test = flatten_samples(np.atleast_2d(np.asarray(x, dtype=np.float32)))
                if x_test.shape[1] != self.n_features:
                    raise ValueError("Reference and test embeddings have different number of features")

            return self._predict_chunked(
                x_test=x_test,
                chunk_indices_override=chunk_indices,
            )

        if x is None:
            raise ValueError("x is required for non-chunked prediction.")
        return self._predict_single(x)

    def _predict_single(self, x: ArrayLike) -> DriftOutput:
        """Non-chunked prediction: single AUROC for entire test set."""
        x_test = flatten_samples(np.atleast_2d(np.asarray(x, dtype=np.float32)))
        if x_test.shape[1] != self.n_features:
            raise ValueError("Reference and test embeddings have different number of features")

        x_ref = self.x_ref
        x_combined = np.concatenate([x_ref, x_test], axis=0)
        y = np.concatenate([np.zeros(len(x_ref), dtype=np.intp), np.ones(len(x_test), dtype=np.intp)])
        auroc, fold_aurocs, feature_importances = compute_auroc(x_combined, y, n_folds=self._n_folds)

        threshold = self._threshold_config if isinstance(self._threshold_config, float) else 0.55
        drifted = auroc > threshold

        return DriftOutput(
            drifted=drifted,
            threshold=threshold,
            distance=auroc,
            metric_name="auroc",
            details=DriftDomainClassifierStats(
                p_val=auroc,
                fold_aurocs=fold_aurocs,
                feature_importances=feature_importances,
            ),
        )
