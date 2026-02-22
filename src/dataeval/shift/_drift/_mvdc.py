"""
Multivariate Domain Classifier for drift detection.

Source code derived from NannyML 0.13.0
https://github.com/NannyML/nannyml/blob/main/nannyml/drift/multivariate/domain_classifier/calculator.py
https://github.com/NannyML/nannyml/blob/main/nannyml/base.py

Licensed under Apache Software License (Apache 2.0)
"""

__all__ = []

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from typing_extensions import Self

from dataeval.config import get_max_processes, get_seed
from dataeval.protocols import Threshold
from dataeval.shift._drift._base import (
    ChunkResult,
    DriftChunkedOutput,
    DriftMVDCStats,
    DriftOutput,
    _chunk_results_to_dataframe,
    _make_chunk_result,
)
from dataeval.shift._drift._chunk import (
    BaseChunker,
    SizeChunker,
    resolve_chunker,
)
from dataeval.types import set_metadata
from dataeval.utils.arrays import flatten_samples
from dataeval.utils.thresholds import ConstantThreshold

logger = logging.getLogger(__name__)


DEFAULT_LGBM_HYPERPARAMS = {
    "boosting_type": "gbdt",
    "class_weight": None,
    "colsample_bytree": 1.0,
    "deterministic": True,
    "importance_type": "split",
    "learning_rate": 0.1,
    "max_depth": -1,
    "min_child_samples": 20,
    "min_child_weight": 0.001,
    "min_split_gain": 0.0,
    "n_estimators": 100,
    "num_leaves": 31,
    "objective": None,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "subsample": 1.0,
    "subsample_for_bin": 200000,
    "subsample_freq": 0,
    "verbosity": -1,
}


def _compute_auroc(
    x: NDArray[np.float32],
    y: NDArray[np.intp],
    cv_folds_num: int = 5,
    hyperparameters: dict[str, Any] | None = None,
) -> tuple[float, NDArray[np.float32], NDArray[np.float32]]:
    """Compute AUROC of a domain classifier distinguishing two classes.

    Parameters
    ----------
    x : NDArray[np.float32]
        Combined feature matrix, shape (n_samples, n_features).
    y : NDArray[np.intp]
        Binary labels (0=reference, 1=test).
    cv_folds_num : int
        Number of stratified k-fold cross-validation splits.
    hyperparameters : dict or None
        LightGBM hyperparameters.

    Returns
    -------
    tuple[float, NDArray[np.float32], NDArray[np.float32]]
        - Overall AUROC score (0.5 = no discrimination, 1.0 = perfect).
        - Per-fold AUROC values, shape (cv_folds_num,).
        - Mean feature importances across folds, shape (n_features,).
    """
    hyperparameters = DEFAULT_LGBM_HYPERPARAMS if hyperparameters is None else hyperparameters
    feature_names = [f"f{i}" for i in range(x.shape[1])]

    skf = StratifiedKFold(n_splits=cv_folds_num)
    all_preds: list[NDArray[np.float32]] = []
    all_tgts: list[NDArray[np.intp]] = []
    fold_aurocs: list[float] = []
    fold_importances: list[NDArray[np.float32]] = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        for train_index, test_index in skf.split(x, y):
            _trx = x[train_index]
            _try = y[train_index]
            _tsx = x[test_index]
            _tsy = y[test_index]
            model = LGBMClassifier(
                **hyperparameters,
                n_jobs=get_max_processes(),
                random_state=get_seed(),
            )
            model.fit(_trx, _try, feature_name=feature_names)
            preds = np.asarray(model.predict_proba(_tsx), dtype=np.float32)[:, 1]
            all_preds.append(preds)
            all_tgts.append(_tsy)
            fold_auroc = roc_auc_score(_tsy, preds)
            fold_aurocs.append(0.5 if fold_auroc == np.nan else float(fold_auroc))
            fold_importances.append(np.asarray(model.feature_importances_, dtype=np.float32))

    np_all_preds = np.concatenate(all_preds, axis=0)
    np_all_tgts = np.concatenate(all_tgts, axis=0)
    result = roc_auc_score(np_all_tgts, np_all_preds)
    auroc = 0.5 if result == np.nan else float(result)
    return (
        auroc,
        np.array(fold_aurocs, dtype=np.float32),
        np.mean(fold_importances, axis=0).astype(np.float32),
    )


class DriftMVDC:
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
    config : DriftMVDC.Config or None, default None
        Optional configuration object with default parameters. Parameters
        specified directly in __init__ will override config defaults.

    Examples
    --------
    Non-chunked mode:

    >>> ref = np.random.randn(200, 4).astype(np.float32)
    >>> test = np.random.randn(100, 4).astype(np.float32)
    >>> detector = DriftMVDC().fit(ref)
    >>> result = detector.predict(test)
    >>> print(f"Drift: {result.drifted}")
    Drift: ...

    Chunked mode:

    >>> detector = DriftMVDC(threshold=(0.45, 0.65)).fit(ref, chunk_size=100)
    >>> result = detector.predict(test)

    Using configuration:

    >>> config = DriftMVDC.Config(n_folds=10, threshold=(0.4, 0.6))
    >>> detector = DriftMVDC(config=config)
    """

    @dataclass
    class Config:
        """
        Configuration for DriftMVDC detector.

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
        # Store config or create default
        self.config: DriftMVDC.Config = config or DriftMVDC.Config()

        # Use config defaults if parameters not specified
        n_folds = n_folds if n_folds is not None else self.config.n_folds
        threshold = threshold if threshold is not None else self.config.threshold

        self._n_folds = n_folds
        self._threshold_config = threshold
        self._fitted = False
        self._chunker: BaseChunker | None = None
        self._x_ref_np: NDArray[np.float32] | None = None
        self._baseline_values: NDArray[np.float32] | None = None
        self._threshold_bounds: tuple[float | None, float | None] = (None, None)

    @property
    def x_ref(self) -> NDArray[np.float32]:
        """Reference data as a numpy array."""
        if self._x_ref_np is None:
            raise RuntimeError("Must call fit() before accessing x_ref.")
        return self._x_ref_np

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
        self._x_ref_np = flatten_samples(np.atleast_2d(np.asarray(x_ref, dtype=np.float32)))
        self.n_features: int = self._x_ref_np.shape[1]

        # Handle prebuilt chunks as a direct path (no chunker stored)
        if chunks is not None:
            prebuilt = [flatten_samples(np.atleast_2d(np.asarray(c, dtype=np.float32))) for c in chunks]
            self._chunker = None
            self._fit_prebuilt(prebuilt)
            self._fitted = True
            return self

        # Resolve chunker from convenience params
        resolved = resolve_chunker(chunker, chunk_size, chunk_count, chunk_indices)

        if resolved is not None:
            self._fit_chunked(resolved)
            # Normalize to SizeChunker so predict() uses the same chunk
            # size regardless of test set size.
            n_ref = len(self._x_ref_np)
            fit_chunk_size = len(resolved.split(n_ref)[0])
            self._chunker = SizeChunker(fit_chunk_size, incomplete="append")
        else:
            self._chunker = None

        self._fitted = True
        return self

    def _resolve_threshold(self, baseline_values: NDArray[np.float32]) -> None:
        """Store baseline values and compute threshold bounds from config."""
        self._baseline_values = baseline_values

        threshold_config = self._threshold_config
        if isinstance(threshold_config, tuple):
            t_lower, t_upper = max(0.0, min(threshold_config)), min(1.0, max(threshold_config))
            threshold: Threshold = ConstantThreshold(lower=t_lower, upper=t_upper)
        else:
            t_lower = max(0.0, threshold_config - 0.1)
            t_upper = min(1.0, threshold_config + 0.1)
            threshold = ConstantThreshold(lower=t_lower, upper=t_upper)

        self._threshold_bounds = threshold(data=self._baseline_values)

    def _fit_chunked(self, chunker: BaseChunker) -> None:
        """Compute per-chunk AUROC on reference data to establish baseline."""
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
            auroc, _, _ = _compute_auroc(x, y, self._n_folds)
            baseline_values.append(auroc)

        self._resolve_threshold(np.array(baseline_values, dtype=np.float32))

    def _fit_prebuilt(self, chunks: list[NDArray[np.float32]]) -> None:
        """Compute per-chunk AUROC from prebuilt reference chunks."""
        baseline_values: list[float] = []
        for i, chunk_data in enumerate(chunks):
            rest_data = np.concatenate([c for j, c in enumerate(chunks) if j != i], axis=0)
            x = np.concatenate([rest_data, chunk_data], axis=0)
            y = np.concatenate([np.zeros(len(rest_data), dtype=np.intp), np.ones(len(chunk_data), dtype=np.intp)])
            auroc, _, _ = _compute_auroc(x, y, self._n_folds)
            baseline_values.append(auroc)

        self._resolve_threshold(np.array(baseline_values, dtype=np.float32))

    @set_metadata
    def predict(
        self,
        x: ArrayLike | None = None,
        chunks: list[ArrayLike] | None = None,
        chunk_indices: list[list[int]] | None = None,
    ) -> DriftOutput | DriftChunkedOutput:
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
        DriftOutput or DriftChunkedOutput
            Non-chunked mode returns :class:`DriftOutput`.
            Chunked mode returns :class:`DriftChunkedOutput`.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict().")

        if self._chunker is not None or chunks is not None or chunk_indices is not None:
            return self._predict_chunked(x, chunks, chunk_indices)

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
        auroc, fold_aurocs, feature_importances = _compute_auroc(x_combined, y, cv_folds_num=self._n_folds)

        threshold = self._threshold_config if isinstance(self._threshold_config, float) else 0.55
        drifted = auroc > threshold

        return DriftOutput(
            drifted=drifted,
            threshold=threshold,
            p_val=auroc,
            distance=auroc,
            metric_name="auroc",
            stats=DriftMVDCStats(fold_aurocs=fold_aurocs, feature_importances=feature_importances),
        )

    def _predict_chunked(
        self,
        x: ArrayLike | None = None,
        chunks_override: list[ArrayLike] | None = None,
        chunk_indices_override: list[list[int]] | None = None,
    ) -> DriftChunkedOutput:
        """Chunked prediction: per-chunk AUROC vs baseline threshold."""
        x_ref = self.x_ref
        lower, upper = self._threshold_bounds
        chunk_results: list[ChunkResult] = []

        if chunks_override is not None:
            # Direct prebuilt path
            for i, chunk_arr in enumerate(chunks_override):
                chunk_data = flatten_samples(np.atleast_2d(np.asarray(chunk_arr, dtype=np.float32)))
                if chunk_data.shape[1] != self.n_features:
                    raise ValueError("Reference and test embeddings have different number of features")
                x_combined = np.concatenate([x_ref, chunk_data], axis=0)
                y = np.concatenate([np.zeros(len(x_ref), dtype=np.intp), np.ones(len(chunk_data), dtype=np.intp)])
                auroc, _, _ = _compute_auroc(x_combined, y, cv_folds_num=self._n_folds)
                alert = (upper is not None and auroc > upper) or (lower is not None and auroc < lower)
                chunk_results.append(
                    ChunkResult(
                        key=f"chunk_{i}",
                        index=i,
                        start_index=-1,
                        end_index=-1,
                        value=auroc,
                        upper_threshold=upper,
                        lower_threshold=lower,
                        drifted=alert,
                    )
                )
        else:
            if x is None:
                raise ValueError("x is required for chunked prediction.")
            x_test = flatten_samples(np.atleast_2d(np.asarray(x, dtype=np.float32)))
            if x_test.shape[1] != self.n_features:
                raise ValueError("Reference and test embeddings have different number of features")

            if chunk_indices_override is not None:
                index_groups = [np.asarray(idx, dtype=np.intp) for idx in chunk_indices_override]
            elif self._chunker is not None:
                index_groups = self._chunker.split(len(x_test))
            else:
                raise ValueError("No chunking specification provided.")

            for i, indices in enumerate(index_groups):
                chunk_data = x_test[indices]
                x_combined = np.concatenate([x_ref, chunk_data], axis=0)
                y = np.concatenate([np.zeros(len(x_ref), dtype=np.intp), np.ones(len(chunk_data), dtype=np.intp)])
                auroc, _, _ = _compute_auroc(x_combined, y, cv_folds_num=self._n_folds)
                chunk_results.append(_make_chunk_result(i, indices, auroc, upper, lower))

        return DriftChunkedOutput(
            metric_name="auroc",
            chunk_results=_chunk_results_to_dataframe(chunk_results),
        )
