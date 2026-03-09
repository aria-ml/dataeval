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
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from dataeval.exceptions import NotFittedError, ShapeMismatchError
from dataeval.protocols import FeatureExtractor, Threshold, UpdateStrategy
from dataeval.shift._drift._base import BaseDrift, ChunkableMixin, DriftAdaptiveMixin, DriftOutput
from dataeval.shift._shared._domain_classifier import compute_auroc
from dataeval.types import set_metadata
from dataeval.utils.thresholds import ConstantThreshold

logger = logging.getLogger(__name__)


class _DriftDomainClassifierStats(TypedDict):
    p_val: float
    fold_aurocs: NDArray[np.float32]
    feature_importances: NDArray[np.float32]


class DriftDomainClassifier(DriftAdaptiveMixin, ChunkableMixin, BaseDrift[_DriftDomainClassifierStats]):
    """Multivariate Domain Classifier based drift detector.

    Detects drift by training a LightGBM classifier to distinguish between
    reference and test data. If the classifier can discriminate well (high AUROC),
    the distributions differ and drift is detected.

    Uses a fit/predict lifecycle: construct with hyperparameters, call
    :meth:`fit` with reference data, then call :meth:`predict` with test data.
    Use :meth:`chunked` to create a chunked wrapper for time-series monitoring.

    Supports two modes:

    - **Non-chunked** (default): Computes a single AUROC for the entire test set
      vs reference. Drift is flagged when AUROC exceeds the threshold (default 0.55).
    - **Chunked** (via :meth:`chunked`): Splits data into chunks, computes AUROC
      per chunk, and uses threshold bounds to flag drift per chunk.

    Parameters
    ----------
    n_folds : int, default 5
        Number of cross-validation (CV) folds.
    threshold : float or tuple[float, float], default 0.55
        For non-chunked mode: float threshold where AUROC > threshold means drift.
        For chunked mode: tuple (lower, upper) bounds on AUROC for identifying drift.
    extractor : FeatureExtractor or None, default None
        Feature extractor for transforming input data before drift detection.
        When provided, raw data is passed through the extractor before
        flattening and comparison. When None, data is used as-is.
    update_strategy : UpdateStrategy or None, default None
        Strategy for updating reference data when new data arrives.
        When None, reference data remains fixed throughout detection.
    config : DriftDomainClassifier.Config or None, default None
        Optional configuration object with default parameters. Parameters
        specified directly in __init__ will override config defaults.

    See Also
    --------
    :class:`DriftDomainClassifier.Stats` : Per-prediction statistics returned in :attr:`DriftOutput.details`.

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

    >>> chunked = DriftDomainClassifier(threshold=(0.45, 0.65)).chunked(chunk_size=100)
    >>> chunked.fit(ref)
    ChunkedDrift(DriftDomainClassifier(n_folds=5, threshold=(0.45, 0.65), extractor=None, update_strategy=None), chunker=SizeChunker(chunk_size=100, incomplete='keep'), fitted=True)
    >>> result = chunked.predict(test)

    Using configuration:

    >>> config = DriftDomainClassifier.Config(n_folds=10, threshold=(0.4, 0.6))
    >>> detector = DriftDomainClassifier(config=config)
    """  # noqa: E501

    class Stats(_DriftDomainClassifierStats):
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
        extractor : FeatureExtractor or None, default None
            Feature extractor for transforming input data before drift detection.
        update_strategy : UpdateStrategy or None, default None
            Strategy for updating reference data over time.
        """

        n_folds: int = 5
        threshold: float | tuple[float, float] = 0.55
        extractor: FeatureExtractor | None = None
        update_strategy: UpdateStrategy | None = None

    def __init__(
        self,
        n_folds: int | None = None,
        threshold: float | tuple[float, float] | None = None,
        extractor: FeatureExtractor | None = None,
        update_strategy: UpdateStrategy | None = None,
        config: Config | None = None,
    ) -> None:
        # Store config or create default
        base_config = config or DriftDomainClassifier.Config()

        # Use config defaults if parameters not specified
        n_folds = n_folds if n_folds is not None else base_config.n_folds
        threshold = threshold if threshold is not None else base_config.threshold
        extractor = extractor if extractor is not None else base_config.extractor
        update_strategy = update_strategy if update_strategy is not None else base_config.update_strategy

        self.config: DriftDomainClassifier.Config = DriftDomainClassifier.Config(
            n_folds=n_folds,
            threshold=threshold,
            extractor=extractor,
            update_strategy=update_strategy,
        )

        # Initialise base + mixins
        BaseDrift.__init__(self)
        self._init_adaptive(extractor=extractor, update_strategy=update_strategy)

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

    def fit(self, reference_data: Any) -> Self:
        """Fit the domain classifier on the reference data.

        Parameters
        ----------
        reference_data : Any
            Reference data. When an extractor is configured, this can be
            any data type accepted by the extractor. Otherwise, must be
            array-like with shape ``(n_samples, n_features)``.

        Returns
        -------
        Self
        """
        self._set_adaptive_data(reference_data)
        ref = self.reference_data  # lazily encoded + flattened
        self.n_features: int = ref.shape[1]
        self._fitted = True
        return self

    def _compute_chunk_metric(self, chunk_data: NDArray[np.float32]) -> float:
        """Compute AUROC: chunk (test) vs full reference."""
        x_ref = self.reference_data
        x_combined = np.concatenate([x_ref, chunk_data], axis=0)
        y = np.concatenate([np.zeros(len(x_ref), dtype=np.intp), np.ones(len(chunk_data), dtype=np.intp)])
        auroc, _, _ = compute_auroc(x_combined, y, n_folds=self._n_folds)
        return auroc

    def _compute_chunk_baselines(
        self,
        chunks: list[NDArray[np.float32]],
    ) -> NDArray[np.float32]:
        """Compute per-chunk AUROC on reference data (chunk vs rest)."""
        baseline_values: list[float] = []
        for i, chunk in enumerate(chunks):
            rest = np.concatenate([c for j, c in enumerate(chunks) if j != i], axis=0)
            x = np.concatenate([rest, chunk], axis=0)
            y = np.concatenate([np.zeros(len(rest), dtype=np.intp), np.ones(len(chunk), dtype=np.intp)])
            auroc, _, _ = compute_auroc(x, y, self._n_folds)
            baseline_values.append(auroc)
        return np.array(baseline_values, dtype=np.float32)

    def _default_chunk_threshold(self) -> Threshold:
        return self._make_default_threshold()

    @set_metadata
    def predict(self, data: Any) -> DriftOutput["DriftDomainClassifier.Stats"]:
        """
        Perform :term:`inference<Inference>` on the test data.

        Parameters
        ----------
        data : Any
            Test data. When an extractor is configured, this can be
            any data type accepted by the extractor. Otherwise, must be
            array-like.

        Returns
        -------
        DriftOutput[DriftDomainClassifier.Stats]
            Drift prediction with AUROC statistics.
        """
        if not self._fitted:
            raise NotFittedError("Must call fit() before predict().")

        x_test = self._prepare_data(data)
        if x_test.shape[1] != self.n_features:
            raise ShapeMismatchError("Reference and test embeddings have different number of features")

        x_ref = self.reference_data
        x_combined = np.concatenate([x_ref, x_test], axis=0)
        y = np.concatenate([np.zeros(len(x_ref), dtype=np.intp), np.ones(len(x_test), dtype=np.intp)])
        auroc, fold_aurocs, feature_importances = compute_auroc(x_combined, y, n_folds=self._n_folds)

        threshold = self._threshold_config if isinstance(self._threshold_config, float) else 0.55
        drifted = auroc > threshold

        self._apply_update_strategy(data)

        return DriftOutput(
            drifted=drifted,
            threshold=threshold,
            distance=auroc,
            metric_name="auroc",
            details=_DriftDomainClassifierStats(
                p_val=auroc,
                fold_aurocs=fold_aurocs,
                feature_importances=feature_importances,
            ),
        )
