"""Reconstruction-based (AE/VAE) drift detector."""

__all__ = []

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm
from typing_extensions import Self

from dataeval.protocols import DeviceLike, EvidenceLowerBoundLossFn, ReconstructionLossFn, Threshold
from dataeval.shift._drift._base import BaseDrift, DriftChunkerMixin, DriftOutput
from dataeval.shift._drift._chunk import BaseChunker
from dataeval.shift._shared._reconstruction import ReconstructionScorer
from dataeval.types import set_metadata
from dataeval.utils.thresholds import ZScoreThreshold


class DriftReconstructionStats(TypedDict):
    """Statistics from reconstruction-based drift detection.

    Attributes
    ----------
    p_val : float
        P-value from z-test on reconstruction errors, between 0 and 1.
    mean_ref_error : float
        Mean reconstruction error on the reference set (baseline).
    mean_test_error : float
        Mean reconstruction error for the test samples.
    """

    p_val: float
    mean_ref_error: float
    mean_test_error: float


class DriftReconstruction(DriftChunkerMixin, BaseDrift):
    """Reconstruction-based drift detector using autoencoders.

    Detects drift by comparing reconstruction errors: if the model (trained
    on reference data) produces higher reconstruction errors on test data,
    the test distribution has likely shifted.

    Uses a fit/predict lifecycle: construct with model and hyperparameters,
    call :meth:`fit` with reference data (trains the model), then call
    :meth:`predict` with test data.

    Supports two modes:

    - **Non-chunked** (default): Computes mean reconstruction error for the
      test set and uses a z-test against the reference baseline.
    - **Chunked**: Splits data into chunks, computes mean reconstruction
      error per chunk, and uses threshold bounds to flag drift.

    Parameters
    ----------
    model : torch.nn.Module
        Autoencoder or VAE model.
    device : DeviceLike or None, default None
        Hardware device.
    model_type : {"ae", "vae", "auto"} or None, default "auto"
        Model type. ``"auto"`` auto-detects.
    use_gmm : bool or None, default None
        Whether to use GMM in latent space.
    p_val : float, default 0.05
        Significance threshold for non-chunked mode.
    config : DriftReconstruction.Config or None, default None
        Optional configuration object.

    Examples
    --------
    >>> from dataeval.utils.models import AE
    >>> import torch
    >>> model = AE(input_shape=(1, 28, 28))
    >>> ref = torch.rand(100, 1, 28, 28).numpy()
    >>> detector = DriftReconstruction(model).fit(ref)
    >>> test = torch.rand(50, 1, 28, 28).numpy()
    >>> result = detector.predict(test)
    """

    @dataclass
    class Config:
        """
        Configuration for DriftReconstruction detector.

        Attributes
        ----------
        p_val : float, default 0.05
            Significance threshold for non-chunked mode.
        loss_fn : Callable or None, default None
            Loss function for training.
        optimizer : torch.optim.Optimizer or None, default None
            Optimizer for training.
        epochs : int, default 20
            Number of training epochs.
        batch_size : int, default 64
            Batch size for training and scoring.
        gmm_weight : float, default 0.5
            Weight for GMM component.
        gmm_score_mode : {"standardized", "percentile"}, default "standardized"
            Method for combining reconstruction and GMM scores.
        """

        p_val: float = 0.05
        loss_fn: Callable[..., torch.Tensor] | None = None
        optimizer: torch.optim.Optimizer | None = None
        epochs: int = 20
        batch_size: int = 64
        gmm_weight: float = 0.5
        gmm_score_mode: Literal["standardized", "percentile"] = "standardized"

    def __init__(
        self,
        model: torch.nn.Module,
        device: DeviceLike | None = None,
        model_type: Literal["ae", "vae", "auto"] | None = "auto",
        use_gmm: bool | None = None,
        p_val: float | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__()
        self._init_chunking()

        self.config: DriftReconstruction.Config = config or DriftReconstruction.Config()

        self._p_val = p_val if p_val is not None else self.config.p_val

        self._scorer = ReconstructionScorer(
            model=model,
            device=device,
            model_type=model_type,
            use_gmm=use_gmm,
            gmm_weight=self.config.gmm_weight,
            gmm_score_mode=self.config.gmm_score_mode,
        )

        self._metric_name = "reconstruction_error"
        self._ref_mean: float = 0.0
        self._ref_std: float = 1.0
        self._score_batch_size: int = int(1e10)

    def fit(
        self,
        x_ref: ArrayLike,
        loss_fn: ReconstructionLossFn | EvidenceLowerBoundLossFn | Callable[..., torch.Tensor] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        chunker: BaseChunker | None = None,
        chunk_size: int | None = None,
        chunk_count: int | None = None,
        chunks: list[ArrayLike] | None = None,
        chunk_indices: list[list[int]] | None = None,
        threshold: Threshold | None = None,
    ) -> Self:
        """Fit the reconstruction drift detector.

        Trains the autoencoder on reference data, then optionally sets up
        chunked baseline.

        Parameters
        ----------
        x_ref : ArrayLike
            Reference data.
        loss_fn : Callable or None, default None
            Loss function for training.
        optimizer : torch.optim.Optimizer or None, default None
            Optimizer for training.
        epochs : int or None, default None
            Number of training epochs.
        batch_size : int or None, default None
            Batch size for training.
        chunker : BaseChunker or None, default None
            Explicit chunker instance for chunked mode.
        chunk_size : int or None, default None
            Create fixed-size chunks.
        chunk_count : int or None, default None
            Split into this many equal chunks.
        chunks : list[ArrayLike] or None, default None
            Pre-split reference data for chunked mode.
        chunk_indices : list[list[int]] or None, default None
            Index groupings for chunking reference data.
        threshold : Threshold or None, default None
            Threshold strategy for chunked mode.

        Returns
        -------
        Self
        """
        loss_fn = loss_fn if loss_fn is not None else self.config.loss_fn
        optimizer = optimizer if optimizer is not None else self.config.optimizer
        epochs = epochs if epochs is not None else self.config.epochs
        batch_size = batch_size if batch_size is not None else self.config.batch_size

        self._x_ref = np.asarray(x_ref, dtype=np.float32)
        self._score_batch_size = batch_size

        # Train the model on reference data
        self._scorer.fit(
            x_ref=self._x_ref,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
        )

        # Compute reference reconstruction error distribution
        ref_instance_scores, _ = self._scorer.score(self._x_ref, batch_size=batch_size)
        self._ref_mean = float(np.mean(ref_instance_scores))
        self._ref_std = float(np.std(ref_instance_scores))

        # Handle chunking (prebuilt chunks are converted here)
        if chunks is not None:
            chunks = [np.asarray(c, dtype=np.float32) for c in chunks]

        self._resolve_fit_chunks(
            len(self._x_ref),
            chunker=chunker,
            chunk_size=chunk_size,
            chunk_count=chunk_count,
            chunks=chunks,
            chunk_indices=chunk_indices,
            threshold=threshold,
            default_threshold=ZScoreThreshold(),
        )

        self._fitted = True
        return self

    def _compute_chunk_metric(self, chunk_data: NDArray[np.float32]) -> float:
        """Compute mean reconstruction error for a chunk."""
        iscore, _ = self._scorer.score(chunk_data, batch_size=self._score_batch_size)
        return float(np.mean(iscore))

    @set_metadata
    def predict(
        self,
        x: ArrayLike | None = None,
        chunks: list[ArrayLike] | None = None,
        chunk_indices: list[list[int]] | None = None,
    ) -> DriftOutput:
        """
        Predict whether test data has drifted from reference data.

        Parameters
        ----------
        x : ArrayLike or None
            Test data.
        chunks : list[ArrayLike] or None, default None
            Pre-built test data chunks.
        chunk_indices : list[list[int]] or None, default None
            Index groupings for chunking test data.

        Returns
        -------
        DriftOutput
            Non-chunked mode: ``details`` is a :class:`DriftReconstructionStats` TypedDict.
            Chunked mode: ``details`` is a :class:`polars.DataFrame` with per-chunk results.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict().")

        if self.is_chunked or chunks is not None or chunk_indices is not None:
            if chunks is not None:
                prepared = [np.asarray(c, dtype=np.float32) for c in chunks]
                return self._predict_chunked(chunks_override=prepared)

            x_test = np.asarray(x, dtype=np.float32) if x is not None else None
            return self._predict_chunked(
                x_test=x_test,
                chunk_indices_override=chunk_indices,
            )

        if x is None:
            raise ValueError("x is required for non-chunked prediction.")
        return self._predict_single(x)

    def _predict_single(self, x: ArrayLike) -> DriftOutput:
        """Non-chunked prediction: z-test on mean reconstruction error."""
        x_test = np.asarray(x, dtype=np.float32)

        test_scores, _ = self._scorer.score(x_test, batch_size=self._score_batch_size)
        mean_test = float(np.mean(test_scores))

        # One-sided z-test: is mean test error notably higher than reference?
        # The effective sample size is capped to prevent the test from being
        # overpowered at large N (where trivially small, meaningless differences
        # become "significant").  A cap of 75 requires shifts of at least ~0.2
        # standard deviations (Cohen's d) for detection.
        if self._ref_std > 0:
            n_eff = min(len(test_scores), 75)
            z = (mean_test - self._ref_mean) / (self._ref_std / np.sqrt(n_eff))
            p_val = float(1.0 - norm.cdf(z))
        else:
            p_val = 0.0 if mean_test > self._ref_mean else 1.0

        drifted = p_val < self._p_val

        return DriftOutput(
            drifted=drifted,
            threshold=self._p_val,
            distance=mean_test,
            metric_name="reconstruction_error",
            details=DriftReconstructionStats(
                p_val=p_val,
                mean_ref_error=self._ref_mean,
                mean_test_error=mean_test,
            ),
        )
