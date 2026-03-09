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

from dataeval.exceptions import NotFittedError
from dataeval.protocols import DeviceLike, Threshold
from dataeval.shift._drift._base import BaseDrift, ChunkableMixin, DriftOutput
from dataeval.shift._shared._reconstruction import ReconstructionScorer
from dataeval.types import set_metadata
from dataeval.utils.thresholds import ZScoreThreshold


class _DriftReconstructionStats(TypedDict):
    p_val: float
    mean_ref_error: float
    mean_test_error: float


class DriftReconstruction(ChunkableMixin, BaseDrift[_DriftReconstructionStats]):
    """Reconstruction-based drift detector using autoencoders.

    Detects drift by comparing reconstruction errors: if the model (trained
    on reference data) produces higher reconstruction errors on test data,
    the test distribution has likely shifted.

    Uses a fit/predict lifecycle: construct with model and hyperparameters,
    call :meth:`fit` with reference data (trains the model), then call
    :meth:`predict` with test data.
    Use :meth:`chunked` to create a chunked wrapper for time-series monitoring.

    Supports two modes:

    - **Non-chunked** (default): Computes mean reconstruction error for the
      test set and uses a z-test against the reference baseline.
    - **Chunked** (via :meth:`chunked`): Splits data into chunks, computes
      mean reconstruction error per chunk, and uses threshold bounds to
      flag drift.

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

    See Also
    --------
    :class:`DriftReconstruction.Stats` : Per-prediction statistics returned in :attr:`DriftOutput.details`.

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

    class Stats(_DriftReconstructionStats):
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

        base_config = config or DriftReconstruction.Config()

        self._p_val = p_val if p_val is not None else base_config.p_val
        self.config: DriftReconstruction.Config = DriftReconstruction.Config(
            p_val=self._p_val,
            loss_fn=base_config.loss_fn,
            optimizer=base_config.optimizer,
            epochs=base_config.epochs,
            batch_size=base_config.batch_size,
            gmm_weight=base_config.gmm_weight,
            gmm_score_mode=base_config.gmm_score_mode,
        )

        self._scorer = ReconstructionScorer(
            model=model,
            device=device,
            model_type=model_type,
            use_gmm=use_gmm,
            gmm_weight=base_config.gmm_weight,
            gmm_score_mode=base_config.gmm_score_mode,
        )

        self._metric_name = "reconstruction_error"
        self._ref_mean: float = 0.0
        self._ref_std: float = 1.0
        self._score_batch_size: int = int(1e10)

    def fit(self, reference_data: ArrayLike) -> Self:
        """Fit the reconstruction drift detector.

        Trains the autoencoder on reference data using parameters from
        :class:`Config` (``loss_fn``, ``optimizer``, ``epochs``, ``batch_size``).

        Parameters
        ----------
        reference_data : ArrayLike
            Reference data.

        Returns
        -------
        Self
        """
        loss_fn = self.config.loss_fn
        optimizer = self.config.optimizer
        epochs = self.config.epochs
        batch_size = self.config.batch_size

        self._reference_data = np.asarray(reference_data, dtype=np.float32)
        self._score_batch_size = batch_size

        # Train the model on reference data
        self._scorer.fit(
            reference_data=self._reference_data,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
        )

        # Compute reference reconstruction error distribution
        ref_instance_scores, _ = self._scorer.score(self._reference_data, batch_size=batch_size)
        self._ref_mean = float(np.mean(ref_instance_scores))
        self._ref_std = float(np.std(ref_instance_scores))

        self._fitted = True
        return self

    def _compute_chunk_metric(self, chunk_data: NDArray[np.float32]) -> float:
        """Compute mean reconstruction error for a chunk."""
        iscore, _ = self._scorer.score(chunk_data, batch_size=self._score_batch_size)
        return float(np.mean(iscore))

    def _default_chunk_threshold(self) -> Threshold:
        return ZScoreThreshold()

    @set_metadata
    def predict(self, data: ArrayLike) -> DriftOutput["DriftReconstruction.Stats"]:
        """
        Predict whether test data has drifted from reference data.

        Parameters
        ----------
        data : ArrayLike
            Test data.

        Returns
        -------
        DriftOutput[DriftReconstruction.Stats]
            Drift prediction with reconstruction error statistics.
        """
        if not self._fitted:
            raise NotFittedError("Must call fit() before predict().")

        x_test = np.asarray(data, dtype=np.float32)

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
            details=_DriftReconstructionStats(
                p_val=p_val,
                mean_ref_error=self._ref_mean,
                mean_test_error=mean_test,
            ),
        )
