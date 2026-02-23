"""
Adapted for Pytorch from.

Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

__all__ = []

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from dataeval.protocols import ArrayLike, DeviceLike, EvidenceLowerBoundLossFn, ReconstructionLossFn
from dataeval.shift._ood._base import BaseOOD, OODScoreOutput
from dataeval.shift._shared._reconstruction import ReconstructionScorer
from dataeval.utils.arrays import to_numpy


class OODReconstruction(BaseOOD):
    """
    Autoencoder (AE) or Variational Autoencoder (VAE) based out-of-distribution detector.

    Supports standard autoencoders and variational autoencoders with optional
    Gaussian Mixture Model (GMM) in the latent space for enhanced detection.
    Model type can be auto-detected from the model structure or explicitly specified.

    Input data must be on the unit interval [0, 1].

    Parameters
    ----------
    model : torch.nn.Module
        An autoencoder or VAE model to use for encoding and reconstruction of images
        for detection of out-of-distribution samples. Model type will be auto-detected
        if not specified.
    device : DeviceLike or None, default None
        The hardware device to use if specified, otherwise uses the DataEval
        default or torch default.
    model_type : {"ae", "vae", "auto"} or None, default "auto"
        Type of model: "ae" for standard autoencoder, "vae" for variational autoencoder,
        or "auto" to auto-detect based on model structure. If None, defaults to "auto".
    use_gmm : bool or None, default None
        Whether to use Gaussian Mixture Model in the latent space for enhanced OOD detection.
        If None, will be auto-detected based on whether model has gmm_density_net attribute.
        When True, the model must output (reconstruction, z, gamma) where z is the latent
        representation and gamma is the mixture assignment probabilities.
    config : OODReconstruction.Config or None, default None
        Optional configuration object with default training parameters. Parameters
        specified in fit() will override these defaults.

    Example
    -------
    Auto-detection (recommended):

    >>> from dataeval.utils.models import AE, VAE, GMMDensityNet
    >>> from dataeval.shift import OODReconstruction
    >>>
    >>> train_data = torch.rand(10, 1, 28, 28)
    >>>
    >>> # Auto-detect AE
    >>> ae = AE(input_shape=(1, 28, 28))
    >>> ood = OODReconstruction(ae)  # Automatically detects as "ae", use_gmm=False
    >>>
    >>> # Auto-detect VAE
    >>> vae = VAE(input_shape=(1, 28, 28))
    >>> ood = OODReconstruction(vae)  # Automatically detects as "vae", use_gmm=False
    >>>
    >>> # Auto-detect GMM
    >>> ae_gmm = AE(input_shape=(1, 28, 28), gmm_density_net=GMMDensityNet(latent_dim=256, n_gmm=3))
    >>> ood = OODReconstruction(ae_gmm)  # Automatically detects as "ae", use_gmm=True

    Using configuration:

    >>> config = OODReconstruction.Config(epochs=10, batch_size=128, threshold_perc=99.0)
    >>> ood = OODReconstruction(vae, config=config)
    >>> ood.fit(train_data)  # Uses config defaults

    Explicit specification:

    >>> ood = OODReconstruction(vae, model_type="vae", use_gmm=False)
    >>> ood.fit(train_data, threshold_perc=95, epochs=20)
    """

    @dataclass
    class Config:
        """
        Configuration for OODReconstruction detector training and threshold computation.

        This dataclass provides default values for common training parameters
        that can be overridden when needed. It's designed to work with the hybrid
        approach where parameters can be set either via config or directly in fit().

        Attributes
        ----------
        loss_fn : Callable or None, default None
            Loss function for training. If None, will use default based on model type:
            - AE: MSELoss()
            - VAE: ELBOLoss()
            - GMM models: MSELoss() (reconstruction only)
        optimizer : torch.optim.Optimizer or None, default None
            Optimizer for training. If None, uses Adam with lr=0.001.
        epochs : int, default 20
            Number of training epochs.
        batch_size : int, default 64
            Batch size for training and scoring.
        threshold_perc : float, default 95.0
            Percentage of reference data considered normal.
        gmm_weight : float, default 0.5
            Weight for GMM component when combining with reconstruction error (α in the formula).
            Final score = (1-α) * recon_score + α * gmm_score, where both are standardized.
            Only used when use_gmm=True. Range [0, 1]: 0=reconstruction only, 1=GMM only.
        gmm_score_mode : {"standardized", "percentile"}, default "standardized"
            Method for combining reconstruction and GMM scores when use_gmm=True:
            - "standardized": Z-score normalization of both components, then weighted sum
            - "percentile": Convert to percentiles, combine as 1 - (P_recon * P_gmm)

        Examples
        --------
        >>> from dataeval.shift import OODReconstruction
        >>> from dataeval.utils.models import VAE
        >>>
        >>> train_data = torch.rand(10, 1, 28, 28)

        Using default configuration:

        >>> ood = OODReconstruction(VAE(input_shape=(1, 28, 28)))
        >>> ood.fit(train_data)  # Uses default config

        Using custom configuration:

        >>> config = OODReconstruction.Config(epochs=10, batch_size=128, threshold_perc=99.0)
        >>> ood = OODReconstruction(VAE(input_shape=(1, 28, 28)), config=config)
        >>> ood.fit(train_data)  # Uses config defaults

        Overriding config in fit():

        >>> config = OODReconstruction.Config(epochs=10, batch_size=128)
        >>> ood = OODReconstruction(VAE(input_shape=(1, 28, 28)), config=config)
        >>> ood.fit(train_data, epochs=100)  # Override config.epochs
        """

        loss_fn: Callable[..., torch.Tensor] | None = None
        optimizer: torch.optim.Optimizer | None = None
        epochs: int = 20
        batch_size: int = 64
        threshold_perc: float = 95.0
        gmm_weight: float = 0.5
        gmm_score_mode: Literal["standardized", "percentile"] = "standardized"

    def __init__(
        self,
        model: torch.nn.Module,
        device: DeviceLike | None = None,
        model_type: Literal["ae", "vae", "auto"] | None = "auto",
        use_gmm: bool | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__()

        # Store config or create default
        self.config: OODReconstruction.Config = config or OODReconstruction.Config()

        # Create scorer (handles auto-detection of model_type and use_gmm)
        self._scorer = ReconstructionScorer(
            model=model,
            device=device,
            model_type=model_type,
            use_gmm=use_gmm,
            gmm_weight=self.config.gmm_weight,
            gmm_score_mode=self.config.gmm_score_mode,
        )

    @property
    def model(self) -> torch.nn.Module:
        """The underlying autoencoder or VAE model."""
        return self._scorer.model

    @property
    def device(self) -> torch.device:
        """The device the model is on."""
        return self._scorer.device

    @property
    def model_type(self) -> str:
        """Model type: ``"ae"`` or ``"vae"``."""
        return self._scorer.model_type

    @property
    def use_gmm(self) -> bool:
        """Whether GMM-based scoring is enabled."""
        return self._scorer.use_gmm

    def _get_data_info(self, x: NDArray) -> tuple[tuple, type]:
        """Validate value range and extract shape/dtype information from data."""
        if not isinstance(x, np.ndarray):
            raise TypeError("Dataset should be of type: `NDArray`.")
        if np.min(x) < 0 or np.max(x) > 1:
            raise ValueError("Data must be on the unit interval [0-1].")
        return x.shape[1:], x.dtype.type

    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float | None = None,
        loss_fn: ReconstructionLossFn | EvidenceLowerBoundLossFn | Callable[..., torch.Tensor] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        """
        Train the model and infer the threshold value.

        Parameters specified here override defaults from the config object.

        Parameters
        ----------
        x_ref : ArrayLike
            Training data.
        threshold_perc : float or None, default None
            Percentage of reference data that is normal (0-100).
            If None, uses config.threshold_perc.
        loss_fn : ReconstructionLossFn | VAELossFn | Callable | None, default None
            Loss function used for training. Can be:
            - A ReconstructionLossFn for AE (e.g., torch.nn.MSELoss())
            - A VAELossFn for VAE (e.g., ELBOLoss(beta=1.0))
            - Any callable with appropriate signature
            If None, uses config.loss_fn, or auto-selects based on model type:
            - For AE/GMM-AE: uses MSELoss()
            - For VAE: uses ELBOLoss()
        optimizer : torch.optim.Optimizer | None, default None
            Optimizer used for training. If None, uses config.optimizer or Adam with lr=0.001.
        epochs : int or None, default None
            Number of training epochs. If None, uses config.epochs (default 20).
        batch_size : int or None, default None
            Batch size used for training. If None, uses config.batch_size (default 64).

        Examples
        --------
        Using config defaults (recommended):

        >>> from dataeval.shift import OODReconstruction
        >>> from dataeval.utils.models import AE, VAE
        >>> from dataeval.utils.losses import ELBOLoss

        >>> input_shape = (1, 28, 28)
        >>> train_data = torch.rand(20, *input_shape)
        >>> config = OODReconstruction.Config(epochs=10, threshold_perc=95)
        >>> ood = OODReconstruction(AE(input_shape), config=config)
        >>> ood.fit(train_data)  # Uses config defaults

        Overriding specific parameters:

        >>> ood = OODReconstruction(VAE(input_shape))
        >>> ood.fit(train_data, epochs=20, batch_size=10)  # Override defaults

        Using custom loss:

        >>> config = OODReconstruction.Config(loss_fn=ELBOLoss(beta=2.0))
        >>> ood = OODReconstruction(VAE(input_shape), config=config)
        >>> ood.fit(train_data)
        """
        # Use config defaults if parameters not specified
        threshold_perc = threshold_perc if threshold_perc is not None else self.config.threshold_perc
        loss_fn = loss_fn if loss_fn is not None else self.config.loss_fn
        optimizer = optimizer if optimizer is not None else self.config.optimizer
        epochs = epochs if epochs is not None else self.config.epochs
        batch_size = batch_size if batch_size is not None else self.config.batch_size

        # Delegate training to scorer
        x_ref_np = to_numpy(x_ref).astype(np.float32)
        self._scorer.fit(
            x_ref=x_ref_np,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
        )

        # Infer the threshold values AFTER GMM parameters are computed
        # This ensures the threshold is based on scores that include the GMM component
        self._ref_score = self.score(x_ref, batch_size)
        self._threshold_perc = threshold_perc

    def _score(self, x: NDArray[np.float32], batch_size: int = int(1e10)) -> OODScoreOutput:
        """Compute OOD scores for the input data."""
        iscore, fscore = self._scorer.score(x, batch_size)
        return OODScoreOutput(iscore, fscore)
