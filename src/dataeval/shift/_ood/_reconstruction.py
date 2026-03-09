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
from typing_extensions import Self

from dataeval.protocols import ArrayLike, DeviceLike
from dataeval.shift._ood._base import BaseOOD, OODScoreOutput
from dataeval.shift._shared._reconstruction import ReconstructionScorer
from dataeval.utils._internal import to_numpy


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
    threshold_perc : float or None, default None
        Percentage of reference data considered normal (0-100).
        If None, uses config.threshold_perc (default 95.0).
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
    OODReconstruction(loss_fn=None, optimizer=None, epochs=10, batch_size=128, threshold_perc=99.0, gmm_weight=0.5, gmm_score_mode='standardized', fitted=False)

    Explicit specification:

    >>> config = OODReconstruction.Config(epochs=20)
    >>> ood = OODReconstruction(vae, model_type="vae", use_gmm=False, threshold_perc=95, config=config)
    >>> ood.fit(train_data)
    OODReconstruction(loss_fn=None, optimizer=None, epochs=20, batch_size=64, threshold_perc=95, gmm_weight=0.5, gmm_score_mode='standardized', fitted=False)
    """  # noqa: E501

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
        OODReconstruction(loss_fn=None, optimizer=None, epochs=20, batch_size=64, threshold_perc=95.0, gmm_weight=0.5, gmm_score_mode='standardized', fitted=False)

        Using custom configuration:

        >>> config = OODReconstruction.Config(epochs=10, batch_size=128, threshold_perc=99.0)
        >>> ood = OODReconstruction(VAE(input_shape=(1, 28, 28)), config=config)
        >>> ood.fit(train_data)  # Uses config defaults
        OODReconstruction(loss_fn=None, optimizer=None, epochs=10, batch_size=128, threshold_perc=99.0, gmm_weight=0.5, gmm_score_mode='standardized', fitted=False)

        Using different config values:

        >>> config = OODReconstruction.Config(epochs=100, batch_size=128)
        >>> ood = OODReconstruction(VAE(input_shape=(1, 28, 28)), config=config)
        >>> ood.fit(train_data)
        OODReconstruction(loss_fn=None, optimizer=None, epochs=100, batch_size=128, threshold_perc=95.0, gmm_weight=0.5, gmm_score_mode='standardized', fitted=False)
        """  # noqa: E501

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
        threshold_perc: float | None = None,
        config: Config | None = None,
    ) -> None:
        # Store config or create default
        base_config = config or OODReconstruction.Config()

        threshold_perc = threshold_perc if threshold_perc is not None else base_config.threshold_perc
        super().__init__(threshold_perc)

        self.config: OODReconstruction.Config = OODReconstruction.Config(
            loss_fn=base_config.loss_fn,
            optimizer=base_config.optimizer,
            epochs=base_config.epochs,
            batch_size=base_config.batch_size,
            threshold_perc=threshold_perc,
            gmm_weight=base_config.gmm_weight,
            gmm_score_mode=base_config.gmm_score_mode,
        )

        # Create scorer (handles auto-detection of model_type and use_gmm)
        self._scorer = ReconstructionScorer(
            model=model,
            device=device,
            model_type=model_type,
            use_gmm=use_gmm,
            gmm_weight=base_config.gmm_weight,
            gmm_score_mode=base_config.gmm_score_mode,
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

    def fit(self, reference_data: ArrayLike) -> Self:
        """
        Train the model and infer the threshold value.

        Training parameters (``loss_fn``, ``optimizer``, ``epochs``,
        ``batch_size``) are taken from :class:`Config`.

        Parameters
        ----------
        reference_data : ArrayLike
            Training data.

        Returns
        -------
        Self
            The fitted detector (for method chaining).

        Examples
        --------
        >>> from dataeval.shift import OODReconstruction
        >>> from dataeval.utils.models import AE, VAE

        >>> input_shape = (1, 28, 28)
        >>> train_data = torch.rand(20, *input_shape)
        >>> config = OODReconstruction.Config(epochs=10, threshold_perc=95)
        >>> ood = OODReconstruction(AE(input_shape), config=config)
        >>> ood.fit(train_data)
        OODReconstruction(loss_fn=None, optimizer=None, epochs=10, batch_size=64, threshold_perc=95, gmm_weight=0.5, gmm_score_mode='standardized', fitted=False)
        """  # noqa: E501
        loss_fn = self.config.loss_fn
        optimizer = self.config.optimizer
        epochs = self.config.epochs
        batch_size = self.config.batch_size

        # Delegate training to scorer
        x_ref_np = to_numpy(reference_data).astype(np.float32)
        self._scorer.fit(
            reference_data=x_ref_np,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
        )

        # Infer the threshold values AFTER GMM parameters are computed
        # This ensures the threshold is based on scores that include the GMM component
        self._ref_score = self.score(reference_data, batch_size)
        return self

    def _score(self, x: NDArray[np.float32], batch_size: int | None = None) -> OODScoreOutput:
        """Compute OOD scores for the input data."""
        from dataeval.config import get_batch_size

        iscore, fscore = self._scorer.score(x, get_batch_size(batch_size))
        return OODScoreOutput(iscore, fscore)
