"""
Adapted for Pytorch from

Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

__all__ = []

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import norm

from dataeval.config import get_device
from dataeval.protocols import ArrayLike, DeviceLike, EvidenceLowerBoundLossFn, ReconstructionLossFn
from dataeval.shift._ood._base import OODOutput, OODScoreOutput
from dataeval.types import set_metadata
from dataeval.utils.arrays import as_numpy, to_numpy
from dataeval.utils.losses import ELBOLoss
from dataeval.utils.training import predict, train

TGMMData = TypeVar("TGMMData")


@dataclass
class GaussianMixtureModelParams:
    """
    phi : torch.Tensor
        Mixture component distribution weights.
    mu : torch.Tensor
        Mixture means.
    cov : torch.Tensor
        Mixture covariance.
    L : torch.Tensor
        Cholesky decomposition of `cov`.
    log_det_cov : torch.Tensor
        Log of the determinant of `cov`.
    """

    phi: torch.Tensor
    mu: torch.Tensor
    cov: torch.Tensor
    L: torch.Tensor
    log_det_cov: torch.Tensor


def gmm_params(z: torch.Tensor, gamma: torch.Tensor) -> GaussianMixtureModelParams:
    """
    Compute parameters of Gaussian Mixture Model.

    Parameters
    ----------
    z : torch.Tensor
        Observations.
    gamma : torch.Tensor
        Mixture probabilities to derive mixture distribution weights from.

    Returns
    -------
    GaussianMixtureModelParams(phi, mu, cov, L, log_det_cov)
        The parameters used to calculate energy.
    """

    # compute gmm parameters phi, mu and cov
    N = gamma.shape[0]  # nb of samples in batch
    sum_gamma = torch.sum(gamma, 0)  # K
    phi = sum_gamma / N  # K
    # K x D (D = latent_dim)
    mu = torch.sum(torch.unsqueeze(gamma, -1) * torch.unsqueeze(z, 1), 0) / torch.unsqueeze(sum_gamma, -1)
    z_mu = torch.unsqueeze(z, 1) - torch.unsqueeze(mu, 0)  # N x K x D
    z_mu_outer = torch.unsqueeze(z_mu, -1) * torch.unsqueeze(z_mu, -2)  # N x K x D x D

    # K x D x D
    cov = torch.sum(torch.unsqueeze(torch.unsqueeze(gamma, -1), -1) * z_mu_outer, 0) / torch.unsqueeze(
        torch.unsqueeze(sum_gamma, -1), -1
    )

    # cholesky decomposition of covariance and determinant derivation
    D = cov.shape[1]
    # Use adaptive epsilon that scales with covariance magnitude to ensure numerical stability
    # For high-dimensional spaces or collapsed latent dimensions, a larger epsilon is needed
    # Use the maximum diagonal element as a reference scale, with a minimum of 1e-6
    max_diag = torch.max(torch.diagonal(cov, dim1=-2, dim2=-1))
    adaptive_epsilon = torch.maximum(max_diag * 1e-6, torch.tensor(1e-6))
    L = torch.linalg.cholesky(cov + torch.eye(D) * adaptive_epsilon)  # K x D x D
    log_det_cov = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), 1)  # K

    return GaussianMixtureModelParams(phi, mu, cov, L, log_det_cov)


def gmm_energy(
    z: torch.Tensor,
    params: GaussianMixtureModelParams,
    return_mean: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute sample energy from Gaussian Mixture Model.

    Parameters
    ----------
    params : GaussianMixtureModelParams
        The gaussian mixture model parameters.
    return_mean : bool, default True
        Take mean across all sample energies in a batch.

    Returns
    -------
    sample_energy
        The sample energy of the GMM.
    cov_diag
        The inverse sum of the diagonal components of the covariance matrix.
    """
    D = params.cov.shape[1]
    z_mu = torch.unsqueeze(z, 1) - torch.unsqueeze(params.mu, 0)  # N x K x D
    z_mu_T = torch.permute(z_mu, dims=[1, 2, 0])  # K x D x N
    v = torch.linalg.solve_triangular(params.L, z_mu_T, upper=False)  # K x D x D

    # rewrite sample energy in logsumexp format for numerical stability
    logits = torch.log(torch.unsqueeze(params.phi, -1)) - 0.5 * (
        torch.sum(torch.square(v), 1) + float(D) * np.log(2.0 * np.pi) + torch.unsqueeze(params.log_det_cov, -1)
    )  # K x N
    sample_energy = -torch.logsumexp(logits, 0)  # N

    if return_mean:
        sample_energy = torch.mean(sample_energy)

    # inverse sum of variances
    cov_diag = torch.sum(torch.divide(torch.tensor(1), torch.diagonal(params.cov, dim1=-2, dim2=-1)))

    return sample_energy, cov_diag


class OODReconstruction:
    """
    Autoencoder (AE) or Variational Autoencoder (VAE) based out-of-distribution detector.

    Supports standard autoencoders and variational autoencoders with optional
    Gaussian Mixture Model (GMM) in the latent space for enhanced detection.
    Model type can be auto-detected from the model structure or explicitly specified.

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
        self.model = model
        self.device: torch.device = get_device(device)

        # Store config or create default
        self.config: OODReconstruction.Config = config or OODReconstruction.Config()

        # Auto-detect model type if needed
        if model_type is None or model_type == "auto":
            self.model_type, self.use_gmm = self._auto_detect_model_type(model, use_gmm)
        else:
            if model_type not in ("ae", "vae"):
                raise ValueError(f"model_type must be 'ae', 'vae', or 'auto', got {model_type}")
            self.model_type = model_type
            # Auto-detect GMM if not specified
            self.use_gmm: bool = self._auto_detect_gmm(model) if use_gmm is None else use_gmm

        # Internal state
        self._gmm_params = None
        self._gmm_energy_ref_mean: float | None = None  # Mean GMM energy on reference data
        self._gmm_energy_ref_std: float | None = None  # Std GMM energy on reference data
        self._recon_ref_mean: float | None = None  # Mean reconstruction error on reference data
        self._recon_ref_std: float | None = None  # Std reconstruction error on reference data
        self._ref_score: OODScoreOutput
        self._threshold_perc: float
        self._data_info: tuple[tuple, type] | None = None

    def _auto_detect_gmm(self, model: torch.nn.Module) -> bool:
        """Auto-detect if model uses GMM based on gmm_density_net attribute."""
        return hasattr(model, "gmm_density_net") and model.gmm_density_net is not None

    def _auto_detect_model_type(
        self, model: torch.nn.Module, use_gmm_hint: bool | None
    ) -> tuple[Literal["ae", "vae"], bool]:
        """
        Auto-detect model type and GMM usage from model structure.

        Detection logic:
        1. Check for gmm_density_net attribute to determine GMM usage
        2. Check for reparameterize method to determine if VAE
        3. Otherwise assume standard AE

        Returns
        -------
        tuple[Literal["ae", "vae"], bool]
            Tuple of (model_type, use_gmm)
        """
        # Detect GMM
        has_gmm = self._auto_detect_gmm(model)
        use_gmm = has_gmm if use_gmm_hint is None else use_gmm_hint

        # Detect VAE by checking for reparameterize method (characteristic of VAE)
        is_vae = hasattr(model, "reparameterize") and callable(getattr(model, "reparameterize", None))

        model_type: Literal["ae", "vae"] = "vae" if is_vae else "ae"

        return model_type, use_gmm

    def _get_data_info(self, X: NDArray) -> tuple[tuple, type]:
        """Validate and extract shape and dtype information from data."""
        if not isinstance(X, np.ndarray):
            raise TypeError("Dataset should be of type: `NDArray`.")
        if np.min(X) < 0 or np.max(X) > 1:
            raise ValueError("Data must be on the unit interval [0-1].")
        return X.shape[1:], X.dtype.type

    def _validate(self, X: NDArray) -> None:
        """Validate that input data matches expected shape and dtype."""
        check_data_info = self._get_data_info(X)
        if self._data_info is not None and check_data_info != self._data_info:
            raise RuntimeError(
                f"Expect data of type: {self._data_info[1]} and shape: {self._data_info[0]}. "
                f"Provided data is type: {check_data_info[1]} and shape: {check_data_info[0]}."
            )

    def _validate_state(self, X: NDArray) -> None:
        """Validate that detector has been fitted and data is valid."""
        if not hasattr(self, "_ref_score") or not hasattr(self, "_threshold_perc"):
            raise RuntimeError("Detector needs to be `fit` before calling predict or score.")
        self._validate(X)

    def _validate_gmm_output(self, x_ref: ArrayLike) -> None:
        """Validate that model output format is correct for GMM usage."""
        # Quick check with a small batch to validate model output format
        # Use at least 10 samples for GMM validation to ensure covariance matrix is computable
        x_ref_np = to_numpy(x_ref).astype(np.float32)
        sample_size = min(len(x_ref_np), max(10, len(x_ref_np) // 10))
        sample_output = predict(x_ref_np[:sample_size], self.model, batch_size=sample_size)

        # Validate model output format for GMM
        if not isinstance(sample_output, tuple):
            raise ValueError(
                "When use_gmm=True, model must return a tuple of (reconstruction, z, gamma), "
                f"but got {type(sample_output).__name__}"
            )
        if len(sample_output) < 3:
            raise ValueError(
                "When use_gmm=True, model must return tuple of at least 3 elements: (reconstruction, z, gamma), "
                f"but got {len(sample_output)} elements"
            )

        # Validate the shapes and types of z and gamma
        z_test = sample_output[1]
        gamma_test = sample_output[2]

        # Validate z (latent representation)
        if not isinstance(z_test, torch.Tensor):
            raise ValueError(
                f"When use_gmm=True, model's second output (z) must be a torch.Tensor, got {type(z_test).__name__}"
            )
        if z_test.ndim != 2:
            raise ValueError(
                f"When use_gmm=True, model's second output (z) must be 2D with shape (batch_size, latent_dim), "
                f"got shape {z_test.shape}"
            )

        # Validate gamma (mixture probabilities)
        if not isinstance(gamma_test, torch.Tensor):
            raise ValueError(
                f"When use_gmm=True, model's third output (gamma) must be a torch.Tensor, "
                f"got {type(gamma_test).__name__}"
            )
        if gamma_test.ndim != 2:
            raise ValueError(
                f"When use_gmm=True, model's third output (gamma) must be 2D with shape (batch_size, n_gmm), "
                f"got shape {gamma_test.shape}"
            )

        # Check gamma sums to approximately 1 (it's a probability distribution)
        gamma_sums = gamma_test.sum(dim=-1)
        if not torch.allclose(gamma_sums, torch.ones_like(gamma_sums), atol=1e-5):
            min_sum = gamma_sums.min()
            max_sum = gamma_sums.max()
            raise ValueError(
                "When use_gmm=True, model's third output (gamma) must be a probability distribution "
                f"(sum to 1 along last dimension). Got sums with min={min_sum:.6f}, max={max_sum:.6f}. "
                "Consider using nn.Softmax(dim=-1) as the final layer of your GMM density network."
            )

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

        # If using GMM, validate model output format BEFORE training
        if self.use_gmm:
            self._validate_gmm_output(x_ref)

        # Set default loss function if not provided
        if loss_fn is None:
            if self.use_gmm:
                # GMM models output (recon, z, gamma), so use reconstruction loss only
                base_mse = torch.nn.MSELoss()

                def gmm_reconstruction_loss(
                    x: torch.Tensor, x_recon: torch.Tensor, z: torch.Tensor, gamma: torch.Tensor
                ) -> Any:
                    """Loss for GMM models - uses only reconstruction error."""
                    return base_mse(x, x_recon)

                loss_fn = gmm_reconstruction_loss
            else:
                # Standard AE/VAE: use appropriate loss
                loss_fn = torch.nn.MSELoss() if self.model_type == "ae" else ELBOLoss()

        # Set default optimizer if not provided
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Train the model
        train(
            model=self.model,
            x_train=to_numpy(x_ref),
            y_train=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            preprocess_fn=None,
            epochs=epochs,
            batch_size=batch_size,
            device=self.device,
        )

        # If using GMM, compute GMM parameters after training but BEFORE computing threshold
        if self.use_gmm:
            x_ref_np = to_numpy(x_ref).astype(np.float32)
            model_output = predict(x_ref_np, self.model, batch_size=batch_size)

            # Expected: (reconstruction, z, gamma)
            z = model_output[1].detach()
            gamma = model_output[2].detach()
            self._gmm_params = gmm_params(z, gamma)

            # Compute GMM energy statistics on reference data
            ref_gmm_energy, _ = gmm_energy(z, self._gmm_params, return_mean=False)
            ref_gmm_energy_np = ref_gmm_energy.detach().cpu().numpy()
            self._gmm_energy_ref_mean = float(ref_gmm_energy_np.mean())
            self._gmm_energy_ref_std = float(ref_gmm_energy_np.std())

            # Compute reconstruction error statistics on reference data
            X_recon = model_output[0].detach().cpu().numpy()
            fscore = np.power(x_ref_np - X_recon, 2)
            fscore_flat = fscore.reshape(fscore.shape[0], -1)
            n_score_features = int(np.ceil(fscore_flat.shape[1]))
            sorted_fscore = np.sort(fscore_flat, axis=1)
            sorted_fscore_perc = sorted_fscore[:, -n_score_features:]
            recon_scores = np.mean(sorted_fscore_perc, axis=1)
            self._recon_ref_mean = float(recon_scores.mean())
            self._recon_ref_std = float(recon_scores.std())

        # Infer the threshold values AFTER GMM parameters are computed
        # This ensures the threshold is based on scores that include the GMM component
        self._ref_score = self.score(x_ref, batch_size)
        self._threshold_perc = threshold_perc

    def _combine_gmm_standardized(self, recon_scores: NDArray, gmm_energy: NDArray, gmm_weight: float) -> NDArray:
        """
        Combine reconstruction and GMM scores using standardized (z-score) fusion.

        Parameters
        ----------
        recon_scores : NDArray
            Reconstruction error scores.
        gmm_energy : NDArray
            GMM energy scores.
        gmm_weight : float
            Weight for GMM component (α in the formula).

        Returns
        -------
        NDArray
            Combined scores using weighted sum: (1-α) * recon_z + α * gmm_z
        """
        # Standardize reconstruction scores
        recon_mean = self._recon_ref_mean if self._recon_ref_mean is not None else recon_scores.mean()
        recon_std = self._recon_ref_std if self._recon_ref_std is not None else (recon_scores.std() + 1e-10)
        recon_standardized = (recon_scores - recon_mean) / (recon_std + 1e-10)

        # Standardize GMM energy scores
        gmm_mean = self._gmm_energy_ref_mean if self._gmm_energy_ref_mean is not None else gmm_energy.mean()
        gmm_std = self._gmm_energy_ref_std if self._gmm_energy_ref_std is not None else (gmm_energy.std() + 1e-10)
        gmm_standardized = (gmm_energy - gmm_mean) / (gmm_std + 1e-10)

        # Weighted combination: (1-α) * recon + α * gmm
        return (1 - gmm_weight) * recon_standardized + gmm_weight * gmm_standardized

    def _combine_gmm_percentile(self, recon_scores: NDArray, gmm_energy: NDArray) -> NDArray:
        """
        Combine reconstruction and GMM scores using percentile-based fusion.

        Converts both scores to percentiles using z-score to CDF transformation,
        then combines as: 1 - (P_in-dist_recon * P_in-dist_gmm)

        Parameters
        ----------
        recon_scores : NDArray
            Reconstruction error scores.
        gmm_energy : NDArray
            GMM energy scores.

        Returns
        -------
        NDArray
            Combined scores as probability of being OOD.
        """
        # Reconstruction percentile
        recon_mean = self._recon_ref_mean if self._recon_ref_mean is not None else recon_scores.mean()
        recon_std = self._recon_ref_std if self._recon_ref_std is not None else (recon_scores.std() + 1e-10)
        recon_z = (recon_scores - recon_mean) / (recon_std + 1e-10)
        recon_percentile = norm.cdf(recon_z)

        # GMM energy percentile
        gmm_mean = self._gmm_energy_ref_mean if self._gmm_energy_ref_mean is not None else gmm_energy.mean()
        gmm_std = self._gmm_energy_ref_std if self._gmm_energy_ref_std is not None else (gmm_energy.std() + 1e-10)
        gmm_z = (gmm_energy - gmm_mean) / (gmm_std + 1e-10)
        gmm_percentile = norm.cdf(gmm_z)

        # Combine as: 1 - (P_in-dist_recon * P_in-dist_gmm)
        # Where P_in-dist = 1 - percentile (probability of being in-distribution)
        # High percentile = high OOD score
        # Final score is probability of being OOD
        return 1.0 - ((1.0 - recon_percentile) * (1.0 - gmm_percentile))

    def _score(self, X: NDArray[np.float32], batch_size: int = int(1e10)) -> OODScoreOutput:
        """
        Compute OOD scores for the input data.

        For AE models, uses reconstruction error.
        For VAE models, uses reconstruction error.
        For models with GMM, combines reconstruction error with GMM energy in latent space.
        """
        # Get model outputs
        model_output = predict(X, self.model, batch_size=batch_size)

        # Extract reconstruction based on model type
        if self.model_type == "ae" and not self.use_gmm:
            model_output = model_output[0] if isinstance(model_output, tuple) else model_output
            X_recon = model_output.detach().cpu().numpy()
        else:  # vae or using gmm
            # Extract reconstruction (first element of tuple)
            if isinstance(model_output, tuple):
                X_recon = model_output[0].detach().cpu().numpy()
            else:
                X_recon = model_output.detach().cpu().numpy()

        # Compute reconstruction-based feature and instance level scores
        fscore = np.power(X - X_recon, 2)
        fscore_flat = fscore.reshape(fscore.shape[0], -1).copy()
        n_score_features = int(np.ceil(fscore_flat.shape[1]))
        sorted_fscore = np.sort(fscore_flat, axis=1)
        sorted_fscore_perc = sorted_fscore[:, -n_score_features:]
        iscore = np.mean(sorted_fscore_perc, axis=1)

        # If using GMM, combine reconstruction error with GMM energy using sensor fusion
        if self.use_gmm and self._gmm_params is not None:
            # Extract latent representation (second element of tuple)
            if isinstance(model_output, tuple) and len(model_output) >= 2:
                z = model_output[1].detach()
                gmm_energy_score, _ = gmm_energy(z, self._gmm_params, return_mean=False)
                gmm_energy_np = gmm_energy_score.detach().cpu().numpy()

                # Get configuration parameters
                score_mode = self.config.gmm_score_mode

                if score_mode == "standardized":
                    iscore = self._combine_gmm_standardized(iscore, gmm_energy_np, self.config.gmm_weight)
                elif score_mode == "percentile":
                    iscore = self._combine_gmm_percentile(iscore, gmm_energy_np)
                else:
                    raise ValueError(f"Unknown gmm_score_mode: {score_mode}")

            else:
                raise ValueError(
                    "When use_gmm=True, model must return tuple with latent representation as second element"
                )

        return OODScoreOutput(iscore, fscore)

    @set_metadata
    def score(self, X: ArrayLike, batch_size: int = int(1e10)) -> OODScoreOutput:
        """
        Compute the :term:`out of distribution<Out-of-distribution (OOD)>` scores for a given dataset.

        Parameters
        ----------
        X : ArrayLike
            Input data to score.
        batch_size : int, default 1e10
            Number of instances to process in each batch.
            Use a smaller batch size if your dataset is large or if you encounter memory issues.

        Raises
        ------
        ValueError
            X input data must be unit interval [0-1].

        Returns
        -------
        OODScoreOutput
            An object containing the instance-level and feature-level OOD scores.
        """
        X_np = as_numpy(X).astype(np.float32)
        self._validate(X_np)
        return self._score(X_np, batch_size)

    def _threshold_score(self, ood_type: Literal["feature", "instance"] = "instance") -> np.floating:
        """Get the threshold score for a given OOD type."""
        return np.percentile(self._ref_score.get(ood_type), self._threshold_perc)

    @set_metadata
    def predict(
        self,
        X: ArrayLike,
        batch_size: int = int(1e10),
        ood_type: Literal["feature", "instance"] = "instance",
    ) -> OODOutput:
        """
        Predict whether instances are :term:`out of distribution<Out-of-distribution (OOD)>` or not.

        Parameters
        ----------
        X : ArrayLike
            Input data for out-of-distribution prediction.
        batch_size : int, default 1e10
            Number of instances to process in each batch.
        ood_type : "feature" | "instance", default "instance"
            Predict out-of-distribution at the 'feature' or 'instance' level.

        Raises
        ------
        ValueError
            X input data must be unit interval [0-1].

        Returns
        -------
        OODOutput
            Dictionary containing the outlier predictions for the selected level,
            and the OOD scores for the data including both 'instance' and 'feature' (if present) level scores.
        """
        X_np = to_numpy(X).astype(np.float32)
        self._validate_state(X_np)

        # Compute outlier scores
        score = self.score(X_np, batch_size=batch_size)
        ood_pred = score.get(ood_type) > self._threshold_score(ood_type)
        return OODOutput(is_ood=ood_pred, **score.data())
