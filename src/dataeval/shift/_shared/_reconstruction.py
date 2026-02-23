"""
Core reconstruction math: model training, scoring, and GMM fusion.

Shared by OODReconstruction (per-instance OOD scoring) and DriftReconstruction (drift detection).

Adapted for Pytorch from Alibi-Detect 0.11.4.
Source code derived from https://github.com/SeldonIO/alibi-detect/tree/v0.11.4
Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

__all__ = []

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import norm

from dataeval.config import get_device
from dataeval.protocols import DeviceLike, EvidenceLowerBoundLossFn, ReconstructionLossFn
from dataeval.utils.losses import ELBOLoss
from dataeval.utils.training import predict, train


@dataclass
class GaussianMixtureModelParams:
    """Parameters for a Gaussian Mixture Model.

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
    n_samples = gamma.shape[0]  # nb of samples in batch
    sum_gamma = torch.sum(gamma, 0)  # K
    phi = sum_gamma / n_samples  # K
    # K x D (D = latent_dim)
    mu = torch.sum(torch.unsqueeze(gamma, -1) * torch.unsqueeze(z, 1), 0) / torch.unsqueeze(sum_gamma, -1)
    z_mu = torch.unsqueeze(z, 1) - torch.unsqueeze(mu, 0)  # N x K x D
    z_mu_outer = torch.unsqueeze(z_mu, -1) * torch.unsqueeze(z_mu, -2)  # N x K x D x D

    # K x D x D
    cov = torch.sum(torch.unsqueeze(torch.unsqueeze(gamma, -1), -1) * z_mu_outer, 0) / torch.unsqueeze(
        torch.unsqueeze(sum_gamma, -1),
        -1,
    )

    # cholesky decomposition of covariance and determinant derivation
    d = cov.shape[1]
    # Use adaptive epsilon that scales with covariance magnitude to ensure numerical stability
    max_diag = torch.max(torch.diagonal(cov, dim1=-2, dim2=-1))
    adaptive_epsilon = torch.maximum(max_diag * 1e-6, torch.tensor(1e-6))
    chol = torch.linalg.cholesky(cov + torch.eye(d) * adaptive_epsilon)  # K x D x D
    log_det_cov = 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)), 1)  # K

    return GaussianMixtureModelParams(phi, mu, cov, chol, log_det_cov)


def gmm_energy(
    z: torch.Tensor,
    params: GaussianMixtureModelParams,
    return_mean: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute sample energy from Gaussian Mixture Model.

    Parameters
    ----------
    z : torch.Tensor
        Observations.
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
    d = params.cov.shape[1]
    z_mu = torch.unsqueeze(z, 1) - torch.unsqueeze(params.mu, 0)  # N x K x D
    z_mu_t = torch.permute(z_mu, dims=[1, 2, 0])  # K x D x N
    v = torch.linalg.solve_triangular(params.L, z_mu_t, upper=False)  # K x D x D

    # rewrite sample energy in logsumexp format for numerical stability
    logits = torch.log(torch.unsqueeze(params.phi, -1)) - 0.5 * (
        torch.sum(torch.square(v), 1) + float(d) * np.log(2.0 * np.pi) + torch.unsqueeze(params.log_det_cov, -1)
    )  # K x N
    sample_energy = -torch.logsumexp(logits, 0)  # N

    if return_mean:
        sample_energy = torch.mean(sample_energy)

    # inverse sum of variances
    cov_diag = torch.sum(torch.divide(torch.tensor(1), torch.diagonal(params.cov, dim1=-2, dim2=-1)))

    return sample_energy, cov_diag


class ReconstructionScorer:
    """Pure reconstruction math: model training, scoring, GMM fusion.

    Parameters
    ----------
    model : torch.nn.Module
        Autoencoder or VAE model.
    device : DeviceLike or None
        Hardware device.
    model_type : {"ae", "vae", "auto"} or None
        Model type. ``"auto"`` auto-detects.
    use_gmm : bool or None
        Whether to use GMM in latent space. ``None`` auto-detects.
    gmm_weight : float
        Weight for GMM component when combining scores.
    gmm_score_mode : {"standardized", "percentile"}
        Method for combining reconstruction and GMM scores.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: DeviceLike | None = None,
        model_type: Literal["ae", "vae", "auto"] | None = "auto",
        use_gmm: bool | None = None,
        gmm_weight: float = 0.5,
        gmm_score_mode: Literal["standardized", "percentile"] = "standardized",
    ) -> None:
        self.model = model
        self.device: torch.device = get_device(device)
        self.gmm_weight = gmm_weight
        self.gmm_score_mode = gmm_score_mode

        # Auto-detect model type if needed
        if model_type is None or model_type == "auto":
            self.model_type, self.use_gmm = self._auto_detect_model_type(model, use_gmm)
        else:
            if model_type not in ("ae", "vae"):
                raise ValueError(f"model_type must be 'ae', 'vae', or 'auto', got {model_type}")
            self.model_type = model_type
            self.use_gmm: bool = self._auto_detect_gmm(model) if use_gmm is None else use_gmm

        # Internal GMM state
        self._gmm_params: GaussianMixtureModelParams | None = None
        self._gmm_energy_ref_mean: float | None = None
        self._gmm_energy_ref_std: float | None = None
        self._recon_ref_mean: float | None = None
        self._recon_ref_std: float | None = None

    @staticmethod
    def _auto_detect_gmm(model: torch.nn.Module) -> bool:
        """Auto-detect if model uses GMM based on gmm_density_net attribute."""
        return hasattr(model, "gmm_density_net") and model.gmm_density_net is not None

    @staticmethod
    def _auto_detect_model_type(
        model: torch.nn.Module,
        use_gmm_hint: bool | None,
    ) -> tuple[Literal["ae", "vae"], bool]:
        """Auto-detect model type and GMM usage from model structure."""
        has_gmm = ReconstructionScorer._auto_detect_gmm(model)
        use_gmm = has_gmm if use_gmm_hint is None else use_gmm_hint
        is_vae = hasattr(model, "reparameterize") and callable(getattr(model, "reparameterize", None))
        model_type: Literal["ae", "vae"] = "vae" if is_vae else "ae"
        return model_type, use_gmm

    def validate_gmm_output(self, x_ref: NDArray[np.float32]) -> None:
        """Validate that model output format is correct for GMM usage.

        Parameters
        ----------
        x_ref : NDArray[np.float32]
            Reference data to validate model output against.
        batch_size : int
            Batch size for model prediction.
        """
        sample_size = min(len(x_ref), max(10, len(x_ref) // 10))
        sample_output = predict(x_ref[:sample_size], self.model, batch_size=sample_size)

        if not isinstance(sample_output, tuple):
            raise ValueError(
                "When use_gmm=True, model must return a tuple of (reconstruction, z, gamma), "
                f"but got {type(sample_output).__name__}",
            )
        if len(sample_output) < 3:
            raise ValueError(
                "When use_gmm=True, model must return tuple of at least 3 elements: (reconstruction, z, gamma), "
                f"but got {len(sample_output)} elements",
            )

        z_test = sample_output[1]
        gamma_test = sample_output[2]

        if not isinstance(z_test, torch.Tensor):
            raise ValueError(
                f"When use_gmm=True, model's second output (z) must be a torch.Tensor, got {type(z_test).__name__}",
            )
        if z_test.ndim != 2:
            raise ValueError(
                f"When use_gmm=True, model's second output (z) must be 2D with shape (batch_size, latent_dim), "
                f"got shape {z_test.shape}",
            )

        if not isinstance(gamma_test, torch.Tensor):
            raise ValueError(
                f"When use_gmm=True, model's third output (gamma) must be a torch.Tensor, "
                f"got {type(gamma_test).__name__}",
            )
        if gamma_test.ndim != 2:
            raise ValueError(
                f"When use_gmm=True, model's third output (gamma) must be 2D with shape (batch_size, n_gmm), "
                f"got shape {gamma_test.shape}",
            )

        gamma_sums = gamma_test.sum(dim=-1)
        if not torch.allclose(gamma_sums, torch.ones_like(gamma_sums), atol=1e-5):
            min_sum = gamma_sums.min()
            max_sum = gamma_sums.max()
            raise ValueError(
                "When use_gmm=True, model's third output (gamma) must be a probability distribution "
                f"(sum to 1 along last dimension). Got sums with min={min_sum:.6f}, max={max_sum:.6f}. "
                "Consider using nn.Softmax(dim=-1) as the final layer of your GMM density network.",
            )

    def fit(
        self,
        x_ref: NDArray[np.float32],
        loss_fn: ReconstructionLossFn | EvidenceLowerBoundLossFn | Callable[..., torch.Tensor] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        epochs: int = 20,
        batch_size: int = 64,
    ) -> None:
        """Train model, compute GMM params + reference stats if applicable.

        Parameters
        ----------
        x_ref : NDArray[np.float32]
            Training data as numpy array.
        loss_fn : Callable or None
            Loss function for training. If None, auto-selects based on model type.
        optimizer : torch.optim.Optimizer or None
            Optimizer for training. If None, uses Adam with lr=0.001.
        epochs : int
            Number of training epochs.
        batch_size : int
            Batch size for training.
        """
        # Validate GMM output format before training
        if self.use_gmm:
            self.validate_gmm_output(x_ref)

        # Set default loss function if not provided
        if loss_fn is None:
            if self.use_gmm:
                base_mse = torch.nn.MSELoss()

                def gmm_reconstruction_loss(
                    x: torch.Tensor,
                    x_recon: torch.Tensor,
                    _z: torch.Tensor,
                    _gamma: torch.Tensor,
                ) -> Any:
                    return base_mse(x, x_recon)

                loss_fn = gmm_reconstruction_loss
            else:
                loss_fn = torch.nn.MSELoss() if self.model_type == "ae" else ELBOLoss()

        # Set default optimizer if not provided
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Train the model
        train(
            model=self.model,
            x_train=x_ref,
            y_train=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            preprocess_fn=None,
            epochs=epochs,
            batch_size=batch_size,
            device=self.device,
        )

        # Compute GMM parameters after training
        if self.use_gmm:
            model_output = predict(x_ref, self.model, batch_size=batch_size)

            z = model_output[1].detach()
            gamma = model_output[2].detach()
            self._gmm_params = gmm_params(z, gamma)

            # GMM energy statistics on reference data
            ref_gmm_energy, _ = gmm_energy(z, self._gmm_params, return_mean=False)
            ref_gmm_energy_np = ref_gmm_energy.detach().cpu().numpy()
            self._gmm_energy_ref_mean = float(ref_gmm_energy_np.mean())
            self._gmm_energy_ref_std = float(ref_gmm_energy_np.std())

            # Reconstruction error statistics on reference data
            x_recon = model_output[0].detach().cpu().numpy()
            fscore = np.power(x_ref - x_recon, 2)
            fscore_flat = fscore.reshape(fscore.shape[0], -1)
            n_score_features = int(np.ceil(fscore_flat.shape[1]))
            sorted_fscore = np.sort(fscore_flat, axis=1)
            sorted_fscore_perc = sorted_fscore[:, -n_score_features:]
            recon_scores = np.mean(sorted_fscore_perc, axis=1)
            self._recon_ref_mean = float(recon_scores.mean())
            self._recon_ref_std = float(recon_scores.std())

    def score(
        self,
        x: NDArray[np.float32],
        batch_size: int = int(1e10),
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Compute reconstruction-based scores.

        Parameters
        ----------
        x : NDArray[np.float32]
            Input data.
        batch_size : int
            Batch size for model prediction.

        Returns
        -------
        tuple[NDArray[np.float32], NDArray[np.float32]]
            (instance_scores, feature_scores) as plain NDArrays.
        """
        model_output = predict(x, self.model, batch_size=batch_size)

        # Extract reconstruction based on model type
        if self.model_type == "ae" and not self.use_gmm:
            model_output = model_output[0] if isinstance(model_output, tuple) else model_output
            x_recon = model_output.detach().cpu().numpy()
        else:
            if isinstance(model_output, tuple):
                x_recon = model_output[0].detach().cpu().numpy()
            else:
                x_recon = model_output.detach().cpu().numpy()

        # Compute reconstruction-based feature and instance level scores
        fscore = np.power(x - x_recon, 2)
        fscore_flat = fscore.reshape(fscore.shape[0], -1).copy()
        n_score_features = int(np.ceil(fscore_flat.shape[1]))
        sorted_fscore = np.sort(fscore_flat, axis=1)
        sorted_fscore_perc = sorted_fscore[:, -n_score_features:]
        iscore = np.mean(sorted_fscore_perc, axis=1)

        # If using GMM, combine reconstruction error with GMM energy
        if self.use_gmm and self._gmm_params is not None:
            if isinstance(model_output, tuple) and len(model_output) >= 2:
                z = model_output[1].detach()
                gmm_energy_score, _ = gmm_energy(z, self._gmm_params, return_mean=False)
                gmm_energy_np = gmm_energy_score.detach().cpu().numpy()

                if self.gmm_score_mode == "standardized":
                    iscore = self._combine_gmm_standardized(iscore, gmm_energy_np, self.gmm_weight)
                elif self.gmm_score_mode == "percentile":
                    iscore = self._combine_gmm_percentile(iscore, gmm_energy_np)
                else:
                    raise ValueError(f"Unknown gmm_score_mode: {self.gmm_score_mode}")
            else:
                raise ValueError(
                    "When use_gmm=True, model must return tuple with latent representation as second element",
                )

        return iscore, fscore

    def _combine_gmm_standardized(
        self, recon_scores: NDArray, gmm_energy_scores: NDArray, gmm_weight: float
    ) -> NDArray:
        """Combine reconstruction and GMM scores using standardized (z-score) fusion."""
        recon_mean = self._recon_ref_mean if self._recon_ref_mean is not None else recon_scores.mean()
        recon_std = self._recon_ref_std if self._recon_ref_std is not None else (recon_scores.std() + 1e-10)
        recon_standardized = (recon_scores - recon_mean) / (recon_std + 1e-10)

        gmm_mean = self._gmm_energy_ref_mean if self._gmm_energy_ref_mean is not None else gmm_energy_scores.mean()
        gmm_std = (
            self._gmm_energy_ref_std if self._gmm_energy_ref_std is not None else (gmm_energy_scores.std() + 1e-10)
        )
        gmm_standardized = (gmm_energy_scores - gmm_mean) / (gmm_std + 1e-10)

        return (1 - gmm_weight) * recon_standardized + gmm_weight * gmm_standardized

    def _combine_gmm_percentile(self, recon_scores: NDArray, gmm_energy_scores: NDArray) -> NDArray:
        """Combine reconstruction and GMM scores using percentile-based fusion."""
        recon_mean = self._recon_ref_mean if self._recon_ref_mean is not None else recon_scores.mean()
        recon_std = self._recon_ref_std if self._recon_ref_std is not None else (recon_scores.std() + 1e-10)
        recon_z = (recon_scores - recon_mean) / (recon_std + 1e-10)
        recon_percentile = norm.cdf(recon_z)

        gmm_mean = self._gmm_energy_ref_mean if self._gmm_energy_ref_mean is not None else gmm_energy_scores.mean()
        gmm_std = (
            self._gmm_energy_ref_std if self._gmm_energy_ref_std is not None else (gmm_energy_scores.std() + 1e-10)
        )
        gmm_z = (gmm_energy_scores - gmm_mean) / (gmm_std + 1e-10)
        gmm_percentile = norm.cdf(gmm_z)

        return 1.0 - ((1.0 - recon_percentile) * (1.0 - gmm_percentile))
