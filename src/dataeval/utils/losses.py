"""
Loss functions for training neural networks.

Adapted for PyTorch from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

__all__ = ["ELBOLoss"]

import torch
import torch.nn.functional as F


class ELBOLoss:
    """
    Class-based ELBO loss function for flexible configuration.

    This class provides a callable loss function that can be initialized
    with specific parameters (like beta for beta-VAE) and then used like
    PyTorch's built-in loss functions (e.g., nn.MSELoss()).

    Parameters
    ----------
    beta : float, default 1.0
        Weight for the KL divergence term. Higher values encourage
        learning a more regular latent space (beta-VAE).
    reduction : str, default "mean"
        Reduction method for reconstruction loss: "mean", "sum", or "none".
        Note: KL divergence always uses mean reduction.

    Examples
    --------
    Basic usage with default beta:

    >>> from dataeval.utils.losses import ELBOLoss
    >>> import torch
    >>>
    >>> loss_fn = ELBOLoss(beta=1.0)
    >>> x = torch.rand(32, 1, 28, 28)
    >>> x_recon = torch.rand(32, 1, 28, 28)
    >>> mu = torch.rand(32, 128)
    >>> logvar = torch.rand(32, 128)
    >>> loss = loss_fn(x, x_recon, mu, logvar)

    Using beta-VAE with higher beta for disentanglement:

    >>> loss_fn = ELBOLoss(beta=4.0)
    >>> loss = loss_fn(x, x_recon, mu, logvar)

    Using with OODReconstruction:

    >>> from dataeval.shift import OODReconstruction
    >>> from dataeval.utils.models import VAE
    >>>
    >>> vae_model = VAE(input_shape=(1, 28, 28))
    >>> ood = OODReconstruction(vae_model, model_type="vae")
    >>> custom_loss = ELBOLoss(beta=2.0)
    >>> ood.fit(x, threshold_perc=95, loss_fn=custom_loss, epochs=20)
    """

    def __init__(self, beta: float = 1.0, reduction: str = "mean") -> None:
        """
        Initialize VAE loss function.

        Parameters
        ----------
        beta : float, default 1.0
            Weight for KL divergence term
        reduction : str, default "mean"
            Reduction method: "mean", "sum", or "none"
        """
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")

        self.beta = beta
        self.reduction = reduction

    def __call__(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute VAE loss.

        Parameters
        ----------
        x : torch.Tensor
            Original input tensor
        x_recon : torch.Tensor
            Reconstructed output tensor
        mu : torch.Tensor
            Mean of latent distribution
        logvar : torch.Tensor
            Log-variance of latent distribution

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(
            x_recon.view(len(x), -1),
            x.view(len(x), -1),
            reduction=self.reduction,
        )

        # KL divergence loss (always use mean reduction)
        kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        return recon_loss + self.beta * kld_loss

    def __repr__(self) -> str:
        """Return string representation of the loss function."""
        return f"ELBOLoss(beta={self.beta}, reduction='{self.reduction}')"
