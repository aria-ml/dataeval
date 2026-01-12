"""Simple PyTorch model architectures used by DataEval."""

__all__ = ["AE", "Encoder", "Decoder", "VAE", "VAEEncoder", "VAEDecoder", "GMMDensityNet"]

import math
from typing import Any

import torch
import torch.nn as nn


class GMMDensityNet(nn.Module):
    """
    Gaussian Mixture Model (GMM) density network for converting latent representations
    to mixture assignment probabilities.

    This network can be appended to AE or VAE models to enable GMM-based OOD detection
    by producing gamma (mixture assignment probabilities) from the latent representation.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space (encoding dimension from AE/VAE).
    n_gmm : int, default 2
        Number of Gaussian mixture components.
    hidden_dim : int, default 10
        Number of hidden units in the dense layer.

    Example
    -------
    Creating a VAE with GMM density estimation:

    >>> from dataeval.shift import OODReconstruction
    >>> from dataeval.utils.models import VAE, GMMDensityNet
    >>>
    >>> # Use with OODReconstruction
    >>> gmm_density_net = GMMDensityNet(latent_dim=256, n_gmm=3)
    >>> vae_gmm_model = VAE(input_shape=(1, 28, 28), gmm_density_net=gmm_density_net)
    >>> ood = OODReconstruction(vae_gmm_model, model_type="vae", use_gmm=True)

    Notes
    -----
    The network architecture is based on the GMM density network from Alibi-Detect,
    adapted from TensorFlow to PyTorch. It consists of:
    - A hidden layer with tanh activation
    - An output layer with softmax activation to produce valid probability distributions
    """

    def __init__(
        self,
        latent_dim: int,
        n_gmm: int = 2,
        hidden_dim: int = 10,
    ) -> None:
        super().__init__()

        if n_gmm < 1:
            raise ValueError(f"n_gmm must be at least 1, got {n_gmm}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be at least 1, got {latent_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be at least 1, got {hidden_dim}")

        self.latent_dim = latent_dim
        self.n_gmm = n_gmm
        self.hidden_dim = hidden_dim

        self.network: nn.Module = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_gmm),
            nn.Softmax(dim=-1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Convert latent representation to mixture assignment probabilities.

        Parameters
        ----------
        z : torch.Tensor
            Latent representation with shape (batch_size, latent_dim).

        Returns
        -------
        torch.Tensor
            Mixture assignment probabilities (gamma) with shape (batch_size, n_gmm).
            Each row sums to 1.0 and represents the probability distribution over
            mixture components for that sample.
        """
        return self.network(z)


class Encoder(nn.Module):
    """
    A simple encoder to be used in an autoencoder model.

    It consists of a CNN followed by a fully connected layer.

    Parameters
    ----------
    input_shape : tuple[int, int, int]
        Shape of the input data in CHW format.
    encoding_dim : int
        The size of the 1D array that emerges from the fully connected layer.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        encoding_dim: int,
    ) -> None:
        super().__init__()

        channels = input_shape[0]
        nc_in, nc_mid, nc_done = 256, 128, 64

        conv_in = nn.Conv2d(channels, nc_in, 2, stride=1, padding=1)
        conv_mid = nn.Conv2d(nc_in, nc_mid, 2, stride=1, padding=1)
        conv_done = nn.Conv2d(nc_mid, nc_done, 2, stride=1)

        self.encoding_ops: nn.Sequential = nn.Sequential(
            conv_in,
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            conv_mid,
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            conv_done,
        )

        ny, nx = input_shape[1:]
        self.encoding_shape: tuple[int, int, int] = (nc_done, ny // 4 - 1, nx // 4 - 1)
        self.flatcon: int = math.prod(self.encoding_shape)
        self.flatten: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.flatcon,
                encoding_dim,
            ),
        )

    def forward(self, x: Any) -> Any:
        """
        Perform a forward pass through the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            The encoded representation of the input tensor.
        """
        x = self.encoding_ops(x)

        return self.flatten(x)


class Decoder(nn.Module):
    """
    A simple decoder to be used in an autoencoder model.

    Parameters
    ----------
    input_shape : tuple[int, ...]
        Shape of the original input data in CHW format.
    encoding_dim : int
        The size of the 1D array that emerges from the fully connected layer.
    encoding_shape : tuple[int, ...]
        Shape of the encoded input data.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        encoding_dim: int,
        encoding_shape: tuple[int, ...],
    ) -> None:
        super().__init__()

        self.encoding_shape = encoding_shape
        self.input_shape = input_shape  # need to store this for use in forward().
        channels = input_shape[0]

        self.input: nn.Linear = nn.Linear(encoding_dim, math.prod(encoding_shape))

        self.decoder: nn.Sequential = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 2, stride=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 256, 2, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, channels, 2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the decoder.

        Parameters
        ----------
        x : torch.Tensor
            The encoded tensor.

        Returns
        -------
        torch.Tensor
            The reconstructed output tensor.
        """
        x = self.input(x)
        x = x.reshape((-1, *self.encoding_shape))
        x = self.decoder(x)
        return x.reshape((-1, *self.input_shape))


class AE(nn.Module):
    """
    An autoencoder model with a separate encoder and decoder.

    Parameters
    ----------
    input_shape : tuple[int, ...]
        Shape of the input data in CHW format.
    gmm_density_net : GMMDensityNet or None, default None
        Optional GMM density network to enable GMM-based OOD detection.
        If provided, the forward pass will return (reconstruction, z, gamma)
        instead of just reconstruction. The GMMDensityNet's latent_dim must
        match the encoder's encoding dimension.

    Example
    -------
    Sample data:

    >>> x = torch.randn(32, 1, 28, 28)

    Standard autoencoder:

    >>> ae = AE(input_shape=(1, 28, 28))
    >>> reconstruction = ae(x)

    Autoencoder with GMM for OOD detection:

    >>> from dataeval.utils.models import GMMDensityNet
    >>> gmm_density_net = GMMDensityNet(latent_dim=256, n_gmm=3)
    >>> ae_gmm = AE(input_shape=(1, 28, 28), gmm_density_net=gmm_density_net)
    >>> reconstruction, z, gamma = ae_gmm(x)
    >>> # Use with OODReconstruction(ae_gmm, model_type="ae", use_gmm=True)
    """

    def __init__(self, input_shape: tuple[int, ...], gmm_density_net: GMMDensityNet | None = None) -> None:
        super().__init__()

        if len(input_shape) != 3:
            raise ValueError("Expected input_shape to be in CHW format.")

        input_dim = math.prod(input_shape)

        # It makes an odd staircase that is basically proportional to the number of
        # numbers in the image to the 0.8 power.
        encoding_dim = int(math.pow(2, int(input_dim.bit_length() * 0.8)))
        self.encoder: Encoder = Encoder(input_shape, encoding_dim)
        self.decoder: Decoder = Decoder(input_shape, encoding_dim, self.encoder.encoding_shape)
        self.gmm_density_net = gmm_density_net

        # Validate GMM density net latent dimension matches encoder output
        if self.gmm_density_net is not None and self.gmm_density_net.latent_dim != encoding_dim:
            raise ValueError(
                f"GMMDensityNet latent_dim ({self.gmm_density_net.latent_dim}) must match "
                f"encoder encoding_dim ({encoding_dim}). Either create GMMDensityNet with "
                f"latent_dim={encoding_dim}, or let AE auto-create it by passing n_gmm instead."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the encoder and decoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            If gmm_density_net is None: returns reconstructed output tensor.
            If gmm_density_net is provided: returns (reconstruction, z, gamma) where
            z is the latent representation and gamma is the mixture assignment probabilities.
        """
        z = self.encoder(x)
        reconstruction = self.decoder(z)

        if self.gmm_density_net is not None:
            gamma = self.gmm_density_net(z)
            return reconstruction, z, gamma

        return reconstruction


class VAEEncoder(nn.Module):
    """
    Variational Autoencoder Encoder that outputs mean and log-variance.

    Parameters
    ----------
    input_shape : tuple[int, int, int]
        Shape of the input data in CHW format.
    latent_dim : int
        The size of the latent space (encoding dimension).
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        latent_dim: int,
    ) -> None:
        super().__init__()

        channels = input_shape[0]
        nc_in, nc_mid, nc_done = 256, 128, 64

        conv_in = nn.Conv2d(channels, nc_in, 2, stride=1, padding=1)
        conv_mid = nn.Conv2d(nc_in, nc_mid, 2, stride=1, padding=1)
        conv_done = nn.Conv2d(nc_mid, nc_done, 2, stride=1)

        self.encoding_ops: nn.Sequential = nn.Sequential(
            conv_in,
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            conv_mid,
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            conv_done,
        )

        ny, nx = input_shape[1:]
        self.encoding_shape: tuple[int, int, int] = (nc_done, ny // 4 - 1, nx // 4 - 1)
        self.flatcon: int = math.prod(self.encoding_shape)

        # Separate layers for mean and log-variance
        self.fc_mu: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatcon, latent_dim),
        )
        self.fc_logvar: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatcon, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Mean (mu) and log-variance (logvar) of the latent distribution.
        """
        x = self.encoding_ops(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    Variational Autoencoder Decoder.

    Parameters
    ----------
    input_shape : tuple[int, ...]
        Shape of the original input data in CHW format.
    latent_dim : int
        The size of the latent space (encoding dimension).
    encoding_shape : tuple[int, ...]
        Shape of the encoded input data.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        latent_dim: int,
        encoding_shape: tuple[int, ...],
    ) -> None:
        super().__init__()

        self.encoding_shape = encoding_shape
        self.input_shape = input_shape
        channels = input_shape[0]

        self.input: nn.Linear = nn.Linear(latent_dim, math.prod(encoding_shape))

        self.decoder: nn.Sequential = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 2, stride=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 256, 2, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, channels, 2, stride=2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the decoder.

        Parameters
        ----------
        z : torch.Tensor
            The latent representation.

        Returns
        -------
        torch.Tensor
            The reconstructed output tensor.
        """
        x = self.input(z)
        x = x.reshape((-1, *self.encoding_shape))
        x = self.decoder(x)
        return x.reshape((-1, *self.input_shape))


class VAE(nn.Module):
    """
    Variational Autoencoder model with separate encoder and decoder.

    Parameters
    ----------
    input_shape : tuple[int, ...]
        Shape of the input data in CHW format.
    latent_dim : int or None, default None
        The size of the latent space. If None, will be computed automatically.
    gmm_density_net : GMMDensityNet or None, default None
        Optional GMM density network to enable GMM-based OOD detection.
        If provided, the forward pass will return (reconstruction, z, gamma)
        instead of (reconstruction, mu, logvar). The GMMDensityNet's latent_dim
        must match the VAE's latent dimension.

    Example
    -------
    Sample data:

    >>> x = torch.randn(32, 1, 28, 28)

    Standard VAE:

    >>> vae = VAE(input_shape=(1, 28, 28))
    >>> recon, mu, logvar = vae(x)

    VAE with GMM for OOD detection:

    >>> from dataeval.utils.models import VAE, GMMDensityNet
    >>> vae_gmm = VAE(input_shape=(1, 28, 28), gmm_density_net=GMMDensityNet(latent_dim=256, n_gmm=3))
    >>> reconstruction, z, gamma = vae_gmm(x)
    >>> # Use with OODReconstruction(vae_gmm) - auto-detects as VAE with GMM
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        latent_dim: int | None = None,
        gmm_density_net: GMMDensityNet | None = None,
    ) -> None:
        super().__init__()

        if len(input_shape) != 3:
            raise ValueError("Expected input_shape to be in CHW format.")

        if latent_dim is None:
            input_dim = math.prod(input_shape)
            latent_dim = int(math.pow(2, int(input_dim.bit_length() * 0.8)))

        self.latent_dim = latent_dim
        self.encoder: VAEEncoder = VAEEncoder(input_shape, latent_dim)
        self.decoder: VAEDecoder = VAEDecoder(input_shape, latent_dim, self.encoder.encoding_shape)
        self.gmm_density_net = gmm_density_net

        # Validate GMM density net latent dimension matches VAE latent dimension
        if self.gmm_density_net is not None and self.gmm_density_net.latent_dim != latent_dim:
            raise ValueError(
                f"GMMDensityNet latent_dim ({self.gmm_density_net.latent_dim}) must match "
                f"VAE latent_dim ({latent_dim}). Create GMMDensityNet with latent_dim={latent_dim}."
            )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) using N(0,1).

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent Gaussian.
        logvar : torch.Tensor
            Log-variance of the latent Gaussian.

        Returns
        -------
        torch.Tensor
            Sampled latent vector.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the VAE.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            If gmm_density_net is None: returns (reconstruction, mu, logvar).
            If gmm_density_net is provided: returns (reconstruction, z, gamma) where
            z is the latent representation (mu) and gamma is the mixture assignment probabilities.
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)

        if self.gmm_density_net is not None:
            # For GMM, use mu (not sampled z) as the latent representation
            # This provides more stable mixture assignments
            gamma = self.gmm_density_net(mu)
            return recon, mu, gamma

        return recon, mu, logvar
