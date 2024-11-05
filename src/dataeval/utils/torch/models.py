from __future__ import annotations

__all__ = ["AriaAutoencoder", "Encoder", "Decoder"]

from typing import Any

import torch.nn as nn


class AriaAutoencoder(nn.Module):
    """
    An autoencoder model with a separate encoder and decoder.

    Parameters
    ----------
    channels : int, default 3
        Number of input channels
    """

    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.encoder: Encoder = Encoder(channels)
        self.decoder: Decoder = Decoder(channels)

    def forward(self, x: Any) -> Any:
        """
        Perform a forward pass through the encoder and decoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            The reconstructed output tensor.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x: Any) -> Any:
        """
        Encode the input tensor using the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            The encoded representation of the input tensor.
        """
        return self.encoder(x)


class Encoder(nn.Module):
    """
    A simple encoder to be used in an autoencoder model.

    This is the encoder used by the AriaAutoencoder model.

    Parameters
    ----------
    channels : int, default 3
        Number of input channels
    """

    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.encoder: nn.Sequential = nn.Sequential(
            nn.Conv2d(channels, 256, 2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 128, 2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 2, stride=1),
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
        return self.encoder(x)


class Decoder(nn.Module):
    """
    A simple decoder to be used in an autoencoder model.

    This is the decoder used by the AriaAutoencoder model.

    Parameters
    ----------
    channels : int
        Number of output channels
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.decoder: nn.Sequential = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 2, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, channels, 2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x: Any) -> Any:
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
        return self.decoder(x)
