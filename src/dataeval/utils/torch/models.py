from __future__ import annotations

__all__ = ["AriaAutoencoder", "Encoder", "Decoder"]

import math
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


# =========================================================================


class AE_torch(nn.Module):
    """
    An autoencoder model with a separate encoder and decoder.

    Parameters
    ----------
    channels : int, default 3
        Number of input channels
    """

    def __init__(self, channels: int = 3, input_shape: tuple[int, int, int, int] = (3, 1, 1, 1)) -> None:
        super().__init__()

        input_dim = math.prod(input_shape)
        encoding_dim = int(math.pow(2, int(input_dim.bit_length() * 0.8)))

        self.encoder: Encoder_AE = Encoder_AE(channels, input_shape, encoding_dim=encoding_dim)
        self.decoder: Decoder_AE = Decoder_AE(channels, encoding_dim)

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


class Encoder_AE(nn.Module):
    """
    A simple encoder to be used in an autoencoder model.

    This is the encoder used to replicate AE, which was a TF function.

    Parameters
    ----------
    channels : int, default 3
        Number of input channels
    """

    def __init__(
        self, channels: int = 3, input_shape: tuple[int, int, int, int] = (1, 3, 1, 1), encoding_dim: int | None = None
    ) -> None:
        super().__init__()

        nc_in, nc_mid, nc_done = 256, 128, 64
        conv_in = nn.Conv2d(channels, nc_in, 2, stride=1, padding=1)
        conv_mid = nn.Conv2d(nc_in, nc_mid, 2, stride=1, padding=1)
        conv_done = nn.Conv2d(nc_mid, nc_done, 2, stride=1)

        self.op_list = [
            conv_in,
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            conv_mid,
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            conv_done,
        ]

        nbatch, channels, ny, nx = input_shape
        post_op_shape = (nbatch, nc_done, ny // 4 - 1, nx // 4 - 1)
        self.crush = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(
                math.prod(post_op_shape[1:]),
                encoding_dim,
            ),
        )

    def forward(self, x: Any) -> Any:
        """
        Perform a forward pass through the AE_torch encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            The encoded representation of the input tensor.
        """
        for op in self.op_list:
            x = op(x)
            print(op, x.shape)
            pass

        x = self.crush(x)

        return x


class Decoder_AE(nn.Module):
    """
    A simple decoder to be used in an autoencoder model.

    This is the decoder used by the AriaAutoencoder model.

    Parameters
    ----------
    channels : int
        Number of output channels
    """

    def __init__(self, channels: int, encoding_dim: int) -> None:
        super().__init__()
        # original
        # self.decoder: nn.Sequential = nn.Sequential(
        #     nn.ConvTranspose2d(64, 128, 2, stride=1),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose2d(128, 256, 2, stride=2),
        #     nn.LeakyReLU(),
        #     nn.ConvTranspose2d(256, channels, 2, stride=2),
        #     nn.Sigmoid(),
        # )
        # hacked
        self.input: nn.Sequential = nn.Sequential(nn.Linear(encoding_dim, 4 * 4 * 128))
        self.decoder: nn.Sequential = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 2, stride=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 256, 2, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, channels, 2, stride=2),
        )

        self.crush = nn.Sequential(nn.Flatten(), nn.Linear(2, 2))

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
        x = self.decoder(x)
        x = self.crush(x)
        return x
