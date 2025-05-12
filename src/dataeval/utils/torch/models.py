"""Simple PyTorch model architectures used by DataEval."""

from __future__ import annotations

__all__ = ["Autoencoder", "Encoder", "Decoder", "ResNet18"]

import math
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


@runtime_checkable
class SupportsEncode(Protocol):
    def encode(self, x: Any) -> Any: ...


class Autoencoder(nn.Module):
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
        return self.decoder(x)

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

    This is the encoder used by the Autoencoder model.

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

    This is the decoder used by the Autoencoder model.

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


class AE(nn.Module):
    """
    An autoencoder model with a separate encoder and decoder used as the core of an autoencoder-based
    OOD detector, i.e. as an argument to OOD_AE().

    Parameters
    ----------
    input_shape : tuple[int, int, int]
        Number of input channels, number of rows, number of columns.() Number of examples per batch will be inferred
        at runtime.)
    """

    def __init__(self, input_shape: tuple[int, int, int]) -> None:
        super().__init__()

        input_dim = math.prod(input_shape)

        # following is lifted from src/dataeval/utils/tensorflow/_internal/utils.py. It makes an odd staircase that is
        #  basically proportional to the number of numbers in the image to the 0.8 power. '
        encoding_dim = int(math.pow(2, int(input_dim.bit_length() * 0.8)))

        self.encoder: Encoder_AE = Encoder_AE(input_shape, encoding_dim)

        self.decoder: Decoder_AE = Decoder_AE(input_shape, encoding_dim, self.encoder.post_op_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return self.decoder(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
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

    This is the encoder used to replicate AE, which was a TF function. It consists of a CNN followed by a fully
      connected layer.

    Parameters
    ----------
    channels : int
        Number of input channels

    input_shape : tuple[int, int, int]
        number of channels, number of rows, number of columns in input images.

    encoding_dim : the size of the 1D array that emerges from the fully connected layer.

    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
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
        self.post_op_shape: tuple[int, int, int] = (nc_done, ny // 4 - 1, nx // 4 - 1)
        self.flatcon: int = math.prod(self.post_op_shape)
        self.flatten: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.flatcon,
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
        x = self.encoding_ops(x)

        return self.flatten(x)


class Decoder_AE(nn.Module):
    """
    A simple decoder to be used in an autoencoder model.

    This is the decoder used by the Autoencoder model.

    Parameters
    ----------
    channels : int
        Number of output channels
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        encoding_dim: int,
        post_op_shape: tuple[int, int, int],
    ) -> None:
        super().__init__()

        self.post_op_shape = post_op_shape
        self.input_shape = input_shape  # need to store this for use in forward().
        channels = input_shape[0]

        self.input: nn.Linear = nn.Linear(encoding_dim, math.prod(post_op_shape))

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
        x = x.reshape((-1, *self.post_op_shape))
        x = self.decoder(x)
        return x.reshape((-1, *self.input_shape))


class ResNet18(nn.Module):
    """
    A wrapper class for the torchvision.models.resnet18 model


    Note
    ----
    This class is provided for the use of DataEval documentation and excludes many features
    of the torchvision implementation.

    Warning
    -------
    This class has been thoroughly tested for the purposes
    of DataEval's documentation but not for operational use.
    Please use with caution if deploying this class or subclasses.
    """

    def __init__(self, embedding_size: int = 128) -> None:
        super().__init__()
        self.model: nn.Module = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @staticmethod
    def transforms() -> Any:
        """(Returns) the default ResNet18 IMAGENET1K_V1 transforms"""

        return ResNet18_Weights.DEFAULT.transforms()

    def __str__(self) -> str:
        return str(self.model)
