from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(0)


def get_images_from_batch(batch: Any) -> Any:
    """Extracts images from a batch of collated data by DataLoader"""
    return batch[0] if isinstance(batch, (list, tuple)) else batch


class AETrainer:
    """
    A class to train and evaluate an autoencoder model.

    Parameters
    ----------
    model : nn.Module
        The model to be trained.
    device : str or torch.device, default "auto"
        The hardware device to use for training.
        If "auto", the device will be set to "cuda" if available, otherwise "cpu".
    batch_size : int, default 8
        The number of images to process in a batch.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "auto",
        batch_size: int = 8,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.to(device)
        self.batch_size = batch_size

    def train(self, dataset: Dataset, epochs: int = 25) -> list[float]:
        """
        Basic image reconstruction training function for Autoencoder models

        Uses `torch.optim.Adam` and `torch.nn.MSELoss` as default hyperparameters

        Parameters
        ----------
        dataset : Dataset
            The dataset to train on.
            Torch Dataset containing images in the first return position.
        epochs : int, default 25
            Number of full training loops

        Returns
        -------
        List[float]
            A list of average loss values for each epoch.

        Note
        ----
        To replace this function with a custom function, do:
            AETrainer.train = custom_function
        """
        # Setup training
        self.model.train()
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        opt = Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss().to(self.device)
        # Record loss
        loss_history: list[float] = []

        for _ in range(epochs):
            epoch_loss: float = 0
            for batch in dataloader:
                imgs = get_images_from_batch(batch)
                imgs = imgs.to(self.device)
                # Zero your gradients for every batch!
                opt.zero_grad()

                # Make predictions for this batch
                pred = self.model(imgs)

                # Compute the loss and its gradients
                loss = criterion(pred, imgs)
                loss.backward()

                # Adjust learning weights
                opt.step()

                # Gather data and report
                epoch_loss += loss.item()
            # Will take the average from all batches
            epoch_loss /= len(dataloader)
            loss_history.append(epoch_loss)

        return loss_history

    @torch.no_grad
    def eval(self, dataset: Dataset) -> float:
        """
        Basic image reconstruction evaluation function for Autoencoder models

        Uses `torch.nn.MSELoss` as default loss function.

        Parameters
        ----------
        dataset : Dataset
            The dataset to evaluate on.
            Torch Dataset containing images in the first return position.

        Returns
        -------
        float
            Total reconstruction loss over the entire dataset

        Note
        ----
        To replace this function with a custom function, do:
            AETrainer.eval = custom_function
        """
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        criterion = nn.MSELoss().to(self.device)
        total_loss: float = 0.0

        for batch in dataloader:
            imgs = get_images_from_batch(batch)
            imgs = imgs.to(self.device)
            pred = self.model(imgs)
            loss = criterion(pred, imgs)
            total_loss += loss.item()
        return total_loss / len(dataloader)

    @torch.no_grad
    def encode(self, dataset: Dataset) -> torch.Tensor:
        """
        Create image embeddings for the dataset using the model's encoder.

        If the model has an `encode` method, it will be used; otherwise,
        `model.forward` will be used.

        Parameters
        ----------
        dataset: Dataset
            The dataset to encode.
            Torch Dataset containing images in the first return position.

        Returns
        -------
        torch.Tensor
            Data encoded by the model

        Note
        ----
        This function should be run after the model has been trained and evaluated.
        """
        self.model.eval()
        dl = DataLoader(dataset, batch_size=self.batch_size)
        encodings = torch.Tensor([])

        # Get encode function if defined
        encode_func = self.model.encode if getattr(self.model, "encode", None) else self.model.forward

        # Accumulate encodings from batches
        for batch in dl:
            imgs = get_images_from_batch(batch)
            imgs = imgs.to(self.device)
            embeddings = encode_func(imgs).to("cpu")
            encodings = torch.vstack((encodings, embeddings)) if len(encodings) else embeddings

        return encodings


class AriaAutoencoder(nn.Module):
    """
    An autoencoder model with a separate encoder and decoder.

    Parameters
    ----------
    channels : int, default 3
        Number of input channels
    """

    def __init__(self, channels=3):
        super().__init__()
        self.encoder = Encoder(channels)
        self.decoder = Decoder(channels)

    def forward(self, x):
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

    def encode(self, x):
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

    def __init__(self, channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 256, 2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 128, 2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 2, stride=1),
        )

    def forward(self, x):
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

    def __init__(self, channels):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 2, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, channels, 2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
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
