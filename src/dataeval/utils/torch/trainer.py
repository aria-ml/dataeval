"""Utility classes for training PyTorch models."""

from __future__ import annotations

__all__ = ["AETrainer"]

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from dataeval.config import DeviceLike, get_device


def get_images_from_batch(batch: Any) -> Any:
    """Extracts images from a batch of collated data by DataLoader"""
    return batch[0] if isinstance(batch, list | tuple) else batch


class AETrainer:
    """
    A class to train and evaluate an autoencoder<Autoencoder>` model.

    Parameters
    ----------
    model : nn.Module
        The model to be trained.
    device : DeviceLike or None, default None
        The hardware device to use if specified, otherwise uses the DataEval
        default or torch default.
    batch_size : int, default 8
        The number of images to process in a batch.
    """

    def __init__(
        self,
        model: nn.Module,
        device: DeviceLike | None = None,
        batch_size: int = 8,
    ) -> None:
        self.device: torch.device = get_device(device)
        self.model: nn.Module = model.to(self.device)
        self.batch_size = batch_size

    def train(self, dataset: Dataset[Any], epochs: int = 25) -> list[float]:
        """
        Basic image reconstruction training function for :term:`Autoencoder` models

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
            A list of average loss values for each :term:`epoch<Epoch>`.

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
    def eval(self, dataset: Dataset[Any]) -> float:
        """
        Basic image reconstruction evaluation function for :term:`autoencoder<Autoencoder>` models

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
    def encode(self, dataset: Dataset[Any]) -> torch.Tensor:
        """
        Create image :term:`embeddings<Embeddings>` for the dataset using the model's encoder.

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
            embeddings = encode_func(imgs).to("cpu")  # type: ignore
            encodings = torch.vstack((encodings, embeddings)) if len(encodings) else embeddings

        return encodings
