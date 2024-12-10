from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

__all__ = ["AETrainer", "trainer"]


def get_images_from_batch(batch: Any) -> Any:
    """Extracts images from a batch of collated data by DataLoader"""
    return batch[0] if isinstance(batch, (list, tuple)) else batch


class AETrainer:
    """
    A class to train and evaluate an autoencoder<Autoencoder>` model.

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
        self.device: torch.device = torch.device(device)
        self.model: nn.Module = model.to(device)
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
            embeddings = encode_func(imgs).to("cpu")
            encodings = torch.vstack((encodings, embeddings)) if len(encodings) else embeddings

        return encodings


def trainer(
    model: torch.nn.Module,
    x_train: NDArray[Any],
    y_train: NDArray[Any] | None,
    loss_fn: Callable[..., torch.Tensor | torch.nn.Module] | None,
    optimizer: torch.optim.Optimizer | None,
    preprocess_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    epochs: int,
    batch_size: int,
    device: torch.device,
    verbose: bool,
) -> None:
    """
    Train Pytorch model.

    Parameters
    ----------
    model
        Model to train.
    loss_fn
        Loss function used for training.
    x_train
        Training data.
    y_train
        Training labels.
    optimizer
        Optimizer used for training.
    preprocess_fn
        Preprocessing function applied to each training batch.
    epochs
        Number of training epochs.
    reg_loss_fn
        Allows an additional regularisation term to be defined as reg_loss_fn(model)
    batch_size
        Batch size used for training.
    buffer_size
        Maximum number of elements that will be buffered when prefetching.
    verbose
        Whether to print training progress.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if y_train is None:
        dataset = TensorDataset(torch.from_numpy(x_train).to(torch.float32))

    else:
        dataset = TensorDataset(
            torch.from_numpy(x_train).to(torch.float32), torch.from_numpy(y_train).to(torch.float32)
        )

    loader = DataLoader(dataset=dataset)

    model = model.to(device)

    # iterate over epochs
    loss = torch.nan
    disable_tqdm = not verbose
    for epoch in (pbar := tqdm(range(epochs), disable=disable_tqdm)):
        epoch_loss = loss
        for step, data in enumerate(loader):
            if step % 250 == 0:
                pbar.set_description(f"Epoch: {epoch} ({epoch_loss:.3f}), loss: {loss:.3f}")

            x, y = [d.to(device) for d in data] if len(data) > 1 else (data[0].to(device), None)

            if isinstance(preprocess_fn, Callable):
                x = preprocess_fn(x)

            y_hat = model(x)
            y = x if y is None else y

            loss = loss_fn(y, y_hat)  # type: ignore

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
