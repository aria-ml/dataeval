from typing import Any, List, Union

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(0)


def get_images_from_batch(batch: Any) -> Any:
    """Extracts images from a batch of collated data by DataLoader"""
    return batch[0] if isinstance(batch, (list, tuple)) else batch


class AETrainer:
    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device] = "auto",
        batch_size: int = 8,
    ):
        """
        model : nn.Module
            Model to be trained
        device : str | torch.device, default "cpu"
            Hardware device for model, optimizer, and data to run on
        batch_size : int, default 8
            Number of images to group together in `torch.utils.data.DataLoader`
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.to(device)
        self.batch_size = batch_size

    def train(self, dataset: Dataset, epochs: int = 25) -> List[float]:
        """
        Basic training function for Autoencoder models for reconstruction tasks

        Uses `torch.optim.Adam` and `torch.nn.MSELoss` as default hyperparameters

        Parameters
        ----------
        dataset : Dataset
            Torch Dataset containing images in the first return position
        epochs : int, default 25
            Number of full training loops

        Note
        ----
        To replace this function with a custom function, do
            AETrainer.train = custom_function
        """
        # Setup training
        self.model.train()
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        opt = Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss().to(self.device)
        # Record loss
        loss_history: List[float] = []

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
        Basic evaluation function for Autoencoder models for reconstruction tasks

        Uses `torch.optim.Adam` and `torch.nn.MSELoss` as default hyperparameters

        Parameters
        ----------
        dataset : Dataset
            Torch Dataset containing images in the first return position

        Returns
        -------
        float
            Total reconstruction loss over all data

        Note
        ----
        To replace this function with a custom function, do
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
        Encode data through model if it has an encode attribute,
        otherwise passes data through model.forward

        Parameters
        ----------
        dataset: Dataset
            Dataset containing images to be encoded by the model

        Returns
        -------
        torch.Tensor
            Data encoded by the model
        """
        self.model.eval()
        dl = DataLoader(dataset, batch_size=self.batch_size)
        encodings = torch.Tensor([])

        # Get encode function if defined
        if getattr(self.model, "encode", None) is not None:
            encode_func = self.model.encode
        else:
            encode_func = self.model.forward
        # Accumulate encodings from batches
        for batch in dl:
            imgs = get_images_from_batch(batch)
            imgs = imgs.to(self.device)
            embeddings = encode_func(imgs).to("cpu")
            if len(encodings):
                encodings = torch.vstack((encodings, embeddings))
            else:
                encodings = embeddings

        return encodings


class AriaAutoencoder(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.encoder = Encoder(channels)
        self.decoder = Decoder(channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)


class Encoder(nn.Module):
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
        return self.encoder(x)


class Decoder(nn.Module):
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
        return self.decoder(x)
