from typing import Any, Union

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(0)


def get_images_from_batch(batch: Any) -> Any:
    """Extracts images from a batch of collated data by DataLoader"""
    if isinstance(batch, (list, tuple)):
        imgs = batch[0]
    else:
        imgs = batch
    return imgs


class AETrainer:
    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device] = "cpu",
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

        self.device = device
        self.model = model.to(device)
        self.batch_size = batch_size

    def train(self, dataset: Dataset, epochs: int = 25):
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
        self.model.train()
        dl = DataLoader(dataset, batch_size=self.batch_size)
        opt = Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss().to(self.device)
        loss = 0
        for _ in range(epochs):
            for batch in dl:
                imgs = get_images_from_batch(batch)
                imgs = imgs.to(self.device)
                opt.zero_grad()
                pred = self.model(imgs)
                loss = criterion(pred, imgs)
                loss.backward()
                opt.step()

    def eval(self, dataset: Dataset) -> float:
        """
        Basic training function for Autoencoder models for reconstruction tasks

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
        dl = DataLoader(dataset, batch_size=self.batch_size)
        criterion = nn.MSELoss().to(self.device)
        total_loss: float = 0.0
        with torch.no_grad():
            for batch in dl:
                imgs = get_images_from_batch(batch)
                imgs = imgs.to(self.device)
                pred = self.model(imgs)
                loss = criterion(pred, imgs)
                total_loss += loss
        return total_loss

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

        if getattr(self.model, "encode", None) is not None:
            encode_func = self.model.encode
        else:
            encode_func = self.model.forward
        with torch.no_grad():
            for batch in dl:
                imgs = get_images_from_batch(batch)
                imgs = imgs.to(self.device)
                embeddings = encode_func(imgs)
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
