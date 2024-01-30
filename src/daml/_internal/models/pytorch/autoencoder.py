from typing import Optional, Union

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

torch.manual_seed(0)


class AERunner:
    def __init__(self, model: nn.Module, device: Union[str, torch.device] = "cpu"):
        self._model = model
        self._device = device

        self._model.to(self._device)

    def encode(self, x) -> torch.Tensor:
        with torch.no_grad():
            if getattr(self._model, "encode", None) is not None:
                return self._model.encode(x)
            else:  # Call model.forward() if no encode function
                return self._model(x)

    def __call__(self, x):
        with torch.no_grad():
            x = self._model(x)
        return x


class AETrainer(AERunner):
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        channels: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        # Channel check only if no model given
        if model is None:
            if channels is None:
                raise TypeError("Channels must be set for default model")
            _model = AriaAutoencoder(channels)
        else:
            _model = model
        super().__init__(_model, device=device)

    def train(self, dataset, epochs: int = 100, batch_size=8):
        dl = DataLoader(dataset, batch_size=batch_size)

        opt = Adam(self._model.parameters(), lr=0.001)
        criterion = nn.MSELoss().to(self._device)
        loss = 0
        for _ in range(epochs):
            for batch in dl:
                imgs = batch[0].to(self._device)  # (imgs, labels, bboxes)
                opt.zero_grad()
                pred = self._model(imgs)
                loss = criterion(pred, imgs)
                loss.backward()
                opt.step()


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
