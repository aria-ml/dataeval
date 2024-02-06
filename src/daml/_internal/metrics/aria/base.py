from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn

from daml._internal.metrics.aria.utils import permute_to_torch, pytorch_to_numpy
from daml._internal.models.pytorch.autoencoder import AERunner, AETrainer

TOutput = TypeVar("TOutput")


class _BaseMetric(ABC, Generic[TOutput]):
    """Abstract base class for metrics"""

    def __init__(
        self,
        data: np.ndarray,
        encode: bool,
        model: Optional[nn.Module] = None,
        fit: Optional[bool] = None,
        epochs: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.data = data
        self.encode = encode
        self.model = model

        # TODO: Model training args will move out of metrics
        self.fit = fit
        self.epochs = epochs
        self._device = device

        self._validate_args()

    def evaluate(self) -> TOutput:
        # TODO: Split model training from metrics and make evaluate the abstractmethod
        if self.fit or self.encode:
            self._fit()
        return self._evaluate()

    @abstractmethod
    def _evaluate(self) -> TOutput:
        """Abstract method to calculate metric based off of constructor parameters"""

    def _fit(self) -> None:
        """
        Trains a model on a dataset to be used during calculation of metrics.
        """
        self._validate_args()
        if isinstance(self.model, nn.Module) and not self.fit:
            self.model = AERunner(model=self.model, device=self._device)
        else:
            images = permute_to_torch(self.data)
            if isinstance(self.model, nn.Module):
                self.model = AETrainer(model=self.model, device=self._device)
            else:
                self.model = AETrainer(channels=images.shape[1], device=self._device)

    def _encode(self, images: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Takes an array or tensor of images and returns a tensor that is encoded if
        `self.encode` is True
        """
        self._validate_args()

        if self.encode:
            if not isinstance(images, torch.Tensor):
                images = permute_to_torch(images)
            assert isinstance(self.model, nn.Module)
            images = self.model.encode(images).detach().cpu().numpy()

        return images if isinstance(images, np.ndarray) else pytorch_to_numpy(images)

    def _validate_args(self):
        if self.model is not None and not isinstance(self.model, nn.Module):
            raise TypeError(
                f"Given model is of type {type(self.model)}, expected nn.Module"
            )

        if not (self.model is None and self.fit is None and self.epochs is None):
            if self.model is None:
                raise ValueError(
                    "Must specify `model` if model arguments are provided."
                )
            if self.fit is None:
                raise ValueError("Must specify `fit` if model arguments are provided.")
            if self.epochs is None and self.fit:
                raise ValueError("Must specify `epochs` to fit model.")

        if not self.encode and self.model is not None:
            raise ValueError("Model not used if `encode` is `false`.")
