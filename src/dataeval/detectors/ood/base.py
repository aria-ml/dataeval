"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

__all__ = []

from typing import Callable, cast

import torch

from dataeval.config import DeviceLike, get_device
from dataeval.detectors.ood.mixin import OODBaseMixin, OODFitMixin, OODGMMMixin
from dataeval.typing import ArrayLike
from dataeval.utils._array import to_numpy
from dataeval.utils.torch._gmm import GaussianMixtureModelParams, gmm_params
from dataeval.utils.torch._internal import trainer


class OODBase(OODBaseMixin[torch.nn.Module], OODFitMixin[Callable[..., torch.nn.Module], torch.optim.Optimizer]):
    def __init__(self, model: torch.nn.Module, device: DeviceLike | None = None) -> None:
        self.device: torch.device = get_device(device)
        super().__init__(model)

    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float,
        loss_fn: Callable[..., torch.nn.Module] | None,
        optimizer: torch.optim.Optimizer | None,
        epochs: int,
        batch_size: int,
        verbose: bool,
    ) -> None:
        """
        Train the model and infer the threshold value.

        Parameters
        ----------
        x_ref : ArrayLike
            Training data.
        threshold_perc : float, default 100.0
            Percentage of reference data that is normal.
        loss_fn : Callable | None, default None
            Loss function used for training.
        optimizer : Optimizer, default keras.optimizers.Adam
            Optimizer used for training.
        epochs : int, default 20
            Number of training epochs.
        batch_size : int, default 64
            Batch size used for training.
        verbose : bool, default True
            Whether to print training progress.
        """

        # Train the model
        trainer(
            model=self.model,
            x_train=to_numpy(x_ref),
            y_train=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            preprocess_fn=None,
            epochs=epochs,
            batch_size=batch_size,
            device=self.device,
            verbose=verbose,
        )

        # Infer the threshold values
        self._ref_score = self.score(x_ref, batch_size)
        self._threshold_perc = threshold_perc


class OODBaseGMM(OODBase, OODGMMMixin[GaussianMixtureModelParams]):
    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float,
        loss_fn: Callable[..., torch.nn.Module] | None,
        optimizer: torch.optim.Optimizer | None,
        epochs: int,
        batch_size: int,
        verbose: bool,
    ) -> None:
        super().fit(x_ref, threshold_perc, loss_fn, optimizer, epochs, batch_size, verbose)

        # Calculate the GMM parameters
        _, z, gamma = cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor], self.model(x_ref))
        self._gmm_params = gmm_params(z, gamma)
