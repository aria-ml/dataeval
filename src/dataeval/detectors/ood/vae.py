"""
Adapted for Pytorch from

Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

__all__ = []

from typing import Callable

import numpy as np
import torch

from dataeval.config import DeviceLike
from dataeval.detectors.ood.base import OODBase
from dataeval.outputs import OODScoreOutput
from dataeval.typing import ArrayLike
from dataeval.utils._array import as_numpy
from dataeval.utils.torch._internal import predict_batch


class OOD_VAE(OODBase):
    """
    Autoencoder based out-of-distribution detector.

    Parameters
    ----------
    model : Autoencoder
        An Autoencoder model.
    """

    def __init__(self, model: torch.nn.Module, device: DeviceLike | None = None) -> None:
        super().__init__(model, device)

    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float,
        loss_fn: Callable[..., torch.nn.Module] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        epochs: int = 20,
        batch_size: int = 64,
        verbose: bool = False,
    ) -> None:
        if loss_fn is None:
            loss_fn = torch.nn.MSELoss()

        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        super().fit(x_ref, threshold_perc, loss_fn, optimizer, epochs, batch_size, verbose)

    def _score(self, X: ArrayLike, batch_size: int = int(1e10)) -> OODScoreOutput:
        self._validate(X := as_numpy(X))

        # reconstruct instances
        X_recon = predict_batch(X, self.model, batch_size=batch_size)[0]  # don't need mu or logvar from model

        # compute feature and instance level scores
        fscore = np.power(X.reshape((len(X), -1)) - X_recon, 2)
        # fscore_flat = fscore.reshape(fscore.shape[0], -1).copy()
        # n_score_features = int(np.ceil(fscore_flat.shape[1]))
        # sorted_fscore = np.sort(fscore_flat, axis=1)
        # sorted_fscore_perc = sorted_fscore[:, -n_score_features:]
        # iscore = np.mean(sorted_fscore_perc, axis=1)
        iscore = np.sum(fscore, axis=1)

        return OODScoreOutput(iscore, fscore)
