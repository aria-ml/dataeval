"""
Adapted for Pytorch from

Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import ArrayLike

from dataeval.detectors.drift.torch import predict_batch
from dataeval.detectors.ood.base_torch import OODBase, OODScoreOutput
from dataeval.interop import as_numpy
from dataeval.output import set_metadata


class OOD_AE(OODBase):
    """
    Autoencoder based out-of-distribution detector.

    Parameters
    ----------
    model : AriaAutoencoder
        An Autoencoder model.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        self.model = model

    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float = 100.0,
        loss_fn: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        epochs: int = 20,
        batch_size: int = 64,
        verbose: bool = False,
    ) -> None:
        if loss_fn is None:
            loss_fn = torch.nn.MSELoss()

        super().fit(as_numpy(x_ref), threshold_perc, loss_fn, optimizer, epochs, batch_size, verbose)

    @set_metadata()
    def score(self, X: ArrayLike, batch_size: int = int(1e10)) -> OODScoreOutput:
        self._validate(X := as_numpy(X))

        # reconstruct instances
        X_recon = predict_batch(X, self.model, batch_size=batch_size)

        # compute feature and instance level scores
        fscore = np.power(X - X_recon, 2)
        fscore_flat = fscore.reshape(fscore.shape[0], -1).copy()
        n_score_features = int(np.ceil(fscore_flat.shape[1]))
        sorted_fscore = np.sort(fscore_flat, axis=1)
        sorted_fscore_perc = sorted_fscore[:, -n_score_features:]
        iscore = np.mean(sorted_fscore_perc, axis=1)

        return OODScoreOutput(iscore, fscore)
