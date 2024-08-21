"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from typing import Callable

import keras
import numpy as np

from dataeval._internal.detectors.ood.base import OODBase, OODScore
from dataeval._internal.interop import ArrayLike, to_numpy
from dataeval._internal.models.tensorflow.autoencoder import AE
from dataeval._internal.models.tensorflow.utils import predict_batch


class OOD_AE(OODBase):
    def __init__(self, model: AE) -> None:
        """
        Autoencoder based out-of-distribution detector.

        Parameters
        ----------
        model : AE
            An Autoencoder model.
        """
        super().__init__(model)

    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float = 100.0,
        loss_fn: Callable = keras.losses.MeanSquaredError(),
        optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam,
        epochs: int = 20,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> None:
        """
        Train the AE model with recommended loss function and optimizer.

        Parameters
        ----------
        x_ref : ArrayLike
            Training batch.
        threshold_perc : float, default 100.0
            Percentage of reference data that is normal.
        loss_fn : Callable, default keras.losses.MeanSquaredError()
            Loss function used for training.
        optimizer : keras.optimizers.Optimizer, default keras.optimizers.Adam
            Optimizer used for training.
        epochs : int, default 20
            Number of training epochs.
        batch_size : int, default 64
            Batch size used for training.
        verbose : bool, default True
            Whether to print training progress.
        """
        super().fit(to_numpy(x_ref), threshold_perc, loss_fn, optimizer, epochs, batch_size, verbose)

    def score(self, X: ArrayLike, batch_size: int = int(1e10)) -> OODScore:
        self._validate(X := to_numpy(X))

        # reconstruct instances
        X_recon = predict_batch(X, self.model, batch_size=batch_size)

        # compute feature and instance level scores
        fscore = np.power(X - X_recon, 2)
        fscore_flat = fscore.reshape(fscore.shape[0], -1).copy()
        n_score_features = int(np.ceil(fscore_flat.shape[1]))
        sorted_fscore = np.sort(fscore_flat, axis=1)
        sorted_fscore_perc = sorted_fscore[:, -n_score_features:]
        iscore = np.mean(sorted_fscore_perc, axis=1)

        return OODScore(iscore, fscore)
