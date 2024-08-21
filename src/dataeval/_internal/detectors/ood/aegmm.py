"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from typing import Callable

import keras

from dataeval._internal.detectors.ood.base import OODGMMBase, OODScore
from dataeval._internal.interop import ArrayLike, to_numpy
from dataeval._internal.models.tensorflow.autoencoder import AEGMM
from dataeval._internal.models.tensorflow.gmm import gmm_energy
from dataeval._internal.models.tensorflow.losses import LossGMM
from dataeval._internal.models.tensorflow.utils import predict_batch


class OOD_AEGMM(OODGMMBase):
    def __init__(self, model: AEGMM) -> None:
        """
        AE with Gaussian Mixture Model based outlier detector.

        Parameters
        ----------
        model : AEGMM
            An AEGMM model.
        """
        super().__init__(model)

    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float = 100.0,
        loss_fn: Callable = LossGMM(),
        optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam,
        epochs: int = 20,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> None:
        """
        Train the AEGMM model with recommended loss function and optimizer.

        Parameters
        ----------
        x_ref : ArrayLike
            Training batch.
        threshold_perc : float, default 100.0
            Percentage of reference data that is normal.
        loss_fn : Callable, default LossGMM()
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
        super().fit(x_ref, threshold_perc, loss_fn, optimizer, epochs, batch_size, verbose)

    def score(self, X: ArrayLike, batch_size: int = int(1e10)) -> OODScore:
        self._validate(X := to_numpy(X))
        _, z, _ = predict_batch(X, self.model, batch_size=batch_size)
        energy, _ = gmm_energy(z, self.gmm_params, return_mean=False)
        return OODScore(energy.numpy())  # type: ignore
