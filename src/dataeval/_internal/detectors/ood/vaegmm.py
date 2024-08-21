"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from typing import Callable

import keras
import numpy as np

from dataeval._internal.detectors.ood.base import OODGMMBase, OODScore
from dataeval._internal.interop import ArrayLike, to_numpy
from dataeval._internal.models.tensorflow.autoencoder import VAEGMM
from dataeval._internal.models.tensorflow.gmm import gmm_energy
from dataeval._internal.models.tensorflow.losses import Elbo, LossGMM
from dataeval._internal.models.tensorflow.utils import predict_batch


class OOD_VAEGMM(OODGMMBase):
    def __init__(self, model: VAEGMM, samples: int = 10) -> None:
        """
        VAE with Gaussian Mixture Model based outlier detector.

        Parameters
        ----------
        model : VAEGMM
            A VAEGMM model.
        samples
            Number of samples sampled to evaluate each instance.
        """
        super().__init__(model)
        self.samples = samples

    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float = 100.0,
        loss_fn: Callable = LossGMM(elbo=Elbo(0.05)),
        optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam,
        epochs: int = 20,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> None:
        """
        Train the AE model with recommended loss function and optimizer.

        Parameters
        ----------
        X : ArrayLike
            Training batch.
        threshold_perc : float, default 100.0
            Percentage of reference data that is normal.
        loss_fn : Callable, default LossGMM(elbo=Elbo(0.05))
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

        # draw samples from latent space
        X_samples = np.repeat(X, self.samples, axis=0)
        _, z, _ = predict_batch(X_samples, self.model, batch_size=batch_size)

        # compute average energy for samples
        energy, _ = gmm_energy(z, self.gmm_params, return_mean=False)
        energy_samples = energy.numpy().reshape((-1, self.samples))  # type: ignore
        iscore = np.mean(energy_samples, axis=-1)
        return OODScore(iscore)
