"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

from typing import Callable

import keras
import numpy as np
import tensorflow as tf
from numpy.typing import ArrayLike

from dataeval._internal.detectors.ood.base import OODGMMBase, OODScoreOutput
from dataeval._internal.interop import to_numpy
from dataeval._internal.models.tensorflow.autoencoder import VAEGMM
from dataeval._internal.models.tensorflow.gmm import gmm_energy
from dataeval._internal.models.tensorflow.losses import Elbo, LossGMM
from dataeval._internal.models.tensorflow.utils import predict_batch
from dataeval._internal.output import set_metadata


class OOD_VAEGMM(OODGMMBase):
    """
    VAE with Gaussian Mixture Model based outlier detector.

    Parameters
    ----------
    model : VAEGMM
        A VAEGMM model.
    samples
        Number of samples sampled to evaluate each instance.
    """

    def __init__(self, model: VAEGMM, samples: int = 10) -> None:
        super().__init__(model)
        self.samples = samples

    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float = 100.0,
        loss_fn: Callable[..., tf.Tensor] | None = None,
        optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam,
        epochs: int = 20,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> None:
        if loss_fn is None:
            loss_fn = LossGMM(elbo=Elbo(0.05))
        super().fit(x_ref, threshold_perc, loss_fn, optimizer, epochs, batch_size, verbose)

    @set_metadata("dataeval.detectors")
    def score(self, X: ArrayLike, batch_size: int = int(1e10)) -> OODScoreOutput:
        """
        Compute the out-of-distribution (OOD) score for a given dataset.

        Parameters
        ----------
        X : ArrayLike
            Input data to score.
        batch_size : int, default 1e10
            Number of instances to process in each batch.
            Use a smaller batch size if your dataset is large or if you encounter memory issues.

        Returns
        -------
        OODScoreOutput
            An object containing the instance-level OOD score.

        Note
        ----
        This model does not produce a feature level score like the OOD_AE or OOD_VAE models.
        """
        self._validate(X := to_numpy(X))

        # draw samples from latent space
        X_samples = np.repeat(X, self.samples, axis=0)
        _, z, _ = predict_batch(X_samples, self.model, batch_size=batch_size)

        # compute average energy for samples
        energy, _ = gmm_energy(z, self.gmm_params, return_mean=False)
        energy_samples = energy.numpy().reshape((-1, self.samples))  # type: ignore
        iscore = np.mean(energy_samples, axis=-1)
        return OODScoreOutput(iscore)
