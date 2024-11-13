"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

__all__ = ["OOD_VAEGMM"]

from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import ArrayLike

from dataeval.detectors.ood.base import OODGMMBase, OODScoreOutput
from dataeval.interop import to_numpy
from dataeval.utils.lazy import lazyload
from dataeval.utils.tensorflow._internal.gmm import gmm_energy
from dataeval.utils.tensorflow._internal.loss import Elbo, LossGMM
from dataeval.utils.tensorflow._internal.utils import predict_batch

if TYPE_CHECKING:
    import tensorflow as tf
    import tf_keras as keras

    import dataeval.utils.tensorflow._internal.models as tf_models
else:
    tf = lazyload("tensorflow")
    keras = lazyload("tf_keras")
    tf_models = lazyload("dataeval.utils.tensorflow._internal.models")


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

    def __init__(self, model: tf_models.VAEGMM, samples: int = 10) -> None:
        super().__init__(model)
        self.samples = samples

    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float = 100.0,
        loss_fn: Callable[..., tf.Tensor] = LossGMM(elbo=Elbo(0.05)),
        optimizer: keras.optimizers.Optimizer | None = None,
        epochs: int = 20,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> None:
        super().fit(x_ref, threshold_perc, loss_fn, optimizer, epochs, batch_size, verbose)

    def _score(self, X: ArrayLike, batch_size: int = int(1e10)) -> OODScoreOutput:
        self._validate(X := to_numpy(X))

        # draw samples from latent space
        X_samples = np.repeat(X, self.samples, axis=0)
        _, z, _ = predict_batch(X_samples, self.model, batch_size=batch_size)

        # compute average energy for samples
        energy, _ = gmm_energy(z, self.gmm_params, return_mean=False)
        energy_samples = energy.numpy().reshape((-1, self.samples))  # type: ignore
        iscore = np.mean(energy_samples, axis=-1)
        return OODScoreOutput(iscore)
