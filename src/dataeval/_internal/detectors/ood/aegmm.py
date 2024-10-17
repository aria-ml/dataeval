"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

from typing import Callable

import keras
import tensorflow as tf
from numpy.typing import ArrayLike

from dataeval._internal.detectors.ood.base import OODGMMBase, OODScoreOutput
from dataeval._internal.interop import to_numpy
from dataeval._internal.models.tensorflow.autoencoder import AEGMM
from dataeval._internal.models.tensorflow.gmm import gmm_energy
from dataeval._internal.models.tensorflow.losses import LossGMM
from dataeval._internal.models.tensorflow.utils import predict_batch
from dataeval._internal.output import set_metadata


class OOD_AEGMM(OODGMMBase):
    """
    AE with Gaussian Mixture Model based outlier detector.

    Parameters
    ----------
    model : AEGMM
        An AEGMM model.
    """

    def __init__(self, model: AEGMM) -> None:
        super().__init__(model)

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
            loss_fn = LossGMM()
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
        _, z, _ = predict_batch(X, self.model, batch_size=batch_size)
        energy, _ = gmm_energy(z, self.gmm_params, return_mean=False)
        return OODScoreOutput(energy.numpy())  # type: ignore
