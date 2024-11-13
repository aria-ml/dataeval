"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

__all__ = ["OOD_AE"]

from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import ArrayLike

from dataeval.detectors.ood.base import OODBase, OODScoreOutput
from dataeval.interop import as_numpy
from dataeval.utils.lazy import lazyload
from dataeval.utils.tensorflow._internal.utils import predict_batch

if TYPE_CHECKING:
    import tensorflow as tf
    import tf_keras as keras

    import dataeval.utils.tensorflow._internal.models as tf_models
else:
    tf = lazyload("tensorflow")
    keras = lazyload("tf_keras")
    tf_models = lazyload("dataeval.utils.tensorflow._internal.models")


class OOD_AE(OODBase):
    """
    Autoencoder-based :term:`out of distribution<Out-of-distribution (OOD)>` detector.

    Parameters
    ----------
    model : AE
       An :term:`autoencoder<Autoencoder>` model.
    """

    def __init__(self, model: tf_models.AE) -> None:
        super().__init__(model)

    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float = 100.0,
        loss_fn: Callable[..., tf.Tensor] | None = None,
        optimizer: keras.optimizers.Optimizer | None = None,
        epochs: int = 20,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> None:
        if loss_fn is None:
            loss_fn = keras.losses.MeanSquaredError()
        super().fit(as_numpy(x_ref), threshold_perc, loss_fn, optimizer, epochs, batch_size, verbose)

    def _score(self, X: ArrayLike, batch_size: int = int(1e10)) -> OODScoreOutput:
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
