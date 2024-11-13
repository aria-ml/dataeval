"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

__all__ = ["OOD_VAE"]

from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import ArrayLike

from dataeval.detectors.ood.base import OODBase, OODScoreOutput
from dataeval.interop import to_numpy
from dataeval.utils.lazy import lazyload
from dataeval.utils.tensorflow._internal.loss import Elbo
from dataeval.utils.tensorflow._internal.utils import predict_batch

if TYPE_CHECKING:
    import tensorflow as tf
    import tf_keras as keras

    import dataeval.utils.tensorflow._internal.models as tf_models
else:
    tf = lazyload("tensorflow")
    keras = lazyload("tf_keras")
    tf_models = lazyload("dataeval.utils.tensorflow._internal.models")


class OOD_VAE(OODBase):
    """
    VAE based outlier detector.

    Parameters
    ----------
    model : VAE
        A VAE model.
    samples : int, default 10
        Number of samples sampled to evaluate each instance.

    Examples
    --------
    Instantiate an OOD detector metric with a generic dataset - batch of images with shape (3,25,25)

    >>> metric = OOD_VAE(create_model("VAE", dataset[0].shape))

    Adjusting fit parameters,
    including setting the fit threshold at 85% for a training set with about 15% out-of-distribution

    >>> metric.fit(dataset, threshold_perc=85, batch_size=128, verbose=False)

    Detect :term:`out of distribution<Out-of-Distribution (OOD)>` samples at the 'feature' level

    >>> result = metric.predict(dataset, ood_type="feature")
    """

    def __init__(self, model: tf_models.VAE, samples: int = 10) -> None:
        super().__init__(model)
        self.samples = samples

    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float = 100.0,
        loss_fn: Callable[..., tf.Tensor] = Elbo(0.05),
        optimizer: keras.optimizers.Optimizer | None = None,
        epochs: int = 20,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> None:
        super().fit(x_ref, threshold_perc, loss_fn, optimizer, epochs, batch_size, verbose)

    def _score(self, X: ArrayLike, batch_size: int = int(1e10)) -> OODScoreOutput:
        self._validate(X := to_numpy(X))

        # sample reconstructed instances
        X_samples = np.repeat(X, self.samples, axis=0)
        X_recon = predict_batch(X_samples, model=self.model, batch_size=batch_size)

        # compute feature scores
        fscore = np.power(X_samples - X_recon, 2)
        fscore = fscore.reshape((-1, self.samples) + X_samples.shape[1:])
        fscore = np.mean(fscore, axis=1)

        # compute instance scores
        fscore_flat = fscore.reshape(fscore.shape[0], -1).copy()
        n_score_features = int(np.ceil(fscore_flat.shape[1]))
        sorted_fscore = np.sort(fscore_flat, axis=1)
        sorted_fscore_perc = sorted_fscore[:, -n_score_features:]
        iscore = np.mean(sorted_fscore_perc, axis=1)

        return OODScoreOutput(iscore, fscore)
