"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, cast

from numpy.typing import ArrayLike

from dataeval.detectors.ood.base import OODBaseMixin, OODFitMixin, OODGMMMixin
from dataeval.interop import to_numpy
from dataeval.utils.lazy import lazyload
from dataeval.utils.tensorflow._internal.gmm import gmm_params
from dataeval.utils.tensorflow._internal.trainer import trainer

if TYPE_CHECKING:
    import tensorflow as tf
    import tf_keras as keras
else:
    tf = lazyload("tensorflow")
    keras = lazyload("tf_keras")


class OODBase(OODBaseMixin[keras.Model], OODFitMixin[Callable[..., tf.Tensor], keras.optimizers.Optimizer]):
    def __init__(self, model: keras.Model) -> None:
        super().__init__(model)

    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float,
        loss_fn: Callable[..., tf.Tensor] | None,
        optimizer: keras.optimizers.Optimizer | None,
        epochs: int,
        batch_size: int,
        verbose: bool,
    ) -> None:
        """
        Train the model and infer the threshold value.

        Parameters
        ----------
        x_ref : ArrayLike
            Training data.
        threshold_perc : float, default 100.0
            Percentage of reference data that is normal.
        loss_fn : Callable | None, default None
            Loss function used for training.
        optimizer : Optimizer, default keras.optimizers.Adam
            Optimizer used for training.
        epochs : int, default 20
            Number of training epochs.
        batch_size : int, default 64
            Batch size used for training.
        verbose : bool, default True
            Whether to print training progress.
        """

        # Train the model
        trainer(
            model=self.model,
            loss_fn=loss_fn,
            x_train=to_numpy(x_ref),
            y_train=None,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

        # Infer the threshold values
        self._ref_score = self.score(x_ref, batch_size)
        self._threshold_perc = threshold_perc


class OODBaseGMM(OODBase, OODGMMMixin[tf.Tensor]):
    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float,
        loss_fn: Callable[..., tf.Tensor] | None,
        optimizer: keras.optimizers.Optimizer | None,
        epochs: int,
        batch_size: int,
        verbose: bool,
    ) -> None:
        # Train the model
        trainer(
            model=self.model,
            loss_fn=loss_fn,
            x_train=to_numpy(x_ref),
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

        # Calculate the GMM parameters
        _, z, gamma = cast(tuple[tf.Tensor, tf.Tensor, tf.Tensor], self.model(x_ref))
        self._gmm_params = gmm_params(z, gamma)

        # Infer the threshold values
        self._ref_score = self.score(x_ref, batch_size)
        self._threshold_perc = threshold_perc
