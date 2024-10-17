"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from numpy.typing import ArrayLike, NDArray

from dataeval._internal.detectors.ood.base import OODBase, OODScoreOutput
from dataeval._internal.interop import to_numpy
from dataeval._internal.models.tensorflow.pixelcnn import PixelCNN
from dataeval._internal.models.tensorflow.trainer import trainer
from dataeval._internal.models.tensorflow.utils import predict_batch
from dataeval._internal.output import set_metadata


def build_model(
    dist: PixelCNN, input_shape: tuple | None = None, filepath: str | None = None
) -> tuple[keras.Model, PixelCNN]:
    """
    Create keras.Model from TF distribution.

    Parameters
    ----------
    dist
        TensorFlow distribution.
    input_shape
        Input shape of the model.
    filepath
        File to load model weights from.

    Returns
    -------
    TensorFlow model.
    """
    x_in = Input(shape=input_shape)
    log_prob = dist.log_prob(x_in)
    model = Model(inputs=x_in, outputs=log_prob)
    model.add_loss(-tf.reduce_mean(log_prob))
    if isinstance(filepath, str):
        model.load_weights(filepath)
    return model, dist


def mutate_categorical(
    X: NDArray,
    rate: float,
    seed: int = 0,
    feature_range: tuple = (0, 255),
) -> tf.Tensor:
    """
    Randomly change integer feature values to values within a set range
    with a specified permutation rate.

    Parameters
    ----------
    X
        Batch of data to be perturbed.
    rate
        Permutation rate (between 0 and 1).
    seed
        Random seed.
    feature_range
        Min and max range for perturbed features.

    Returns
    -------
    Array with perturbed data.
    """
    frange = (feature_range[0] + 1, feature_range[1] + 1)
    shape = X.shape
    n_samples = np.prod(shape)
    mask = tf.random.categorical(tf.math.log([[1.0 - rate, rate]]), n_samples, seed=seed, dtype=tf.int32)
    mask = tf.reshape(mask, shape)
    possible_mutations = tf.random.uniform(shape, minval=frange[0], maxval=frange[1], dtype=tf.int32, seed=seed + 1)
    X = tf.math.floormod(tf.cast(X, tf.int32) + mask * possible_mutations, frange[1])  # type: ignore py38
    return tf.cast(X, tf.float32)  # type: ignore


class OOD_LLR(OODBase):
    """
    Likelihood Ratios based outlier detector.

    Parameters
    ----------
    model : PixelCNN
        Generative distribution model.
    model_background : Optional[PixelCNN], default None
        Optional model for the background. Only needed if it is different from `model`.
    log_prob : Optional[Callable], default None
        Function used to evaluate log probabilities under the model
        if the model does not have a `log_prob` function.
    sequential : bool, default False
        Whether the data is sequential. Used to create targets during training.
    """

    def __init__(
        self,
        model: PixelCNN,
        model_background: PixelCNN | None = None,
        log_prob: Callable | None = None,
        sequential: bool = False,
    ) -> None:
        self.dist_s = model
        self.dist_b = (
            model.copy()
            if hasattr(model, "copy")
            else keras.models.clone_model(model)
            if model_background is None
            else model_background
        )
        self.has_log_prob = hasattr(model, "log_prob")
        self.sequential = sequential
        self.log_prob = log_prob

        self._ref_score: OODScoreOutput
        self._threshold_perc: float
        self._data_info: tuple[tuple, type] | None = None

    def fit(
        self,
        x_ref: ArrayLike,
        threshold_perc: float = 100.0,
        loss_fn: Callable | None = None,
        optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam,
        epochs: int = 20,
        batch_size: int = 64,
        verbose: bool = True,
        mutate_fn: Callable = mutate_categorical,
        mutate_fn_kwargs: dict = {"rate": 0.2, "seed": 0, "feature_range": (0, 255)},
        mutate_batch_size: int = int(1e10),
    ) -> None:
        """
        Train semantic and background generative models.

        Parameters
        ----------
        x_ref : ArrayLike
            Training data.
        threshold_perc : float, default 100.0
            Percentage of reference data that is normal.
        loss_fn : Callable | None, default None
            Loss function used for training.
        optimizer : keras.optimizers.Optimizer, default keras.optimizers.Adam
            Optimizer used for training.
        epochs : int, default 20
            Number of training epochs.
        batch_size : int, default 64
            Batch size used for training.
        verbose : bool, default True
            Whether to print training progress.
        mutate_fn : Callable, default mutate_categorical
            Mutation function used to generate the background dataset.
        mutate_fn_kwargs : dict, default {"rate": 0.2, "seed": 0, "feature_range": (0, 255)}
            Kwargs for the mutation function used to generate the background dataset.
            Default values set for an image dataset.
        mutate_batch_size: int, default int(1e10)
            Batch size used to generate the mutations for the background dataset.
        """
        x_ref = to_numpy(x_ref)
        input_shape = x_ref.shape[1:]
        optimizer = optimizer() if isinstance(optimizer, type) else optimizer
        # Separate into two separate optimizers, one for semantic model and one for background model
        optimizer_s = optimizer
        optimizer_b = optimizer.__class__.from_config(optimizer.get_config())

        # training arguments
        kwargs = {
            "epochs": epochs,
            "batch_size": batch_size,
            "verbose": verbose,
        }

        # create background data
        mutate_fn = partial(mutate_fn, **mutate_fn_kwargs)
        X_back = predict_batch(x_ref, mutate_fn, batch_size=mutate_batch_size, dtype=x_ref.dtype)  # type: ignore

        # prepare sequential data
        if self.sequential and not self.has_log_prob:
            y, y_back = x_ref[:, 1:], X_back[:, 1:]  # type: ignore
            X, X_back = x_ref[:, :-1], X_back[:, :-1]  # type: ignore
        else:
            X = x_ref
            y, y_back = None, None

        # check if model needs to be built
        use_build = self.has_log_prob and not isinstance(self.dist_s, keras.Model)

        if use_build:
            # build and train semantic model
            self.model_s = build_model(self.dist_s, input_shape)[0]
            self.model_s.compile(optimizer=optimizer_s)
            self.model_s.fit(X, **kwargs)
            # build and train background model
            self.model_b = build_model(self.dist_b, input_shape)[0]
            self.model_b.compile(optimizer=optimizer_b)
            self.model_b.fit(X_back, **kwargs)
        else:
            # train semantic model
            args = [self.dist_s, X]
            kwargs.update({"y_train": y, "loss_fn": loss_fn, "optimizer": optimizer_s})
            trainer(*args, **kwargs)

            # train background model
            args = [self.dist_b, X_back]
            kwargs.update({"y_train": y_back, "loss_fn": loss_fn, "optimizer": optimizer_b})
            trainer(*args, **kwargs)

        self._datainfo = self._get_data_info(x_ref)
        self._ref_score = self.score(x_ref, batch_size=batch_size)
        self._threshold_perc = threshold_perc

    def _logp(
        self,
        dist,
        X: NDArray,
        return_per_feature: bool = False,
        batch_size: int = int(1e10),
    ) -> NDArray:
        """
        Compute log probability of a batch of instances under the generative model.
        """
        logp_fn = partial(dist.log_prob, return_per_feature=return_per_feature)
        # TODO: TBD: can this be any of the other types from predict_batch? i.e. tf.Tensor or tuple
        return predict_batch(X, logp_fn, batch_size=batch_size)  # type: ignore[return-value]

    def _logp_alt(
        self,
        model: keras.Model,
        X: NDArray,
        return_per_feature: bool = False,
        batch_size: int = int(1e10),
    ) -> NDArray:
        """
        Compute log probability of a batch of instances with the user defined log_prob function.
        """
        if self.sequential:
            y, X = X[:, 1:], X[:, :-1]
        else:
            y = X.copy()
        y_preds = predict_batch(X, model, batch_size=batch_size)
        logp = self.log_prob(y, y_preds).numpy()  # type: ignore
        if return_per_feature:
            return logp
        else:
            axis = tuple(np.arange(len(logp.shape))[1:])
            return np.mean(logp, axis=axis)

    def _llr(self, X: NDArray, return_per_feature: bool, batch_size: int = int(1e10)) -> NDArray:
        """
        Compute likelihood ratios.

        Parameters
        ----------
        X
            Batch of instances.
        return_per_feature
            Return likelihood ratio per feature.
        batch_size
            Batch size for the generative model evaluations.

        Returns
        -------
        Likelihood ratios.
        """
        logp_fn = self._logp if not isinstance(self.log_prob, Callable) else self._logp_alt  # type: ignore
        logp_s = logp_fn(self.dist_s, X, return_per_feature=return_per_feature, batch_size=batch_size)
        logp_b = logp_fn(self.dist_b, X, return_per_feature=return_per_feature, batch_size=batch_size)
        return logp_s - logp_b

    @set_metadata("dataeval.detectors")
    def score(
        self,
        X: ArrayLike,
        batch_size: int = int(1e10),
    ) -> OODScoreOutput:
        self._validate(X := to_numpy(X))
        fscore = -self._llr(X, True, batch_size=batch_size)
        iscore = -self._llr(X, False, batch_size=batch_size)
        return OODScoreOutput(iscore, fscore)
