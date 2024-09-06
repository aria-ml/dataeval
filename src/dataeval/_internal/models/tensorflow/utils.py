"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

import math
from typing import Callable, Union, cast

import keras as keras
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    InputLayer,
    Reshape,
)
from numpy.typing import NDArray
from tensorflow._api.v2.nn import relu, softmax, tanh

from dataeval._internal.models.tensorflow.autoencoder import AE, AEGMM, VAE, VAEGMM
from dataeval._internal.models.tensorflow.pixelcnn import PixelCNN


def predict_batch(
    x: list | NDArray | tf.Tensor,
    model: Callable | keras.Model,
    batch_size: int = int(1e10),
    preprocess_fn: Callable | None = None,
    dtype: type[np.generic] | tf.DType = np.float32,
) -> NDArray | tf.Tensor | tuple | list:
    """
    Make batch predictions on a model.

    Parameters
    ----------
    x
        Batch of instances.
    model
        tf.keras model or one of the other permitted types defined in Data.
    batch_size
        Batch size used during prediction.
    preprocess_fn
        Optional preprocessing function for each batch.
    dtype
        Model output type, e.g. np.float32 or tf.float32.

    Returns
    -------
    Numpy array, tensorflow tensor or tuples of those with model outputs.
    """
    n = len(x)
    n_minibatch = int(np.ceil(n / batch_size))
    return_np = not isinstance(dtype, tf.DType)
    return_list = False
    preds: list | tuple = []
    for i in range(n_minibatch):
        istart, istop = i * batch_size, min((i + 1) * batch_size, n)
        x_batch = x[istart:istop]  # type: ignore
        if isinstance(preprocess_fn, Callable):  # type: ignore
            x_batch = preprocess_fn(x_batch)
        preds_tmp = model(x_batch)
        if isinstance(preds_tmp, (list, tuple)):
            if len(preds) == 0:  # init tuple with lists to store predictions
                preds = tuple([] for _ in range(len(preds_tmp)))
                return_list = isinstance(preds_tmp, list)
            for j, p in enumerate(preds_tmp):
                preds[j].append(p if not return_np or isinstance(p, np.ndarray) else p.numpy())
        elif isinstance(preds_tmp, (np.ndarray, tf.Tensor)):
            preds.append(  # type: ignore
                preds_tmp
                if not return_np or isinstance(preds_tmp, np.ndarray)  # type: ignore
                else preds_tmp.numpy()  # type: ignore
            )
        else:
            raise TypeError(
                f"Model output type {type(preds_tmp)} not supported. The model output "
                f"type needs to be one of list, tuple, NDArray or tf.Tensor."
            )
    concat = np.concatenate if return_np else tf.concat
    out = cast(
        Union[tuple, tf.Tensor, np.ndarray],
        tuple(concat(p, axis=0) for p in preds) if isinstance(preds, tuple) else concat(preds, axis=0),
    )
    if return_list:
        out = list(out)
    return out


def _get_default_encoder_net(input_shape: tuple[int, int, int], encoding_dim: int):
    return Sequential(
        [
            InputLayer(input_shape=input_shape),
            Conv2D(64, 4, strides=2, padding="same", activation=relu),
            Conv2D(128, 4, strides=2, padding="same", activation=relu),
            Conv2D(512, 4, strides=2, padding="same", activation=relu),
            Flatten(),
            Dense(encoding_dim),
        ]
    )


def _get_default_decoder_net(input_shape: tuple[int, int, int], encoding_dim: int):
    return Sequential(
        [
            InputLayer(input_shape=(encoding_dim,)),
            Dense(4 * 4 * 128),
            Reshape(target_shape=(4, 4, 128)),
            Conv2DTranspose(256, 4, strides=2, padding="same", activation=relu),
            Conv2DTranspose(64, 4, strides=2, padding="same", activation=relu),
            Flatten(),
            Dense(math.prod(input_shape)),
            Reshape(target_shape=input_shape),
        ]
    )


def create_model(
    model_type: AE | AEGMM | PixelCNN | VAE | VAEGMM,
    input_shape: tuple[int, int, int],
    encoding_dim: int | None = None,
    n_gmm: int | None = None,
    gmm_latent_dim: int | None = None,
):
    """
    Create a default model for the specified model type.

    Parameters
    ----------
    model_type : Union[AE, AEGMM, PixelCNN, VAE, VAEGMM]
        The model type to create.
    input_shape : Tuple[int, int, int]
        The input shape of the data used.
    encoding_dim : int, optional - default None
        The target encoding dimensionality.
    n_gmm : int, optional - default None
        Number of components used in the GMM layer.
    gmm_latent_dim : int, optional - default None
        Latent dimensionality of the GMM layer.
    """
    input_dim = math.prod(input_shape)
    encoding_dim = int(math.pow(2, int(input_dim.bit_length() * 0.8)) if encoding_dim is None else encoding_dim)
    if model_type == AE:
        return AE(
            _get_default_encoder_net(input_shape, encoding_dim),
            _get_default_decoder_net(input_shape, encoding_dim),
        )

    if model_type == VAE:
        return VAE(
            _get_default_encoder_net(input_shape, encoding_dim),
            _get_default_decoder_net(input_shape, encoding_dim),
            encoding_dim,
        )

    if model_type == AEGMM:
        n_gmm = 2 if n_gmm is None else n_gmm
        gmm_latent_dim = 1 if gmm_latent_dim is None else gmm_latent_dim
        # The outlier detector is an encoder/decoder architecture
        encoder_net = Sequential(
            [
                Flatten(),
                InputLayer(input_shape=(input_dim,)),
                Dense(60, activation=tanh),
                Dense(30, activation=tanh),
                Dense(10, activation=tanh),
                Dense(gmm_latent_dim, activation=None),
            ]
        )
        # Here we define the decoder
        decoder_net = Sequential(
            [
                InputLayer(input_shape=(gmm_latent_dim,)),
                Dense(10, activation=tanh),
                Dense(30, activation=tanh),
                Dense(60, activation=tanh),
                Dense(input_dim, activation=None),
                Reshape(target_shape=input_shape),
            ]
        )
        # GMM autoencoders have a density network too
        gmm_density_net = Sequential(
            [
                InputLayer(input_shape=(gmm_latent_dim + 2,)),
                Dense(10, activation=tanh),
                Dense(n_gmm, activation=softmax),
            ]
        )
        return AEGMM(
            encoder_net=encoder_net,
            decoder_net=decoder_net,
            gmm_density_net=gmm_density_net,
            n_gmm=n_gmm,
        )

    if model_type == VAEGMM:
        n_gmm = 2 if n_gmm is None else n_gmm
        gmm_latent_dim = 2 if gmm_latent_dim is None else gmm_latent_dim
        # The outlier detector is an encoder/decoder architecture
        # Here we define the encoder
        encoder_net = Sequential(
            [
                Flatten(),
                InputLayer(input_shape=(input_dim,)),
                Dense(20, activation=relu),
                Dense(15, activation=relu),
                Dense(7, activation=relu),
            ]
        )
        # Here we define the decoder
        decoder_net = Sequential(
            [
                InputLayer(input_shape=(gmm_latent_dim,)),
                Dense(7, activation=relu),
                Dense(15, activation=relu),
                Dense(20, activation=relu),
                Dense(input_dim, activation=None),
                Reshape(target_shape=input_shape),
            ]
        )
        # GMM autoencoders have a density network too
        gmm_density_net = Sequential(
            [
                InputLayer(input_shape=(gmm_latent_dim + 2,)),
                Dense(10, activation=relu),
                Dense(n_gmm, activation=softmax),
            ]
        )
        return VAEGMM(
            encoder_net=encoder_net,
            decoder_net=decoder_net,
            gmm_density_net=gmm_density_net,
            n_gmm=n_gmm,
            latent_dim=gmm_latent_dim,
        )

    if model_type == PixelCNN:
        return PixelCNN(
            image_shape=input_shape,
            num_resnet=5,
            num_hierarchies=2,
            num_filters=32,
            num_logistic_mix=1,
            receptive_field_dims=(3, 3),
            dropout_p=0.3,
            l2_weight=0.0,
        )

    raise TypeError(f"Unknown model specified: {model_type}.")
