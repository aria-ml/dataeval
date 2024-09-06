"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

from typing import Callable, Iterable, cast

import keras
import numpy as np
import tensorflow as tf
from numpy.typing import NDArray


def trainer(
    model: keras.Model,
    x_train: NDArray,
    y_train: NDArray | None = None,
    loss_fn: Callable[..., tf.Tensor] | None = None,
    optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam,
    preprocess_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
    epochs: int = 20,
    reg_loss_fn: Callable[[keras.Model], tf.Tensor] = (lambda _: cast(tf.Tensor, tf.Variable(0, dtype=tf.float32))),
    batch_size: int = 64,
    buffer_size: int = 1024,
    verbose: bool = True,
) -> None:
    """
    Train TensorFlow model.

    Parameters
    ----------
    model
        Model to train.
    loss_fn
        Loss function used for training.
    x_train
        Training data.
    y_train
        Training labels.
    optimizer
        Optimizer used for training.
    preprocess_fn
        Preprocessing function applied to each training batch.
    epochs
        Number of training epochs.
    reg_loss_fn
        Allows an additional regularisation term to be defined as reg_loss_fn(model)
    batch_size
        Batch size used for training.
    buffer_size
        Maximum number of elements that will be buffered when prefetching.
    verbose
        Whether to print training progress.
    """
    loss_fn = loss_fn() if isinstance(loss_fn, type) else loss_fn
    optimizer = optimizer() if isinstance(optimizer, type) else optimizer

    train_data = x_train if y_train is None else (x_train, y_train)
    dataset = tf.data.Dataset.from_tensor_slices(train_data)
    dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size)
    n_minibatch = len(dataset)

    # iterate over epochs
    for epoch in range(epochs):
        pbar = keras.utils.Progbar(n_minibatch, 1) if verbose else None
        if hasattr(dataset, "on_epoch_end"):
            dataset.on_epoch_end()  # type: ignore py39
        loss_val_ma = 0.0
        for step, data in enumerate(dataset):
            x, y = data if isinstance(data, tuple) else (data, None)
            if isinstance(preprocess_fn, Callable):
                x = preprocess_fn(x)
            with tf.GradientTape() as tape:
                y_hat = model(x)
                y = x if y is None else y
                if isinstance(loss_fn, Callable):
                    args = [y] + list(y_hat) if isinstance(y_hat, tuple) else [y, y_hat]
                    loss = loss_fn(*args)
                else:
                    loss = cast(tf.Tensor, tf.constant(0.0, dtype=tf.float32))
                if model.losses:  # additional model losses
                    loss = cast(tf.Tensor, tf.add(sum(model.losses), loss))
                loss = cast(tf.Tensor, tf.add(reg_loss_fn(model), loss))  # alternative way they might be specified

            grads = cast(Iterable, tape.gradient(loss, model.trainable_weights))
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            if pbar is not None:
                loss_val = getattr(loss, "numpy")() if hasattr(loss, "numpy") else np.float32(0.0)
                if loss_val.shape and loss_val.shape[0] != batch_size:
                    if len(loss_val.shape) == 1:
                        shape = (batch_size - loss_val.shape[0],)
                    elif len(loss_val.shape) == 2:
                        shape = (batch_size - loss_val.shape[0], loss_val.shape[1])
                    else:
                        continue
                    add_mean = np.ones(shape) * loss_val.mean()
                    loss_val = np.r_[loss_val, add_mean]
                loss_val_ma = loss_val_ma + (loss_val - loss_val_ma) / (step + 1)
                pbar_values = [("loss_ma", loss_val_ma)]
                pbar.add(1, values=pbar_values)
