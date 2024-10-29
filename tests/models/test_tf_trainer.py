"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

from functools import partial
from itertools import product
from typing import Union

import numpy as np
import pytest
import tensorflow as tf
import tf_keras as keras

from dataeval._internal.models.tensorflow.trainer import trainer

Indexable = Union[np.ndarray, tf.Tensor, list]


N, F = 100, 2
x = np.random.rand(N, F).astype(np.float32)
y = np.concatenate([np.zeros((N, 1)), np.ones((N, 1))], axis=1).astype(np.float32)

inputs = keras.Input(shape=(x.shape[1],))
outputs = keras.layers.Dense(F, activation=tf.nn.softmax)(inputs)
model = keras.Model(inputs=inputs, outputs=outputs)
check_model_weights = model.weights[0].numpy()


def preprocess_fn(x: np.ndarray) -> np.ndarray:
    return x


X_train = [x]
y_train = [None, y]
loss_fn_kwargs = [{}, {"from_logits": False}]
preprocess = [lambda x: x, None]
verbose = [False, True]

tests = list(product(X_train, y_train, loss_fn_kwargs, preprocess, verbose))
n_tests = len(tests)


@pytest.fixture
def trainer_params(request):
    x_train, y_train, loss_fn_kwargs, preprocess, verbose = tests[request.param]
    return x_train, y_train, loss_fn_kwargs, preprocess, verbose


@pytest.mark.parametrize("trainer_params", list(range(n_tests)), indirect=True)
def test_trainer(trainer_params):
    x_train, y_train, loss_fn_kwargs, preprocess, verbose = trainer_params
    trainer(
        model,
        x_train,
        y_train=y_train,
        loss_fn=partial(keras.losses.categorical_crossentropy, **loss_fn_kwargs),
        preprocess_fn=preprocess,
        epochs=2,
        verbose=verbose,
    )
    assert (model.weights[0].numpy() != check_model_weights).any()
