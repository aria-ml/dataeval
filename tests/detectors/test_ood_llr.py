"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from itertools import product

import numpy as np
import pytest
import tensorflow as tf
from tf_keras import Model
from tf_keras.layers import LSTM, Dense, Input

from dataeval.detectors.ood.llr import OOD_LLR

input_dim = 5
hidden_dim = 20

shape = (1000, 6)
X_train = np.zeros(shape, dtype=np.int32)
X_train[:, ::2] = 1
X_test = np.zeros(shape, dtype=np.int32)
X_test[:, ::2] = 2
X_val = np.concatenate([X_train[:50], X_test[:50]])


def loss_fn(y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    y = tf.one_hot(tf.cast(y, tf.int32), input_dim)
    return tf.nn.softmax_cross_entropy_with_logits(y, x, axis=-1)  # type: ignore


def likelihood_fn(y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    return -loss_fn(y, x)  # type: ignore


threshold_perc = [50.0]
ood_type = ["instance", "feature"]
tests = list(product(threshold_perc, ood_type))
n_tests = len(tests)


@pytest.fixture
def llr_params(request):
    return tests[request.param]


@pytest.mark.parametrize("llr_params", list(range(n_tests)), indirect=True)
def test_llr(llr_params):
    # LLR parameters
    threshold_perc, ood_type = llr_params

    # define model and detector
    inputs = Input(shape=(shape[-1] - 1,), dtype=tf.int32)
    x = tf.one_hot(tf.cast(inputs, tf.int32), input_dim)
    x = LSTM(hidden_dim, return_sequences=True)(x)
    logits = Dense(input_dim, activation=None)(x)
    model = Model(inputs=inputs, outputs=logits)

    od = OOD_LLR(sequential=True, model=model, log_prob=likelihood_fn)

    od.fit(
        X_train,
        threshold_perc=threshold_perc,
        loss_fn=loss_fn,
        mutate_fn_kwargs={"rate": 0.5, "feature_range": (0, input_dim)},
        epochs=1,
        verbose=False,
    )

    od_preds = od.predict(
        X_test,
        ood_type=ood_type,
    )

    scores = od._threshold_score(ood_type)
    if ood_type == "instance":
        assert od_preds.is_ood.shape == (X_test.shape[0],)
        assert od_preds.is_ood.sum() == (od_preds.instance_score > scores).sum()
    elif ood_type == "feature":
        assert od_preds.is_ood.shape == (X_test.shape[0], X_test.shape[1] - 1)
        assert od_preds.feature_score is not None
        assert od_preds.feature_score.shape == (X_test.shape[0], X_test.shape[1] - 1)
        assert od_preds.is_ood.sum() == (od_preds.feature_score > scores).sum()

    assert od_preds.instance_score.shape == (X_test.shape[0],)
