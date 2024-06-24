"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from itertools import product
from typing import cast

import keras.api._v2.keras as keras
import numpy as np
import pytest
import tensorflow as tf
from keras.api._v2.keras.layers import Dense, InputLayer
from sklearn.datasets import load_iris

from daml._internal.detectors.ood.ae import OOD_AE
from daml._internal.models.tensorflow.autoencoder import AE

threshold_perc = [90.0]
ood_perc = [50, 100]
ood_type = ["instance", "feature"]

tests = list(product(threshold_perc, ood_perc, ood_type))
n_tests = len(tests)

# load iris data
X, y = load_iris(return_X_y=True)
X = cast(np.ndarray, X.astype(np.float32))

input_dim = X.shape[1]
encoding_dim = 1


@pytest.fixture
def ae_params(request):
    return tests[request.param]


@pytest.mark.parametrize("ae_params", list(range(n_tests)), indirect=True)
def test_ae(ae_params):
    # OutlierAE parameters
    threshold_perc, ood_perc, ood_type = ae_params

    # define encoder and decoder
    encoder_net = keras.Sequential(
        [InputLayer(input_shape=(input_dim,)), Dense(5, activation=tf.nn.relu), Dense(encoding_dim, activation=None)]
    )

    decoder_net = keras.Sequential(
        [
            InputLayer(input_shape=(encoding_dim,)),
            Dense(5, activation=tf.nn.relu),
            Dense(input_dim, activation=tf.nn.sigmoid),
        ]
    )

    # init OutlierAE
    ae = OOD_AE(AE(encoder_net=encoder_net, decoder_net=decoder_net))

    # fit OutlierAE
    ae.fit(X, threshold_perc=threshold_perc, epochs=5, verbose=True)  # type: ignore

    # check thresholds and scores
    iscore = ae._ref_score.instance_score
    perc_score = 100 * (iscore < ae._threshold_score()).astype(int).sum() / iscore.shape[0]
    assert threshold_perc + 5 > perc_score > threshold_perc - 5

    # make and check predictions
    od_preds = ae.predict(X, ood_type=ood_type)  # type: ignore
    scores = ae._threshold_score(ood_type)

    if ood_type == "instance":
        assert od_preds["is_ood"].shape == (X.shape[0],)
        assert od_preds["is_ood"].sum() == (od_preds["instance_score"] > scores).astype(int).sum()
    elif ood_type == "feature":
        assert od_preds["is_ood"].shape == X.shape
        assert od_preds["is_ood"].sum() == (od_preds["feature_score"] > scores).astype(int).sum()

    assert od_preds["feature_score"].shape == X.shape
    assert od_preds["instance_score"].shape == (X.shape[0],)
