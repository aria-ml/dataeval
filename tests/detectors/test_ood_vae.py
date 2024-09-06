"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from itertools import product
from typing import cast

import keras as keras
import numpy as np
import pytest
import tensorflow as tf
from keras.layers import Dense, InputLayer
from sklearn.datasets import load_iris

from dataeval._internal.detectors.ood.vae import OOD_VAE
from dataeval._internal.models.tensorflow.autoencoder import VAE
from dataeval._internal.models.tensorflow.losses import Elbo

score_type = ["mse"]
samples = [10]
loss_fn = [Elbo, keras.losses.mse, None]
threshold_perc = [90.0]
ood_type = ["instance", "feature"]

tests = list(
    product(
        samples,
        loss_fn,
        threshold_perc,
        ood_type,
    )
)[:-1]
n_tests = len(tests)

# load iris data
X, y = load_iris(return_X_y=True)
X = cast(np.ndarray, X.astype(np.float32))

input_dim = X.shape[1]
latent_dim = 2


@pytest.fixture
def vae_params(request):
    return tests[request.param]


@pytest.mark.parametrize("vae_params", list(range(n_tests)), indirect=True)
def test_vae(vae_params):
    # OutlierVAE parameters
    samples, loss_fn, threshold_perc, ood_type = vae_params

    # define encoder and decoder
    encoder_net = keras.Sequential(
        [InputLayer(input_shape=(input_dim,)), Dense(5, activation=tf.nn.relu), Dense(latent_dim, activation=None)]
    )

    decoder_net = keras.Sequential(
        [
            InputLayer(input_shape=(latent_dim,)),
            Dense(5, activation=tf.nn.relu),
            Dense(input_dim, activation=tf.nn.sigmoid),
        ]
    )

    # init OutlierVAE
    vae = OOD_VAE(VAE(encoder_net, decoder_net, latent_dim), samples=samples)

    # fit OutlierVAE, infer threshold and compute scores
    vae.fit(X, threshold_perc=threshold_perc, loss_fn=loss_fn, epochs=5, verbose=False)  # type: ignore
    iscore = vae.score(X).instance_score  # type: ignore
    perc_score = 100 * (iscore < vae._threshold_score()).sum() / iscore.shape[0]
    assert threshold_perc + 5 > perc_score > threshold_perc - 5

    # make and check predictions
    od_preds = vae.predict(X, ood_type=ood_type)  # type: ignore
    scores = vae._threshold_score(ood_type)

    if ood_type == "instance":
        assert od_preds.is_ood.shape == (X.shape[0],)
        assert od_preds.is_ood.sum() == (od_preds.instance_score > scores).sum()
    elif ood_type == "feature":
        assert od_preds.is_ood.shape == X.shape
        assert od_preds.feature_score is not None
        assert od_preds.feature_score.shape == X.shape
        assert od_preds.is_ood.sum() == (od_preds.feature_score > scores).sum()

    assert od_preds.instance_score.shape == (X.shape[0],)
