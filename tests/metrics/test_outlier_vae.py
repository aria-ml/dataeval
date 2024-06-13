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

from daml._internal.metrics.outlier.vae import VAEOutlier
from daml._internal.models.tensorflow.autoencoder import VAE
from daml._internal.models.tensorflow.losses import Elbo

score_type = ["mse"]
samples = [10]
loss_fn = [Elbo, keras.losses.mse]
threshold_perc = [90.0]
outlier_perc = [50, 100]
outlier_type = ["instance", "feature"]

tests = list(
    product(
        samples,
        loss_fn,
        threshold_perc,
        outlier_perc,
        outlier_type,
    )
)
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
    (
        samples,
        loss_fn,
        threshold_perc,
        outlier_perc,
        outlier_type,
    ) = vae_params

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
    vae = VAEOutlier(VAE(encoder_net, decoder_net, latent_dim), samples=samples)

    # fit OutlierVAE, infer threshold and compute scores
    vae.fit(X, threshold_perc=threshold_perc, loss_fn=loss_fn, epochs=5, verbose=False)  # type: ignore
    iscore = vae.score(X).instance_score  # type: ignore
    perc_score = 100 * (iscore < vae._threshold_score()).astype(int).sum() / iscore.shape[0]
    assert threshold_perc + 5 > perc_score > threshold_perc - 5

    # make and check predictions
    od_preds = vae.predict(X, outlier_type=outlier_type)  # type: ignore
    scores = vae._threshold_score(outlier_type)

    if outlier_type == "instance":
        assert od_preds["is_outlier"].shape == (X.shape[0],)
        assert od_preds["is_outlier"].sum() == (od_preds["instance_score"] > scores).astype(int).sum()
    elif outlier_type == "feature":
        assert od_preds["is_outlier"].shape == X.shape
        assert od_preds["is_outlier"].sum() == (od_preds["feature_score"] > scores).astype(int).sum()

    assert od_preds["feature_score"].shape == X.shape
    assert od_preds["instance_score"].shape == (X.shape[0],)
