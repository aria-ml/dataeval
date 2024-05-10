"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from itertools import product

import keras.api._v2.keras as keras
import numpy as np
import pytest
import tensorflow as tf
from keras.api._v2.keras.layers import Dense, InputLayer

from daml._internal.metrics.outlier.vaegmm import VAEGMMOutlier
from daml._internal.models.tensorflow.autoencoder import VAEGMM
from daml._internal.models.tensorflow.losses import LossGMM

n_gmm = [1, 2]
w_energy = [0.1, 0.5]
w_recon = [0.0, 1e-7]
samples = [1, 10]
threshold_perc = [90.0]

tests = list(product(n_gmm, w_energy, w_recon, samples, threshold_perc))
n_tests = len(tests)

# load and preprocess MNIST data
(X_train, _), (X_test, _) = keras.datasets.mnist.load_data()
X = X_train.reshape(X_train.shape[0], -1)[:1000]  # only train on 1000 instances
X = X.astype(np.float32)
X /= 255

input_dim = X.shape[1]
latent_dim = 2


@pytest.fixture
def vaegmm_params(request):
    return tests[request.param]


@pytest.mark.parametrize("vaegmm_params", list(range(n_tests)), indirect=True)
def test_vaegmm(vaegmm_params):
    # OutlierVAEGMM parameters
    n_gmm, w_energy, w_recon, samples, threshold_perc = vaegmm_params

    # define encoder, decoder and GMM density net
    encoder_net = keras.Sequential(
        [InputLayer(input_shape=(input_dim,)), Dense(128, activation=tf.nn.relu), Dense(latent_dim, activation=None)]
    )

    decoder_net = keras.Sequential(
        [
            InputLayer(input_shape=(latent_dim,)),
            Dense(128, activation=tf.nn.relu),
            Dense(input_dim, activation=tf.nn.sigmoid),
        ]
    )

    gmm_density_net = keras.Sequential(
        [
            InputLayer(input_shape=(latent_dim + 2,)),
            Dense(10, activation=tf.nn.relu),
            Dense(n_gmm, activation=tf.nn.softmax),
        ]
    )

    # init OutlierAEGMM
    vaegmm = VAEGMMOutlier(VAEGMM(encoder_net, decoder_net, gmm_density_net, n_gmm, latent_dim), samples=samples)

    # fit OutlierAEGMM, infer threshold and compute scores
    loss_fn = LossGMM(w_recon=w_recon, w_energy=w_energy)
    vaegmm.fit(X, threshold_perc=threshold_perc, loss_fn=loss_fn, epochs=5, batch_size=1000, verbose=False)
    energy = vaegmm.score(X).instance_score
    perc_score = 100 * (energy < vaegmm._threshold_score()).astype(int).sum() / energy.shape[0]
    assert threshold_perc + 5 > perc_score > threshold_perc - 5

    # make and check predictions
    od_preds = vaegmm.predict(X)
    assert od_preds["is_outlier"].shape == (X.shape[0],)
    assert od_preds["is_outlier"].sum() == (od_preds["instance_score"] > vaegmm._threshold_score()).astype(int).sum()
