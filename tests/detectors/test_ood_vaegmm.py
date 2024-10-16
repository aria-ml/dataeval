"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from itertools import product

import keras as keras
import numpy as np
import pytest
import tensorflow as tf
from keras.layers import Dense, InputLayer

from dataeval._internal.detectors.ood.vaegmm import OOD_VAEGMM
from dataeval._internal.models.tensorflow.autoencoder import VAEGMM
from dataeval._internal.models.tensorflow.losses import LossGMM

n_gmm = [1, 2]
w_energy = [0.1, 0.5]
w_recon = [0.0, 1e-7]
samples = [1, 10]
threshold_perc = [90.0]
loss_fn = [True]

tests = list(product(n_gmm, w_energy, w_recon, samples, threshold_perc, loss_fn))
tests.append((n_gmm[0], w_energy[0], w_recon[0], samples[0], threshold_perc[0], False))
n_tests = len(tests)


@pytest.fixture
def vaegmm_params(request):
    return tests[request.param]


@pytest.mark.parametrize("vaegmm_params", list(range(n_tests)), indirect=True)
def test_vaegmm(vaegmm_params):
    # OutlierVAEGMM parameters
    n_gmm, w_energy, w_recon, samples, threshold_perc, loss_fn = vaegmm_params

    # load and preprocess random data
    rng = np.random.default_rng(3)
    X_train = rng.random((1000, 28 * 28)).astype(np.float32)
    input_dim = X_train.shape[1]
    latent_dim = 2

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

    # init OutlierVAEGMM
    vaegmm = OOD_VAEGMM(VAEGMM(encoder_net, decoder_net, gmm_density_net, n_gmm, latent_dim), samples=samples)

    # fit OutlierVAEGMM, infer threshold and compute scores
    if loss_fn:
        loss_fn = LossGMM(w_recon=w_recon, w_energy=w_energy)
        vaegmm.fit(X_train, threshold_perc=threshold_perc, loss_fn=loss_fn, epochs=5, batch_size=1000, verbose=False)
    else:
        vaegmm.fit(X_train, threshold_perc=threshold_perc, epochs=5, batch_size=1000, verbose=False)
    energy = vaegmm.score(X_train).instance_score
    perc_score = 100 * (energy < vaegmm._threshold_score()).sum() / energy.shape[0]
    assert threshold_perc + 5 > perc_score > threshold_perc - 5

    # make and check predictions
    od_preds = vaegmm.predict(X_train)
    assert od_preds.is_ood.shape == (X_train.shape[0],)
    assert od_preds.is_ood.sum() == (od_preds.instance_score > vaegmm._threshold_score()).sum()
