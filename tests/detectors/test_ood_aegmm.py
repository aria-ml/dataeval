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

from dataeval._internal.detectors.ood.aegmm import OOD_AEGMM
from dataeval._internal.models.tensorflow.autoencoder import AEGMM
from dataeval._internal.models.tensorflow.losses import LossGMM
from tests.utils.datasets import wait_lock

n_gmm = [1, 2]
w_energy = [0.1, 0.5]
threshold_perc = [90.0]
loss_fn = [True]

tests = list(product(n_gmm, w_energy, threshold_perc, loss_fn))
tests.append((n_gmm[0], w_energy[0], threshold_perc[0], False))
n_tests = len(tests)

# load and preprocess MNIST data
with wait_lock("mnist"):
    (X_train, _), (X_test, _) = keras.datasets.mnist.load_data()
X = X_train.reshape(X_train.shape[0], -1)[:1000]  # only train on 1000 instances
X = X.astype(np.float32)
X /= 255

input_dim = X.shape[1]
latent_dim = 2


@pytest.fixture
def aegmm_params(request):
    return tests[request.param]


@pytest.mark.parametrize("aegmm_params", list(range(n_tests)), indirect=True)
def test_aegmm(aegmm_params):
    # OutlierAEGMM parameters
    n_gmm, w_energy, threshold_perc, loss_fn = aegmm_params

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
    aegmm = OOD_AEGMM(AEGMM(encoder_net, decoder_net, gmm_density_net, n_gmm))

    # fit OutlierAEGMM, infer threshold and compute scores
    if loss_fn:
        loss_fn = LossGMM(w_energy=w_energy)
        aegmm.fit(X, threshold_perc=threshold_perc, loss_fn=loss_fn, epochs=5, batch_size=1000, verbose=False)
    else:
        aegmm.fit(X, threshold_perc=threshold_perc, epochs=5, batch_size=1000, verbose=False)
    energy = aegmm.score(X).instance_score
    perc_score = 100 * (energy < aegmm._threshold_score()).sum() / energy.shape[0]
    assert threshold_perc + 5 > perc_score > threshold_perc - 5

    # make and check predictions
    od_preds = aegmm.predict(X)
    assert od_preds.is_ood.shape == (X.shape[0],)
    assert od_preds.is_ood.sum() == (od_preds.instance_score > aegmm._threshold_score()).sum()
