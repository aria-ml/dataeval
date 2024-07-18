"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

import numpy as np
import pytest
import tensorflow as tf

from dataeval._internal.models.tensorflow.losses import Elbo, LossGMM

N, K, D, F = 10, 5, 1, 3
x = tf.convert_to_tensor(np.random.rand(N, F), dtype=tf.float32)
y = tf.convert_to_tensor(np.random.rand(N, F), dtype=tf.float32)
sim = 1.0
cov_diag = tf.ones(F, dtype=tf.float32)
cov_full = tf.eye(F, dtype=tf.float32)


def test_elbo():
    elbo_cov_full = Elbo("cov_full", x)
    elbo_cov_diag = Elbo("cov_diag", x)
    elbo_cov_sim = Elbo(sim)

    assert elbo_cov_full(x, y) != elbo_cov_diag(x, y) != elbo_cov_sim(x, y)

    elbo_cov_full.cov = ("cov_full", cov_full)
    elbo_cov_diag.cov = ("cov_diag", cov_diag)

    assert elbo_cov_full(x, y) == elbo_cov_diag(x, y) == elbo_cov_sim(x, y)

    elbo_cov_sim = Elbo(0.05)

    assert elbo_cov_sim(x, y).numpy() > 0  # type: ignore
    assert elbo_cov_sim(x, x).numpy() < 0  # type: ignore


def test_elbo_invalid_type():
    with pytest.raises(ValueError):
        Elbo("invalid")  # type: ignore


z = tf.convert_to_tensor(np.random.rand(N, D), dtype=tf.float32)
gamma = tf.convert_to_tensor(np.random.rand(N, K), dtype=tf.float32)


def test_loss_gmm_ae():
    loss = LossGMM(w_energy=0.1, w_cov_diag=0.005)(x, y, z, gamma)
    loss_no_cov = LossGMM(w_energy=0.1, w_cov_diag=0.0)(x, y, z, gamma)
    loss_xx = LossGMM(w_energy=0.1, w_cov_diag=0.0)(x, x, z, gamma)
    assert tf.greater(loss, loss_no_cov)
    assert tf.greater(loss, loss_xx)


def test_loss_gmm_vae():
    elbo = Elbo(0.05)
    loss = LossGMM(w_recon=1e-7, w_energy=0.1, w_cov_diag=0.005, elbo=elbo)(x, y, z, gamma)
    loss_no_recon = LossGMM(w_recon=0.0, w_energy=0.1, w_cov_diag=0.005, elbo=elbo)(x, y, z, gamma)
    loss_no_recon_cov = LossGMM(w_recon=0.0, w_energy=0.1, w_cov_diag=0.0)(x, y, z, gamma)
    loss_xx = LossGMM(w_recon=1e-7, w_energy=0.1, w_cov_diag=0.005)(x, x, z, gamma)
    assert tf.greater(loss, loss_no_recon)
    assert tf.greater(loss_no_recon, loss_no_recon_cov)
    assert tf.greater(loss, loss_xx)
