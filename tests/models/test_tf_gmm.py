"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

import numpy as np
import tensorflow as tf

from daml._internal.models.tensorflow.gmm import gmm_energy, gmm_params

N, K, D = 10, 5, 1
z = tf.convert_to_tensor(np.random.rand(N, D).astype(np.float32))
gamma = tf.convert_to_tensor(np.random.rand(N, K).astype(np.float32))


def test_gmm_params_energy():
    params = gmm_params(z, gamma)
    phi, mu, cov, L, log_det_cov = params
    assert phi.numpy().shape[0] == K == log_det_cov.shape[0]  # type: ignore
    assert mu.numpy().shape == (K, D)  # type: ignore
    assert cov.numpy().shape == L.numpy().shape == (K, D, D)  # type: ignore
    for _ in range(cov.numpy().shape[0]):  # type: ignore
        assert (np.diag(cov[_].numpy()) >= 0.0).all()  # type: ignore
        assert (np.diag(L[_].numpy()) >= 0.0).all()  # type: ignore

    sample_energy, cov_diag = gmm_energy(z, params, return_mean=True)
    assert sample_energy.numpy().shape == cov_diag.numpy().shape == ()  # type: ignore

    sample_energy, _ = gmm_energy(z, params, return_mean=False)
    assert sample_energy.numpy().shape[0] == N  # type: ignore
