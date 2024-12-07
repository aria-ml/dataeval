"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

import numpy as np
import pytest
import tensorflow as tf
import torch

import dataeval.utils.tensorflow._internal.gmm as tensorflow_gmm
import dataeval.utils.torch.gmm as torch_gmm

N, K, D = 10, 5, 1
z = np.random.rand(N, D).astype(np.float32)
gamma = np.random.rand(N, K).astype(np.float32)


@pytest.mark.parametrize(
    "module, tz, tg",
    [
        (tensorflow_gmm, tf.convert_to_tensor(z), tf.convert_to_tensor(gamma)),
        (torch_gmm, torch.from_numpy(z), torch.from_numpy(gamma)),
    ],
)
def test_gmm_params_energy(module, tz, tg):
    params = module.gmm_params(tz, tg)
    assert params.phi.numpy().shape[0] == K == params.log_det_cov.shape[0]  # type: ignore
    assert params.mu.numpy().shape == (K, D)  # type: ignore
    assert params.cov.numpy().shape == params.L.numpy().shape == (K, D, D)  # type: ignore
    for _ in range(params.cov.numpy().shape[0]):  # type: ignore
        assert (np.diag(params.cov[_].numpy()) >= 0.0).all()  # type: ignore
        assert (np.diag(params.L[_].numpy()) >= 0.0).all()  # type: ignore

    sample_energy, cov_diag = module.gmm_energy(tz, params, return_mean=True)
    assert sample_energy.numpy().shape == cov_diag.numpy().shape == ()  # type: ignore

    sample_energy, _ = module.gmm_energy(tz, params, return_mean=False)
    assert sample_energy.numpy().shape[0] == N  # type: ignore
