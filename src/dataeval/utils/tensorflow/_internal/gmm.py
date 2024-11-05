"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import tensorflow as tf


class GaussianMixtureModelParams(NamedTuple):
    """
    phi : tf.Tensor
        Mixture component distribution weights.
    mu : tf.Tensor
        Mixture means.
    cov : tf.Tensor
        Mixture covariance.
    L : tf.Tensor
        Cholesky decomposition of `cov`.
    log_det_cov : tf.Tensor
        Log of the determinant of `cov`.
    """

    phi: tf.Tensor
    mu: tf.Tensor
    cov: tf.Tensor
    L: tf.Tensor
    log_det_cov: tf.Tensor


def gmm_params(z: tf.Tensor, gamma: tf.Tensor) -> GaussianMixtureModelParams:
    """
    Compute parameters of Gaussian Mixture Model.

    Parameters
    ----------
    z : tf.Tensor
        Observations.
    gamma : tf.Tensor
        Mixture probabilities to derive mixture distribution weights from.

    Returns
    -------
    GaussianMixtureModelParams(phi, mu, cov, L, log_det_cov)
        The parameters used to calculate energy.
    """
    # compute gmm parameters phi, mu and cov
    N = gamma.shape[0]  # nb of samples in batch
    sum_gamma = tf.reduce_sum(gamma, 0)  # K
    phi = sum_gamma / N  # K
    mu = tf.reduce_sum(tf.expand_dims(gamma, -1) * tf.expand_dims(z, 1), 0) / tf.expand_dims(
        sum_gamma, -1
    )  # K x D (D = latent_dim)
    z_mu = tf.expand_dims(z, 1) - tf.expand_dims(mu, 0)  # N x K x D
    z_mu_outer = tf.expand_dims(z_mu, -1) * tf.expand_dims(z_mu, -2)  # N x K x D x D
    cov = tf.reduce_sum(tf.expand_dims(tf.expand_dims(gamma, -1), -1) * z_mu_outer, 0) / tf.expand_dims(
        tf.expand_dims(sum_gamma, -1), -1
    )  # K x D x D

    # cholesky decomposition of covariance and determinant derivation
    D = tf.shape(cov)[1]  # type: ignore
    eps = 1e-6
    L = tf.linalg.cholesky(cov + tf.eye(D) * eps)  # K x D x D
    log_det_cov = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), 1)  # K

    return GaussianMixtureModelParams(phi, mu, cov, L, log_det_cov)


def gmm_energy(
    z: tf.Tensor,
    params: GaussianMixtureModelParams,
    return_mean: bool = True,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Compute sample energy from Gaussian Mixture Model.

    Parameters
    ----------
    params : GaussianMixtureModelParams
        The gaussian mixture model parameters.
    return_mean : bool, default True
        Take mean across all sample energies in a batch.

    Returns
    -------
    sample_energy
        The sample energy of the GMM.
    cov_diag
        The inverse sum of the diagonal components of the covariance matrix.
    """
    D = tf.shape(params.cov)[1]  # type: ignore
    z_mu = tf.expand_dims(z, 1) - tf.expand_dims(params.mu, 0)  # N x K x D
    z_mu_T = tf.transpose(z_mu, perm=[1, 2, 0])  # K x D x N
    v = tf.linalg.triangular_solve(params.L, z_mu_T, lower=True)  # K x D x D

    # rewrite sample energy in logsumexp format for numerical stability
    logits = tf.math.log(tf.expand_dims(params.phi, -1)) - 0.5 * (
        tf.reduce_sum(tf.square(v), 1)
        + tf.cast(D, tf.float32) * tf.math.log(2.0 * np.pi)  # type: ignore py38
        + tf.expand_dims(params.log_det_cov, -1)
    )  # K x N
    sample_energy = -tf.reduce_logsumexp(logits, axis=0)  # N

    if return_mean:
        sample_energy = tf.reduce_mean(sample_energy)

    # inverse sum of variances
    cov_diag = tf.reduce_sum(tf.divide(1, tf.linalg.diag_part(params.cov)))

    return sample_energy, cov_diag
