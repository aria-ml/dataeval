"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

from typing import Literal, cast

import tensorflow as tf
from keras.layers import Flatten
from numpy.typing import NDArray
from tensorflow_probability.python.distributions.mvn_diag import MultivariateNormalDiag
from tensorflow_probability.python.distributions.mvn_tril import MultivariateNormalTriL
from tensorflow_probability.python.stats import covariance

from dataeval._internal.models.tensorflow.gmm import gmm_energy, gmm_params


class Elbo:
    """
    Compute ELBO loss.

    The covariance matrix can be specified by passing the full covariance matrix, the matrix
    diagonal, or a scale identity multiplier. Only one of these should be specified. If none are specified, the
    identity matrix is used.

    Parameters
    ----------
    cov_type : Union[Literal["cov_full", "cov_diag"], float], default 1.0
        Full covariance matrix, diagonal variance matrix, or scale identity multiplier.
    x : ArrayLike, optional - default None
        Dataset used to calculate the covariance matrix.  Required for full and diagonal covariance matrix types.
    """

    def __init__(
        self,
        cov_type: Literal["cov_full", "cov_diag"] | float = 1.0,
        x: tf.Tensor | NDArray | None = None,
    ):
        if isinstance(cov_type, float):
            self.cov = ("sim", cov_type)
        elif cov_type in ["cov_full", "cov_diag"]:
            x_np: NDArray = x.numpy() if tf.is_tensor(x) else x  # type: ignore
            cov = covariance(x_np.reshape(x_np.shape[0], -1))  # type: ignore py38
            if cov_type == "cov_diag":  # infer standard deviation from covariance matrix
                cov = tf.math.sqrt(tf.linalg.diag_part(cov))
            self.cov = (cov_type, cov)
        else:
            raise ValueError("Only cov_full, cov_diag or sim value should be specified.")

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred_flat = cast(tf.Tensor, Flatten()(y_pred))

        if self.cov[0] == "cov_full":
            y_mn = MultivariateNormalTriL(y_pred_flat, scale_tril=tf.linalg.cholesky(self.cov[1]))
        else:  # cov_diag and sim
            cov_diag = self.cov[1] if self.cov[0] == "cov_diag" else self.cov[1] * tf.ones(y_pred_flat.shape[-1])
            y_mn = MultivariateNormalDiag(y_pred_flat, scale_diag=cov_diag)

        loss = -tf.reduce_mean(y_mn.log_prob(Flatten()(y_true)))
        return loss


class LossGMM:
    """
    Loss function used for AE and VAE with GMM.

    Parameters
    ----------
    w_recon : float, default 1e-7
        Weight on elbo loss term.
    w_energy : float, default 0.1
        Weight on sample energy loss term.
    w_cov_diag : float, default 0.005
        Weight on covariance regularizing loss term.
    elbo : Elbo, optional - default None
        ELBO loss function used to calculate w_recon.
    """

    def __init__(
        self,
        w_recon: float = 1e-7,
        w_energy: float = 0.1,
        w_cov_diag: float = 0.005,
        elbo: Elbo | None = None,
    ):
        self.w_recon = w_recon
        self.w_energy = w_energy
        self.w_cov_diag = w_cov_diag
        self.elbo = elbo

    def __call__(
        self,
        x_true: tf.Tensor,
        x_pred: tf.Tensor,
        z: tf.Tensor,
        gamma: tf.Tensor,
    ) -> tf.Tensor:
        w_recon = (
            tf.reduce_mean(tf.subtract(x_true, x_pred) ** 2)
            if self.elbo is None
            else tf.multiply(self.w_recon, self.elbo(x_true, x_pred))
        )
        sample_energy, cov_diag = gmm_energy(z, gmm_params(z, gamma))
        w_energy = tf.multiply(self.w_energy, sample_energy)
        w_cov_diag = tf.multiply(self.w_cov_diag, cov_diag)
        return w_recon + w_energy + w_cov_diag
