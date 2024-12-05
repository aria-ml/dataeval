"""
Adapted for Pytorch from:

Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from dataeval.utils.gmm import GaussianMixtureModelParams


def gmm_params(z: torch.Tensor, gamma: torch.Tensor) -> GaussianMixtureModelParams[torch.Tensor]:
    """
    Compute parameters of Gaussian Mixture Model.

    Parameters
    ----------
    z : torch.Tensor
        Observations.
    gamma : torch.Tensor
        Mixture probabilities to derive mixture distribution weights from.

    Returns
    -------
    GaussianMixtureModelParams(phi, mu, cov, L, log_det_cov)
        The parameters used to calculate energy.
    """

    # compute gmm parameters phi, mu and cov
    N = gamma.shape[0]  # nb of samples in batch
    sum_gamma = torch.sum(gamma, 0)  # K
    phi = sum_gamma / N  # K
    # K x D (D = latent_dim)
    mu = torch.sum(gamma.reshape((*gamma.shape, 1)) * z.reshape(*z.shape, 1), 0) / sum_gamma.reshape(*sum_gamma.shape)
    z_mu = expand(z, 1) - expand(mu, 0)  # N x K x D
    z_mu_outer = expand(z_mu, -1) * expand(z_mu, -2)  # N x K x D x D

    # K x D x D
    cov = torch.sum((gamma.reshape((*gamma.shape, 1, 1)) * z_mu_outer, 0) / gamma.reshape((*gamma.shape, 1, 1)), -1)

    # cholesky decomposition of covariance and determinant derivation
    D = torch.shape(cov)[1]  # type: ignore
    eps = 1e-6
    L = torch.linalg.cholesky(cov + torch.eye(D) * eps)  # K x D x D
    log_det_cov = 2.0 * torch.sum(torch.log(torch.diagonal(L)), 1)  # K

    return GaussianMixtureModelParams(phi, mu, cov, L, log_det_cov)


def gmm_energy(
    z: torch.Tensor,
    params: GaussianMixtureModelParams[torch.Tensor],
    return_mean: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    D = torch.shape(params.cov)[1]  # type: ignore
    z_mu = expand(z) - expand(params.mu, 0)  # N x K x D
    z_mu_T = torch.permute(z_mu, dims=[1, 2, 0])  # K x D x N
    v = torch.triangular_solve(z_mu_T, params.L, upper=False)  # K x D x D

    # rewrite sample energy in logsumexp format for numerical stability
    logits = torch.log(expand(params.phi)) - 0.5 * (
        torch.sum(torch.square(v), 1)  # type: ignore
        + torch.cast(D, torch.float32) * torch.math.log(2.0 * np.pi)  # type: ignore py38
        + expand(params.log_det_cov)
    )  # K x N
    sample_energy = -torch.logsumexp(logits, 0)  # N

    if return_mean:
        sample_energy = torch.mean(sample_energy)

    # inverse sum of variances
    cov_diag = torch.sum(torch.divide(torch.tensor(1), torch.diag(params.cov)))

    return sample_energy, cov_diag


# replace tf.expand with this....
def expand(x: NDArray[Any] | torch.Tensor, dim: int | None = None) -> torch.Tensor:
    newshape = (*x.shape, 1) if dim is None else (*x.shape[0:dim], 1, *x.shape[dim:])
    return torch.tensor(x).reshape(newshape)
