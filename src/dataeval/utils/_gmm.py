"""
Adapted for Pytorch from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

__all__ = []

from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import torch

from dataeval.config import EPSILON

TGMMData = TypeVar("TGMMData")


@dataclass
class GaussianMixtureModelParams:
    """
    phi : torch.Tensor
        Mixture component distribution weights.
    mu : torch.Tensor
        Mixture means.
    cov : torch.Tensor
        Mixture covariance.
    L : torch.Tensor
        Cholesky decomposition of `cov`.
    log_det_cov : torch.Tensor
        Log of the determinant of `cov`.
    """

    phi: torch.Tensor
    mu: torch.Tensor
    cov: torch.Tensor
    L: torch.Tensor
    log_det_cov: torch.Tensor


def gmm_params(z: torch.Tensor, gamma: torch.Tensor) -> GaussianMixtureModelParams:
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
    mu = torch.sum(torch.unsqueeze(gamma, -1) * torch.unsqueeze(z, 1), 0) / torch.unsqueeze(sum_gamma, -1)
    z_mu = torch.unsqueeze(z, 1) - torch.unsqueeze(mu, 0)  # N x K x D
    z_mu_outer = torch.unsqueeze(z_mu, -1) * torch.unsqueeze(z_mu, -2)  # N x K x D x D

    # K x D x D
    cov = torch.sum(torch.unsqueeze(torch.unsqueeze(gamma, -1), -1) * z_mu_outer, 0) / torch.unsqueeze(
        torch.unsqueeze(sum_gamma, -1), -1
    )

    # cholesky decomposition of covariance and determinant derivation
    D = cov.shape[1]
    L = torch.linalg.cholesky(cov + torch.eye(D) * EPSILON)  # K x D x D
    log_det_cov = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), 1)  # K

    return GaussianMixtureModelParams(phi, mu, cov, L, log_det_cov)


def gmm_energy(
    z: torch.Tensor,
    params: GaussianMixtureModelParams,
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
    D = params.cov.shape[1]
    z_mu = torch.unsqueeze(z, 1) - torch.unsqueeze(params.mu, 0)  # N x K x D
    z_mu_T = torch.permute(z_mu, dims=[1, 2, 0])  # K x D x N
    v = torch.linalg.solve_triangular(params.L, z_mu_T, upper=False)  # K x D x D

    # rewrite sample energy in logsumexp format for numerical stability
    logits = torch.log(torch.unsqueeze(params.phi, -1)) - 0.5 * (
        torch.sum(torch.square(v), 1) + float(D) * np.log(2.0 * np.pi) + torch.unsqueeze(params.log_det_cov, -1)
    )  # K x N
    sample_energy = -torch.logsumexp(logits, 0)  # N

    if return_mean:
        sample_energy = torch.mean(sample_energy)

    # inverse sum of variances
    cov_diag = torch.sum(torch.divide(torch.tensor(1), torch.diagonal(params.cov, dim1=-2, dim2=-1)))

    return sample_energy, cov_diag
