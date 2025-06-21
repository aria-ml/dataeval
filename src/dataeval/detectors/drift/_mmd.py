"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

__all__ = []

from collections.abc import Callable
from typing import Any

import torch

from dataeval.config import DeviceLike, get_device
from dataeval.data._embeddings import Embeddings
from dataeval.detectors.drift._base import BaseDrift, UpdateStrategy, update_strategy
from dataeval.outputs import DriftMMDOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import Array


class DriftMMD(BaseDrift):
    """
    :term:`Maximum Mean Discrepancy (MMD) Drift Detection` algorithm \
    using a permutation test.

    Parameters
    ----------
    data : Embeddings or Array
        Data used as reference distribution.
    p_val : float or None, default 0.05
        :term:`P-value` used for significance of the statistical test for each feature.
        If the FDR correction method is used, this corresponds to the acceptable
        q-value.
    update_strategy : UpdateStrategy or None, default None
        Reference data can optionally be updated using an UpdateStrategy class. Update
        using the last n instances seen by the detector with LastSeenUpdateStrategy
        or via reservoir sampling with ReservoirSamplingUpdateStrategy.
    sigma : Array or None, default None
        Optionally set the internal GaussianRBF kernel bandwidth. Can also pass multiple
        bandwidth values as an array. The kernel evaluation is then averaged over
        those bandwidths.
    n_permutations : int, default 100
        Number of permutations used in the permutation test.
    device : DeviceLike or None, default None
        The hardware device to use if specified, otherwise uses the DataEval
        default or torch default.

    Example
    -------
    >>> from dataeval.data import Embeddings

    Use Embeddings to encode images before testing for drift

    >>> train_emb = Embeddings(train_images, model=encoder, batch_size=64)
    >>> drift = DriftMMD(train_emb)

    Test incoming images for drift

    >>> drift.predict(test_images).drifted
    True
    """

    def __init__(
        self,
        data: Embeddings | Array,
        p_val: float = 0.05,
        update_strategy: UpdateStrategy | None = None,
        sigma: Array | None = None,
        n_permutations: int = 100,
        device: DeviceLike | None = None,
    ) -> None:
        super().__init__(data, p_val, update_strategy)

        self.n_permutations = n_permutations  # nb of iterations through permutation test

        # set device
        self.device: torch.device = get_device(device)

        # initialize kernel
        sigma_tensor = torch.as_tensor(sigma, device=self.device) if sigma is not None else None
        self._kernel = GaussianRBF(sigma_tensor).to(self.device)

        # compute kernel matrix for the reference data
        if isinstance(sigma_tensor, torch.Tensor):
            self._k_xx = self._kernel(self.x_ref, self.x_ref)
        else:
            self._k_xx = None

    def _kernel_matrix(self, x: Array, y: Array) -> torch.Tensor:
        """Compute and return full kernel matrix between arrays x and y."""
        k_xy = self._kernel(x, y)
        k_xx = self._k_xx if self._k_xx is not None and self.update_strategy is None else self._kernel(x, x)
        k_yy = self._kernel(y, y)
        return torch.cat([torch.cat([k_xx, k_xy], 1), torch.cat([k_xy.T, k_yy], 1)], 0)

    def score(self, data: Embeddings | Array) -> tuple[float, float, float]:
        """
        Compute the :term:`p-value<P-Value>` resulting from a permutation test using the maximum mean
        discrepancy as a distance measure between the reference data and the data to
        be tested.

        Parameters
        ----------
        data : Embeddings or Array
            Batch of instances to score.

        Returns
        -------
        tuple(float, float, float)
            p-value obtained from the permutation test, MMD^2 between the reference and test set,
            and MMD^2 threshold above which :term:`drift<Drift>` is flagged
        """
        x_test = self._encode(data)
        n = x_test.shape[0]
        kernel_mat = self._kernel_matrix(self.x_ref, x_test)
        kernel_mat = kernel_mat - torch.diag(kernel_mat.diag())  # zero diagonal
        mmd2 = mmd2_from_kernel_matrix(kernel_mat, n, permute=False, zero_diag=False)
        mmd2_permuted = torch.tensor(
            [mmd2_from_kernel_matrix(kernel_mat, n, permute=True, zero_diag=False)] * self.n_permutations,
            device=self.device,
        )
        p_val = (mmd2 <= mmd2_permuted).float().mean()
        # compute distance threshold
        idx_threshold = int(self.p_val * len(mmd2_permuted))
        distance_threshold = torch.sort(mmd2_permuted, descending=True).values[idx_threshold]
        return float(p_val.item()), float(mmd2.item()), float(distance_threshold.item())

    @set_metadata
    @update_strategy
    def predict(self, data: Embeddings | Array) -> DriftMMDOutput:
        """
        Predict whether a batch of data has drifted from the reference data and then
        updates reference data using specified strategy.

        Parameters
        ----------
        data : Embeddings or Array
            Batch of instances to predict drift on.

        Returns
        -------
        DriftMMDOutput
            Output class containing the :term:`drift<Drift>` prediction, :term:`p-value<P-Value>`,
            threshold and MMD metric.
        """
        # compute drift scores
        p_val, dist, distance_threshold = self.score(data)
        drift_pred = bool(p_val < self.p_val)

        # populate drift dict
        return DriftMMDOutput(drift_pred, self.p_val, p_val, dist, distance_threshold)


@torch.jit.script
def _squared_pairwise_distance(
    x: torch.Tensor, y: torch.Tensor, a_min: float = 1e-30
) -> torch.Tensor:  # pragma: no cover - torch.jit.script code is compiled and copied
    """
    PyTorch pairwise squared Euclidean distance between samples x and y.

    Parameters
    ----------
    x : torch.Tensor
        Batch of instances of shape [Nx, features].
    y : torch.Tensor
        Batch of instances of shape [Ny, features].
    a_min : float
        Lower bound to clip distance values.

    Returns
    -------
    torch.Tensor
        Pairwise squared Euclidean distance [Nx, Ny].
    """
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    dist = torch.addmm(y2.transpose(-2, -1), x, y.transpose(-2, -1), alpha=-2).add_(x2)
    return dist.clamp_min_(a_min)


def sigma_median(x: torch.Tensor, y: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    """
    Bandwidth estimation using the median heuristic `Gretton2012`

    Parameters
    ----------
    x : torch.Tensor
        Tensor of instances with dimension [Nx, features].
    y : torch.Tensor
        Tensor of instances with dimension [Ny, features].
    dist : torch.Tensor
        Tensor with dimensions [Nx, Ny], containing the pairwise distances
        between `x` and `y`.

    Returns
    -------
    torch.Tensor
        The computed bandwidth, `sigma`.
    """
    n = min(x.shape[0], y.shape[0])
    n = n if (x[:n] == y[:n]).all() and x.shape == y.shape else 0
    n_median = n + (torch.prod(torch.as_tensor(dist.shape)) - n) // 2 - 1
    return (0.5 * dist.flatten().sort().values[int(n_median)].unsqueeze(dim=-1)) ** 0.5


class GaussianRBF(torch.nn.Module):
    """
    Gaussian RBF kernel: k(x,y) = exp(-(1/(2*sigma^2)||x-y||^2).

    A forward pass takes a batch of instances x [Nx, features] and
    y [Ny, features] and returns the kernel matrix [Nx, Ny].

    Parameters
    ----------
    sigma : torch.Tensor | None, default None
        Bandwidth used for the kernel. Needn't be specified if being inferred or
        trained. Can pass multiple values to eval kernel with and then average.
    init_sigma_fn : Callable | None, default None
        Function used to compute the bandwidth ``sigma``. Used when ``sigma`` is to be
        inferred. The function's signature should take in the tensors ``x``, ``y`` and
        ``dist`` and return ``sigma``. If ``None``, it is set to ``sigma_median``.
    trainable : bool, default False
        Whether or not to track gradients w.r.t. `sigma` to allow it to be trained.
    """

    def __init__(
        self,
        sigma: torch.Tensor | None = None,
        init_sigma_fn: Callable | None = None,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        init_sigma_fn = sigma_median if init_sigma_fn is None else init_sigma_fn
        self.config: dict[str, Any] = {
            "sigma": sigma,
            "trainable": trainable,
            "init_sigma_fn": init_sigma_fn,
        }
        if sigma is None:
            self.log_sigma: torch.nn.Parameter = torch.nn.Parameter(torch.empty(1), requires_grad=trainable)
            self.init_required: bool = True
        else:
            sigma = sigma.reshape(-1)  # [Ns,]
            self.log_sigma: torch.nn.Parameter = torch.nn.Parameter(sigma.log(), requires_grad=trainable)
            self.init_required: bool = False
        self.init_sigma_fn = init_sigma_fn
        self.trainable = trainable

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    def forward(
        self,
        x: Array,
        y: Array,
        infer_sigma: bool = False,
    ) -> torch.Tensor:
        x, y = torch.as_tensor(x), torch.as_tensor(y)
        dist = _squared_pairwise_distance(x.flatten(1), y.flatten(1))  # [Nx, Ny]

        if infer_sigma or self.init_required:
            if self.trainable and infer_sigma:
                raise ValueError("Gradients cannot be computed w.r.t. an inferred sigma value")
            sigma = self.init_sigma_fn(x, y, dist)
            with torch.no_grad():
                self.log_sigma.copy_(sigma.log().clone())
            self.init_required: bool = False

        gamma = 1.0 / (2.0 * self.sigma**2)  # [Ns,]
        # TODO: do matrix multiplication after all?
        kernel_mat = torch.exp(-torch.cat([(g * dist)[None, :, :] for g in gamma], dim=0))  # [Ns, Nx, Ny]
        return kernel_mat.mean(dim=0)  # [Nx, Ny]


def mmd2_from_kernel_matrix(
    kernel_mat: torch.Tensor, m: int, permute: bool = False, zero_diag: bool = True
) -> torch.Tensor:
    """
    Compute maximum mean discrepancy (MMD^2) between 2 samples x and y from the
    full kernel matrix between the samples.

    Parameters
    ----------
    kernel_mat : torch.Tensor
        Kernel matrix between samples x and y.
    m : int
        Number of instances in y.
    permute : bool, default False
        Whether to permute the row indices. Used for permutation tests.
    zero_diag : bool, default True
        Whether to zero out the diagonal of the kernel matrix.

    Returns
    -------
    torch.Tensor
        MMD^2 between the samples from the kernel matrix.
    """
    n = kernel_mat.shape[0] - m
    if zero_diag:
        kernel_mat = kernel_mat - torch.diag(kernel_mat.diag())
    if permute:
        idx = torch.randperm(kernel_mat.shape[0])
        kernel_mat = kernel_mat[idx][:, idx]
    k_xx, k_yy, k_xy = kernel_mat[:-m, :-m], kernel_mat[-m:, -m:], kernel_mat[-m:, :-m]
    c_xx, c_yy = 1 / (n * (n - 1)), 1 / (m * (m - 1))
    return c_xx * k_xx.sum() + c_yy * k_yy.sum() - 2.0 * k_xy.mean()
