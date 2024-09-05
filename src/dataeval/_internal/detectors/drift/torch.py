"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray


def get_device(device: str | torch.device | None = None) -> torch.device:
    """
    Instantiates a PyTorch device object.

    Parameters
    ----------
    device : str | torch.device | None, default None
        Either ``None``, a str ('gpu' or 'cpu') indicating the device to choose, or an
        already instantiated device object. If ``None``, the GPU is selected if it is
        detected, otherwise the CPU is used as a fallback.

    Returns
    -------
    The instantiated device object.
    """
    if isinstance(device, torch.device):  # Already a torch device
        return device
    else:  # Instantiate device
        if device is None or device.lower() in ["gpu", "cuda"]:
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            torch_device = torch.device("cpu")
    return torch_device


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
    mmd2 = c_xx * k_xx.sum() + c_yy * k_yy.sum() - 2.0 * k_xy.mean()
    return mmd2


def predict_batch(
    x: NDArray | torch.Tensor,
    model: Callable | nn.Module | nn.Sequential,
    device: torch.device | None = None,
    batch_size: int = int(1e10),
    preprocess_fn: Callable | None = None,
    dtype: type[np.generic] | torch.dtype = np.float32,
) -> NDArray | torch.Tensor | tuple:
    """
    Make batch predictions on a model.

    Parameters
    ----------
    x : np.ndarray | torch.Tensor
        Batch of instances.
    model : Callable | nn.Module | nn.Sequential
        PyTorch model.
    device : torch.device | None, default None
        Device type used. The default None tries to use the GPU and falls back on CPU.
        Can be specified by passing either torch.device('cuda') or torch.device('cpu').
    batch_size : int, default 1e10
        Batch size used during prediction.
    preprocess_fn : Callable | None, default None
        Optional preprocessing function for each batch.
    dtype : np.dtype | torch.dtype, default np.float32
        Model output type, either a numpy or torch dtype, e.g. np.float32 or torch.float32.

    Returns
    -------
    NDArray | torch.Tensor | tuple
        Numpy array, torch tensor or tuples of those with model outputs.
    """
    device = get_device(device)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    n = len(x)
    n_minibatch = int(np.ceil(n / batch_size))
    return_np = not isinstance(dtype, torch.dtype)
    preds = []
    with torch.no_grad():
        for i in range(n_minibatch):
            istart, istop = i * batch_size, min((i + 1) * batch_size, n)
            x_batch = x[istart:istop]
            if isinstance(preprocess_fn, Callable):
                x_batch = preprocess_fn(x_batch)
            preds_tmp = model(x_batch.to(device))
            if isinstance(preds_tmp, (list, tuple)):
                if len(preds) == 0:  # init tuple with lists to store predictions
                    preds = tuple([] for _ in range(len(preds_tmp)))
                for j, p in enumerate(preds_tmp):
                    if isinstance(p, torch.Tensor):
                        p = p.cpu()
                    preds[j].append(p if not return_np or isinstance(p, np.ndarray) else p.numpy())
            elif isinstance(preds_tmp, (np.ndarray, torch.Tensor)):
                if isinstance(preds_tmp, torch.Tensor):
                    preds_tmp = preds_tmp.cpu()
                if isinstance(preds, tuple):
                    preds = list(preds)
                preds.append(
                    preds_tmp
                    if not return_np or isinstance(preds_tmp, np.ndarray)  # type: ignore
                    else preds_tmp.numpy()
                )
            else:
                raise TypeError(
                    f"Model output type {type(preds_tmp)} not supported. The model \
                    output type needs to be one of list, tuple, NDArray or \
                    torch.Tensor."
                )
    concat = partial(np.concatenate, axis=0) if return_np else partial(torch.cat, dim=0)
    out: tuple | np.ndarray | torch.Tensor = (
        tuple(concat(p) for p in preds) if isinstance(preds, tuple) else concat(preds)  # type: ignore
    )
    return out


def preprocess_drift(
    x: NDArray,
    model: nn.Module,
    device: torch.device | None = None,
    preprocess_batch_fn: Callable | None = None,
    batch_size: int = int(1e10),
    dtype: type[np.generic] | torch.dtype = np.float32,
) -> NDArray | torch.Tensor | tuple:
    """
    Prediction function used for preprocessing step of drift detector.

    Parameters
    ----------
    x : NDArray
        Batch of instances.
    model : nn.Module
        Model used for preprocessing.
    device : torch.device | None, default None
        Device type used. The default None tries to use the GPU and falls back on CPU.
        Can be specified by passing either torch.device('cuda') or torch.device('cpu').
    preprocess_batch_fn : Callable | None, default None
        Optional batch preprocessing function. For example to convert a list of objects
        to a batch which can be processed by the PyTorch model.
    batch_size : int, default 1e10
        Batch size used during prediction.
    dtype : np.dtype | torch.dtype, default np.float32
        Model output type, either a numpy or torch dtype, e.g. np.float32 or torch.float32.

    Returns
    -------
    NDArray | torch.Tensor | tuple
        Numpy array, torch tensor or tuples of those with model outputs.
    """
    return predict_batch(
        x,
        model,
        device=device,
        batch_size=batch_size,
        preprocess_fn=preprocess_batch_fn,
        dtype=dtype,
    )


@torch.jit.script
def squared_pairwise_distance(
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
    Bandwidth estimation using the median heuristic :cite:t:`Gretton2012`.

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
    n_median = n + (np.prod(dist.shape) - n) // 2 - 1
    sigma = (0.5 * dist.flatten().sort().values[int(n_median)].unsqueeze(dim=-1)) ** 0.5
    return sigma


class GaussianRBF(nn.Module):
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
        self.config = {
            "sigma": sigma,
            "trainable": trainable,
            "init_sigma_fn": init_sigma_fn,
        }
        if sigma is None:
            self.log_sigma = nn.Parameter(torch.empty(1), requires_grad=trainable)
            self.init_required = True
        else:
            sigma = sigma.reshape(-1)  # [Ns,]
            self.log_sigma = nn.Parameter(sigma.log(), requires_grad=trainable)
            self.init_required = False
        self.init_sigma_fn = init_sigma_fn
        self.trainable = trainable

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    def forward(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        infer_sigma: bool = False,
    ) -> torch.Tensor:
        x, y = torch.as_tensor(x), torch.as_tensor(y)
        dist = squared_pairwise_distance(x.flatten(1), y.flatten(1))  # [Nx, Ny]

        if infer_sigma or self.init_required:
            if self.trainable and infer_sigma:
                raise ValueError("Gradients cannot be computed w.r.t. an inferred sigma value")
            sigma = self.init_sigma_fn(x, y, dist)
            with torch.no_grad():
                self.log_sigma.copy_(sigma.log().clone())
            self.init_required = False

        gamma = 1.0 / (2.0 * self.sigma**2)  # [Ns,]
        # TODO: do matrix multiplication after all?
        kernel_mat = torch.exp(-torch.cat([(g * dist)[None, :, :] for g in gamma], dim=0))  # [Ns, Nx, Ny]
        return kernel_mat.mean(dim=0)  # [Nx, Ny]
