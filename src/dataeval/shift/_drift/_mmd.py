"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

__all__ = []

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch

from dataeval.config import get_device
from dataeval.protocols import Array, DeviceLike, UpdateStrategy
from dataeval.shift._drift._base import BaseDrift, DriftBaseOutput, update_strategy
from dataeval.types import set_metadata


def _auto_detect_permutation_batch_size(
    kernel_mat_size: int,
    n_permutations: int,
    device: torch.device,
) -> int:
    """
    Auto-detect appropriate permutation batch size based on available GPU memory.

    Parameters
    ----------
    kernel_mat_size : int
        Size of the kernel matrix (n_ref + n_test).
    n_permutations : int
        Total number of permutations to compute.
    device : torch.device
        Device where computation will occur.

    Returns
    -------
    int
        Suggested batch size, or 1 if batching is not needed (CPU or sufficient memory).

    Notes
    -----
    This function estimates memory requirements and suggests a batch size that keeps
    memory usage reasonable. The heuristic is very conservative to avoid OOM errors,
    accounting for:

    - CUDA caching allocator overhead and memory fragmentation
    - Unreleased memory from previous operations
    - Intermediate computation buffers
    - Other concurrent GPU operations

    The function uses only 50% of available memory and suggests batch sizes at 50%
    of the calculated maximum, ensuring safe operation even in memory-constrained
    environments.

    Memory is primarily consumed by the permuted kernel matrices which have shape
    (batch_size, kernel_mat_size, kernel_mat_size) in float32 (4 bytes per element).
    """
    # Only auto-batch on CUDA devices
    if device.type != "cuda":
        return 1

    try:
        # Get available GPU memory
        total_memory = torch.cuda.get_device_properties(device).total_memory

        # Use reserved memory (not allocated) as it includes CUDA caching allocator overhead
        # This is more conservative and accounts for memory that hasn't been freed yet
        reserved_memory = torch.cuda.memory_reserved(device)

        # Be very conservative: only use 50% of remaining memory after reserved
        safe_available_memory = (total_memory - reserved_memory) * 0.5

        # Each permuted kernel matrix: kernel_mat_size^2 * 4 bytes (float32)
        # Account for intermediate computations and overhead (multiply by 2.0 for safety)
        memory_per_permutation = kernel_mat_size * kernel_mat_size * 4 * 2.0

        # Calculate how many permutations can fit in available memory
        max_batch_size = max(1, int(safe_available_memory / memory_per_permutation))

        # If we can fit all permutations, no need to batch
        if max_batch_size >= n_permutations:
            return 1

        # Otherwise, use a very conservative batch size (50% of max to be extra safe)
        return max(1, int(max_batch_size * 0.5))

    except (RuntimeError, AttributeError):
        # If we can't determine memory, use a conservative default
        # Heuristic: batch size of 10 for large matrices (>500), otherwise don't batch
        if kernel_mat_size > 500:
            return 10
        return 1


@dataclass(frozen=True)
class DriftMMDOutput(DriftBaseOutput):
    """
    Output class for :class:`.DriftMMD` (Maximum Mean Discrepancy) drift detector.

    Extends :class:`.DriftBaseOutput` with MMD-specific distance threshold information.
    Used by MMD-based drift detectors that compare kernel embeddings between
    reference and test distributions.

    Attributes
    ----------
    drifted : bool
        Whether drift was detected based on MMD permutation test.
    threshold : float
        P-value threshold used for significance of the permutation test.
    p_val : float
        P-value obtained from the MMD permutation test, between 0 and 1.
    distance : float
        Squared Maximum Mean Discrepancy between reference and test set.
        Always >= 0, with higher values indicating greater distributional difference.
    distance_threshold : float
        Squared Maximum Mean Discrepancy threshold above which drift is flagged, always >= 0.
        Determined from permutation test at specified significance level.

    Notes
    -----
    MMD uses kernel methods to compare distributions in reproducing kernel
    Hilbert spaces, making it effective for high-dimensional data like images.
    """

    distance_threshold: float


class DriftMMD(BaseDrift):
    """Drift detector using :term:`Maximum Mean Discrepancy (MMD) Drift Detection` with permutation test.

    Detects distributional differences by comparing kernel embeddings of reference
    and test datasets in a reproducing kernel Hilbert space (RKHS). Uses permutation
    testing to assess statistical significance of the observed MMD^2 statistic.

    MMD is particularly effective for high-dimensional data like images as it can
    capture complex distributional differences that univariate tests might miss.
    The kernel-based approach enables detection of both marginal and dependency
    changes between features.

    Parameters
    ----------
    data : Array
        Reference dataset used as baseline distribution for drift detection.
        Should represent the expected data distribution.
    p_val : float, default 0.05
        Significance threshold for statistical tests, between 0 and 1.
        For FDR correction, this represents the acceptable false discovery rate.
        Default 0.05 provides 95% confidence level for drift detection.
    update_strategy : UpdateStrategy or None, default None
        Strategy for updating reference data when new data arrives.
        When None, reference data remains fixed throughout detection.
    sigma : Array or None, default None
        Bandwidth parameter(s) for the Gaussian RBF kernel. Controls the
        kernel's sensitivity to distance between data points. When None,
        automatically selects bandwidth using median heuristic. Can provide
        multiple values as array to average over different scales.
    n_permutations : int, default 100
        Number of random permutations used in the permutation test to estimate
        the null distribution of MMD² under no drift. Higher values provide
        more accurate p-value estimates but increase computation time.
        Default 100 balances statistical accuracy with computational efficiency.
    permutation_batch_size : int or "auto", default "auto"
        Batch size for computing permutations to reduce memory usage. When "auto" (default),
        automatically detects appropriate batch size based on available GPU memory
        (on CUDA devices) or computes all permutations at once (on CPU). Set to an
        integer to manually specify batch size. Useful when working with large kernel
        matrices or many permutations to avoid GPU out-of-memory errors. For example,
        with n_permutations=100 and permutation_batch_size=10, permutations are computed
        in 10 batches of 10 each.
    device : DeviceLike or None, default None
        Hardware device for computation. When None, automatically selects
        DataEval's configured device, falling back to PyTorch's default.
    config : DriftMMD.Config or None, default None
        Optional configuration object with default parameters. Parameters
        specified directly in __init__ will override config defaults.

    Attributes
    ----------
    p_val : float
        Significance threshold for statistical tests.
    update_strategy : UpdateStrategy or None
        Reference data update strategy.
    n : int
        Number of samples in the reference dataset.
    sigma : Array or None
        Gaussian RBF kernel bandwidth parameter(s).
    n_permutations : int
        Number of permutations for statistical testing.
    permutation_batch_size : int or "auto"
        Batch size for computing permutations, or "auto" for automatic detection.
    device : torch.device
        Hardware device used for computations.

    Example
    -------
    Initialize with image embeddings

    >>> train_emb = np.ones((100, 128), dtype=np.float32)
    >>> drift = DriftMMD(train_emb)

    Test incoming images for drift

    >>> test_emb = np.zeros((20, 128), dtype=np.float32)
    >>> result = drift.predict(test_emb)

    >>> print(f"Drift detected: {result.drifted}")
    Drift detected: True

    >>> print(f"Mean MMD statistic: {result.distance:.2f}")
    Mean MMD statistic: 1.26

    Using configuration:

    >>> config = DriftMMD.Config(p_val=0.01, n_permutations=200)
    >>> drift = DriftMMD(train_emb, config=config)
    """

    @dataclass
    class Config:
        """
        Configuration for DriftMMD detector.

        Attributes
        ----------
        p_val : float, default 0.05
            Significance threshold for statistical tests.
        sigma : Array or None, default None
            Bandwidth parameter(s) for the Gaussian RBF kernel.
        n_permutations : int, default 100
            Number of random permutations for the permutation test.
        permutation_batch_size : int or "auto", default "auto"
            Batch size for computing permutations.
        device : DeviceLike or None, default None
            Hardware device for computation.
        """

        p_val: float = 0.05
        sigma: Array | None = None
        n_permutations: int = 100
        permutation_batch_size: int | Literal["auto"] = "auto"
        device: DeviceLike | None = None

    def __init__(
        self,
        data: Array,
        p_val: float | None = None,
        update_strategy: UpdateStrategy | None = None,
        sigma: Array | None = None,
        n_permutations: int | None = None,
        permutation_batch_size: int | Literal["auto"] | None = None,
        device: DeviceLike | None = None,
        config: Config | None = None,
    ) -> None:
        # Store config or create default
        self.config: DriftMMD.Config = config or DriftMMD.Config()

        # Use config defaults if parameters not specified
        p_val = p_val if p_val is not None else self.config.p_val
        sigma = sigma if sigma is not None else self.config.sigma
        n_permutations = n_permutations if n_permutations is not None else self.config.n_permutations
        permutation_batch_size = (
            permutation_batch_size if permutation_batch_size is not None else self.config.permutation_batch_size
        )
        device = device if device is not None else self.config.device

        super().__init__(data, p_val, update_strategy)

        self.n_permutations = n_permutations  # nb of iterations through permutation test
        self.permutation_batch_size: int | Literal["auto"] = permutation_batch_size

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

    def score(self, data: Array) -> tuple[float, float, float]:
        """
        Compute the :term:`p-value<P-Value>` resulting from a permutation test using the maximum mean
        discrepancy as a distance measure between the reference data and the data to
        be tested.

        Parameters
        ----------
        data : Array
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
        mmd2 = mmd2_from_kernel_matrix(kernel_mat, n, False)

        # auto-detect batch size if set to "auto"
        batch_size = (
            _auto_detect_permutation_batch_size(kernel_mat.shape[0], self.n_permutations, kernel_mat.device)
            if self.permutation_batch_size == "auto"
            else self.permutation_batch_size
        )

        mmd2_permuted = mmd2_from_kernel_matrix(kernel_mat, n, False, self.n_permutations, batch_size)
        p_val = (mmd2 <= mmd2_permuted).float().mean()

        # compute distance threshold
        idx_threshold = int(self.p_val * len(mmd2_permuted))
        distance_threshold = torch.sort(mmd2_permuted, descending=True).values[idx_threshold]
        return float(p_val.item()), float(mmd2.item()), float(distance_threshold.item())

    @set_metadata
    @update_strategy
    def predict(self, data: Array) -> DriftMMDOutput:
        """
        Predict whether a batch of data has drifted from the reference data and then
        updates reference data using specified strategy.

        Parameters
        ----------
        data : Array
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
    kernel_mat: torch.Tensor,
    m: int,
    zero_diag: bool = True,
    n_permutations: int = 0,
    permutation_batch_size: int | None = None,
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
    zero_diag : bool, default True
        Whether to zero out the diagonal of the kernel matrix.
    n_permutations : int, default 0
        Number of random permutations to compute. If 0, computes the non-permuted
        MMD^2 and returns a scalar. If > 0, computes MMD^2 for this many random
        permutations in batch and returns tensor of shape (n_permutations,).
    permutation_batch_size : int or None, default None
        Batch size for computing permutations to reduce memory usage. When None,
        all permutations are computed at once (no batching). Set to a positive
        integer to enable batched computation. Useful for large kernel matrices
        or many permutations to avoid GPU OOM errors. Note: The DriftMMD class
        uses auto-detection when None; this parameter requires explicit values.

    Returns
    -------
    torch.Tensor
        MMD^2 between the samples. Scalar if n_permutations is 0,
        otherwise shape (n_permutations,).

    Notes
    -----
    This function computes an unbiased estimator of MMD^2 that can produce small
    negative values even though the true MMD^2 is theoretically non-negative.
    This occurs due to:

    - Finite sample variance: With limited samples, the unbiased estimator has
    variance that can push estimates slightly negative when the true MMD^2 is
    close to zero (e.g., under the null hypothesis of no distributional difference).
    - Numerical precision: Floating-point arithmetic errors can accumulate.

    These small negative values are statistically valid and should NOT be clamped
    to zero, as doing so would bias permutation tests that rely on the empirical
    distribution of MMD^2 values.
    """
    n = kernel_mat.shape[0] - m

    if zero_diag:
        kernel_mat = kernel_mat - torch.diag(kernel_mat.diag())

    if n_permutations > 0:
        # Determine if we should use batched computation
        if permutation_batch_size is not None and permutation_batch_size < n_permutations:
            # Batched permutation computation to reduce memory usage
            results = []
            for i in range(0, n_permutations, permutation_batch_size):
                batch_perms = min(permutation_batch_size, n_permutations - i)
                perm_indices = torch.argsort(
                    torch.rand(batch_perms, kernel_mat.shape[0], device=kernel_mat.device), dim=1
                )
                kernel_mat_batch = kernel_mat[perm_indices[:, :, None], perm_indices[:, None, :]]
                k_xx, k_yy, k_xy = (
                    kernel_mat_batch[:, :-m, :-m],
                    kernel_mat_batch[:, -m:, -m:],
                    kernel_mat_batch[:, -m:, :-m],
                )
                c_xx, c_yy, c_xy = 1 / (n * (n - 1)), 1 / (m * (m - 1)), 1 / (n * m)
                batch_result = (
                    c_xx * k_xx.sum(dim=(1, 2)) + c_yy * k_yy.sum(dim=(1, 2)) - 2.0 * c_xy * k_xy.sum(dim=(1, 2))
                )
                results.append(batch_result)
            return torch.cat(results)

        # Original single-batch computation
        perm_indices = torch.argsort(torch.rand(n_permutations, kernel_mat.shape[0], device=kernel_mat.device), dim=1)
        kernel_mat = kernel_mat[perm_indices[:, :, None], perm_indices[:, None, :]]
        k_xx, k_yy, k_xy = kernel_mat[:, :-m, :-m], kernel_mat[:, -m:, -m:], kernel_mat[:, -m:, :-m]
        c_xx, c_yy, c_xy = 1 / (n * (n - 1)), 1 / (m * (m - 1)), 1 / (n * m)
        return c_xx * k_xx.sum(dim=(1, 2)) + c_yy * k_yy.sum(dim=(1, 2)) - 2.0 * c_xy * k_xy.sum(dim=(1, 2))

    # Non-permuted single computation (no random permutation)
    k_xx, k_yy, k_xy = kernel_mat[:-m, :-m], kernel_mat[-m:, -m:], kernel_mat[-m:, :-m]
    c_xx, c_yy = 1 / (n * (n - 1)), 1 / (m * (m - 1))
    return c_xx * k_xx.sum() + c_yy * k_yy.sum() - 2.0 * k_xy.mean()
