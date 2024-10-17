"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from numpy.typing import ArrayLike

from dataeval._internal.interop import as_numpy
from dataeval._internal.output import set_metadata

from .base import BaseDrift, DriftBaseOutput, UpdateStrategy, preprocess_x, update_x_ref
from .torch import GaussianRBF, get_device, mmd2_from_kernel_matrix


@dataclass(frozen=True)
class DriftMMDOutput(DriftBaseOutput):
    """
    Output class for :class:`DriftMMD` drift detector

    Attributes
    ----------
    is_drift : bool
        Drift prediction for the images
    threshold : float
        P-value used for significance of the permutation test
    p_val : float
        P-value obtained from the permutation test
    distance : float
        MMD^2 between the reference and test set
    distance_threshold : float
        MMD^2 threshold above which drift is flagged
    """

    # is_drift: bool
    # threshold: float
    # p_val: float
    # distance: float
    distance_threshold: float


class DriftMMD(BaseDrift):
    """
    Maximum Mean Discrepancy (MMD) data drift detector using a permutation test.

    Parameters
    ----------
    x_ref : ArrayLike
        Data used as reference distribution.
    p_val : float | None, default 0.05
        p-value used for significance of the statistical test for each feature.
        If the FDR correction method is used, this corresponds to the acceptable
        q-value.
    x_ref_preprocessed : bool, default False
        Whether the given reference data ``x_ref`` has been preprocessed yet.
        If ``True``, only the test data ``x`` will be preprocessed at prediction time.
        If ``False``, the reference data will also be preprocessed.
    update_x_ref : UpdateStrategy | None, default None
        Reference data can optionally be updated using an UpdateStrategy class. Update
        using the last n instances seen by the detector with LastSeenUpdateStrategy
        or via reservoir sampling with ReservoirSamplingUpdateStrategy.
    preprocess_fn : Callable | None, default None
        Function to preprocess the data before computing the data drift metrics.
        Typically a dimensionality reduction technique.
    kernel : Callable, default GaussianRBF
        Kernel used for the MMD computation, defaults to Gaussian RBF kernel.
    sigma : ArrayLike | None, default None
        Optionally set the GaussianRBF kernel bandwidth. Can also pass multiple
        bandwidth values as an array. The kernel evaluation is then averaged over
        those bandwidths.
    configure_kernel_from_x_ref : bool, default True
        Whether to already configure the kernel bandwidth from the reference data.
    n_permutations : int, default 100
        Number of permutations used in the permutation test.
    device : str | None, default None
        Device type used. The default None uses the GPU and falls back on CPU.
        Can be specified by passing either 'cuda', 'gpu' or 'cpu'.
    """

    def __init__(
        self,
        x_ref: ArrayLike,
        p_val: float = 0.05,
        x_ref_preprocessed: bool = False,
        update_x_ref: UpdateStrategy | None = None,
        preprocess_fn: Callable[[ArrayLike], ArrayLike] | None = None,
        kernel: Callable = GaussianRBF,
        sigma: ArrayLike | None = None,
        configure_kernel_from_x_ref: bool = True,
        n_permutations: int = 100,
        device: str | None = None,
    ) -> None:
        super().__init__(x_ref, p_val, x_ref_preprocessed, update_x_ref, preprocess_fn)

        self.infer_sigma = configure_kernel_from_x_ref
        if configure_kernel_from_x_ref and sigma is not None:
            self.infer_sigma = False

        self.n_permutations = n_permutations  # nb of iterations through permutation test

        # set device
        self.device = get_device(device)

        # initialize kernel
        sigma_tensor = torch.from_numpy(as_numpy(sigma)).to(self.device) if sigma is not None else None
        self.kernel = kernel(sigma_tensor).to(self.device) if kernel == GaussianRBF else kernel

        # compute kernel matrix for the reference data
        if self.infer_sigma or isinstance(sigma_tensor, torch.Tensor):
            x = torch.from_numpy(self.x_ref).to(self.device)
            self.k_xx = self.kernel(x, x, infer_sigma=self.infer_sigma)
            self.infer_sigma = False
        else:
            self.k_xx, self.infer_sigma = None, True

    def _kernel_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute and return full kernel matrix between arrays x and y."""
        k_xy = self.kernel(x, y, self.infer_sigma)
        k_xx = self.k_xx if self.k_xx is not None and self.update_x_ref is None else self.kernel(x, x)
        k_yy = self.kernel(y, y)
        kernel_mat = torch.cat([torch.cat([k_xx, k_xy], 1), torch.cat([k_xy.T, k_yy], 1)], 0)
        return kernel_mat

    @preprocess_x
    def score(self, x: ArrayLike) -> tuple[float, float, float]:
        """
        Compute the p-value resulting from a permutation test using the maximum mean
        discrepancy as a distance measure between the reference data and the data to
        be tested.

        Parameters
        ----------
        x : ArrayLike
            Batch of instances.

        Returns
        -------
        tuple(float, float, float)
            p-value obtained from the permutation test, MMD^2 between the reference and test set,
            and MMD^2 threshold above which drift is flagged
        """
        x = as_numpy(x)
        x_ref = torch.from_numpy(self.x_ref).to(self.device)
        n = x.shape[0]
        kernel_mat = self._kernel_matrix(x_ref, torch.from_numpy(x).to(self.device))
        kernel_mat = kernel_mat - torch.diag(kernel_mat.diag())  # zero diagonal
        mmd2 = mmd2_from_kernel_matrix(kernel_mat, n, permute=False, zero_diag=False)
        mmd2_permuted = torch.Tensor(
            [mmd2_from_kernel_matrix(kernel_mat, n, permute=True, zero_diag=False) for _ in range(self.n_permutations)]
        )
        mmd2, mmd2_permuted = mmd2.detach().cpu(), mmd2_permuted.detach().cpu()
        p_val = (mmd2 <= mmd2_permuted).float().mean()
        # compute distance threshold
        idx_threshold = int(self.p_val * len(mmd2_permuted))
        distance_threshold = torch.sort(mmd2_permuted, descending=True).values[idx_threshold]
        return p_val.numpy().item(), mmd2.numpy().item(), distance_threshold.numpy()

    @set_metadata("dataeval.detectors")
    @preprocess_x
    @update_x_ref
    def predict(self, x: ArrayLike) -> DriftMMDOutput:
        """
        Predict whether a batch of data has drifted from the reference data and then
        updates reference data using specified strategy.

        Parameters
        ----------
        x : ArrayLike
            Batch of instances.

        Returns
        -------
        DriftMMDOutput
            Output class containing the drift prediction, p-value, threshold and MMD metric.
        """
        # compute drift scores
        p_val, dist, distance_threshold = self.score(x)
        drift_pred = bool(p_val < self.p_val)

        # populate drift dict
        return DriftMMDOutput(drift_pred, self.p_val, p_val, dist, distance_threshold)
