"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from typing import Callable, Dict, Optional, Tuple, Union

import torch

from dataeval._internal.interop import ArrayLike, to_numpy

from .base import BaseDrift, UpdateStrategy, preprocess_x, update_x_ref
from .torch import GaussianRBF, get_device, mmd2_from_kernel_matrix


class DriftMMD(BaseDrift):
    """
    Maximum Mean Discrepancy (MMD) data drift detector using a permutation test.

    Parameters
    ----------
    x_ref : ArrayLike
        Data used as reference distribution.
    p_val : float, default 0.05
        p-value used for the significance of the permutation test.
    x_ref_preprocessed : bool, default False
        Whether the given reference data `x_ref` has been preprocessed yet. If
        `x_ref_preprocessed=True`, only the test data `x` will be preprocessed
        at prediction time. If `x_ref_preprocessed=False`, the reference data
        will also be preprocessed.
    preprocess_at_init : bool, default True
        Whether to preprocess the reference data when the detector is instantiated.
        Otherwise, the reference data will be preprocessed at prediction time. Only
        applies if `x_ref_preprocessed=False`.
    update_x_ref : Optional[UpdateStrategy], default None
        Reference data can optionally be updated using an UpdateStrategy class. Update
        using the last n instances seen by the detector with
        :py:class:`dataeval.detectors.LastSeenUpdateStrategy`
        or via reservoir sampling with
        :py:class:`dataeval.detectors.ReservoirSamplingUpdateStrategy`.
    preprocess_fn : Optional[Callable], default None
        Function to preprocess the data before computing the data drift metrics.
    kernel : Callable, default :py:class:`dataeval.detectors.GaussianRBF`
        Kernel used for the MMD computation, defaults to Gaussian RBF kernel.
    sigma : Optional[ArrayLike], default None
        Optionally set the GaussianRBF kernel bandwidth. Can also pass multiple
        bandwidth values as an array. The kernel evaluation is then averaged over
        those bandwidths.
    configure_kernel_from_x_ref : bool, default True
        Whether to already configure the kernel bandwidth from the reference data.
    n_permutations : int, default 100
        Number of permutations used in the permutation test.
    device : Optional[str], default None
        Device type used. The default None uses the GPU and falls back on CPU.
        Can be specified by passing either 'cuda', 'gpu' or 'cpu'.
    """

    def __init__(
        self,
        x_ref: ArrayLike,
        p_val: float = 0.05,
        x_ref_preprocessed: bool = False,
        update_x_ref: Optional[UpdateStrategy] = None,
        preprocess_fn: Optional[Callable[[ArrayLike], ArrayLike]] = None,
        kernel: Callable = GaussianRBF,
        sigma: Optional[ArrayLike] = None,
        configure_kernel_from_x_ref: bool = True,
        n_permutations: int = 100,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(x_ref, p_val, x_ref_preprocessed, update_x_ref, preprocess_fn)

        self.infer_sigma = configure_kernel_from_x_ref
        if configure_kernel_from_x_ref and isinstance(sigma, ArrayLike):
            self.infer_sigma = False

        self.n_permutations = n_permutations  # nb of iterations through permutation test

        # set device
        self.device = get_device(device)

        # initialize kernel
        sigma_tensor = torch.from_numpy(to_numpy(sigma)).to(self.device) if isinstance(sigma, ArrayLike) else None
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
    def score(self, x: ArrayLike) -> Tuple[float, float, float]:
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
        p-value obtained from the permutation test, the MMD^2 between the reference and
        test set, and the MMD^2 threshold above which drift is flagged.
        """
        x = to_numpy(x)
        x_ref = torch.from_numpy(self.x_ref).to(self.device)
        n = x.shape[0]
        kernel_mat = self._kernel_matrix(x_ref, torch.from_numpy(x).to(self.device))
        kernel_mat = kernel_mat - torch.diag(kernel_mat.diag())  # zero diagonal
        mmd2 = mmd2_from_kernel_matrix(kernel_mat, n, permute=False, zero_diag=False)
        mmd2_permuted = torch.Tensor(
            [mmd2_from_kernel_matrix(kernel_mat, n, permute=True, zero_diag=False) for _ in range(self.n_permutations)]
        )
        mmd2, mmd2_permuted = mmd2.cpu(), mmd2_permuted.cpu()
        p_val = (mmd2 <= mmd2_permuted).float().mean()
        # compute distance threshold
        idx_threshold = int(self.p_val * len(mmd2_permuted))
        distance_threshold = torch.sort(mmd2_permuted, descending=True).values[idx_threshold]
        return p_val.numpy().item(), mmd2.numpy().item(), distance_threshold.numpy()

    @preprocess_x
    @update_x_ref
    def predict(
        self,
        x: ArrayLike,
    ) -> Dict[str, Union[int, float]]:
        """
        Predict whether a batch of data has drifted from the reference data and then
        updates reference data using specified strategy.

        Parameters
        ----------
        x : ArrayLike
            Batch of instances.

        Returns
        -------
        Dictionary containing the drift prediction, p-value, threshold and MMD metric.
        """
        # compute drift scores
        p_val, dist, distance_threshold = self.score(x)
        drift_pred = int(p_val < self.p_val)

        # populate drift dict
        return {
            "is_drift": drift_pred,
            "p_val": p_val,
            "threshold": self.p_val,
            "distance": dist,
            "distance_threshold": distance_threshold,
        }
