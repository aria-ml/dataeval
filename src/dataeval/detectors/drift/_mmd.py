"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

__all__ = []

from typing import Callable

import torch

from dataeval.config import DeviceLike, get_device
from dataeval.detectors.drift._base import BaseDrift, UpdateStrategy, preprocess_x, update_x_ref
from dataeval.detectors.drift._torch import GaussianRBF, mmd2_from_kernel_matrix
from dataeval.outputs import DriftMMDOutput
from dataeval.outputs._base import set_metadata
from dataeval.typing import ArrayLike


class DriftMMD(BaseDrift):
    """
    :term:`Maximum Mean Discrepancy (MMD) Drift Detection` algorithm \
    using a permutation test.

    Parameters
    ----------
    x_ref : ArrayLike
        Data used as reference distribution.
    p_val : float or None, default 0.05
        :term:`P-value` used for significance of the statistical test for each feature.
        If the FDR correction method is used, this corresponds to the acceptable
        q-value.
    x_ref_preprocessed : bool, default False
        Whether the given reference data ``x_ref`` has been preprocessed yet.
        If ``True``, only the test data ``x`` will be preprocessed at prediction time.
        If ``False``, the reference data will also be preprocessed.
    update_x_ref : UpdateStrategy or None, default None
        Reference data can optionally be updated using an UpdateStrategy class. Update
        using the last n instances seen by the detector with LastSeenUpdateStrategy
        or via reservoir sampling with ReservoirSamplingUpdateStrategy.
    preprocess_fn : Callable or None, default None
        Function to preprocess the data before computing the data drift metrics.
        Typically a :term:`dimensionality reduction<Dimensionality Reduction>` technique.
    sigma : ArrayLike or None, default None
        Optionally set the internal GaussianRBF kernel bandwidth. Can also pass multiple
        bandwidth values as an array. The kernel evaluation is then averaged over
        those bandwidths.
    configure_kernel_from_x_ref : bool, default True
        Whether to already configure the kernel bandwidth from the reference data.
    n_permutations : int, default 100
        Number of permutations used in the permutation test.
    device : DeviceLike or None, default None
        The hardware device to use if specified, otherwise uses the DataEval
        default or torch default.

    Example
    -------
    >>> from functools import partial
    >>> from dataeval.detectors.drift import preprocess_drift

    Use a preprocess function to encode images before testing for drift

    >>> preprocess_fn = partial(preprocess_drift, model=encoder, batch_size=64)
    >>> drift = DriftMMD(train_images, preprocess_fn=preprocess_fn)

    Test incoming images for drift

    >>> drift.predict(test_images).drifted
    True
    """

    def __init__(
        self,
        x_ref: ArrayLike,
        p_val: float = 0.05,
        x_ref_preprocessed: bool = False,
        update_x_ref: UpdateStrategy | None = None,
        preprocess_fn: Callable[..., ArrayLike] | None = None,
        sigma: ArrayLike | None = None,
        configure_kernel_from_x_ref: bool = True,
        n_permutations: int = 100,
        device: DeviceLike | None = None,
    ) -> None:
        super().__init__(x_ref, p_val, x_ref_preprocessed, update_x_ref, preprocess_fn)

        self._infer_sigma = configure_kernel_from_x_ref
        if configure_kernel_from_x_ref and sigma is not None:
            self._infer_sigma = False

        self.n_permutations = n_permutations  # nb of iterations through permutation test

        # set device
        self.device: torch.device = get_device(device)

        # initialize kernel
        sigma_tensor = torch.as_tensor(sigma, device=self.device) if sigma is not None else None
        self._kernel = GaussianRBF(sigma_tensor).to(self.device)

        # compute kernel matrix for the reference data
        if self._infer_sigma or isinstance(sigma_tensor, torch.Tensor):
            x = torch.as_tensor(self.x_ref, device=self.device)
            self._k_xx = self._kernel(x, x, infer_sigma=self._infer_sigma)
            self._infer_sigma = False
        else:
            self._k_xx, self._infer_sigma = None, True

    def _kernel_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute and return full kernel matrix between arrays x and y."""
        k_xy = self._kernel(x, y, self._infer_sigma)
        k_xx = self._k_xx if self._k_xx is not None and self.update_x_ref is None else self._kernel(x, x)
        k_yy = self._kernel(y, y)
        kernel_mat = torch.cat([torch.cat([k_xx, k_xy], 1), torch.cat([k_xy.T, k_yy], 1)], 0)
        return kernel_mat

    @preprocess_x
    def score(self, x: ArrayLike) -> tuple[float, float, float]:
        """
        Compute the :term:`p-value<P-Value>` resulting from a permutation test using the maximum mean
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
            and MMD^2 threshold above which :term:`drift<Drift>` is flagged
        """
        x_ref = torch.as_tensor(self.x_ref, device=self.device)
        x_test = torch.as_tensor(x, device=self.device)
        n = x_test.shape[0]
        kernel_mat = self._kernel_matrix(x_ref, x_test)
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
            Output class containing the :term:`drift<Drift>` prediction, :term:`p-value<P-Value>`,
            threshold and MMD metric.
        """
        # compute drift scores
        p_val, dist, distance_threshold = self.score(x)
        drift_pred = bool(p_val < self.p_val)

        # populate drift dict
        return DriftMMDOutput(drift_pred, self.p_val, p_val, dist, distance_threshold)
