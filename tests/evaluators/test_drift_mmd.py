"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from itertools import product

import numpy as np
import pytest
import torch
import torch.nn as nn

# from maite_datasets import to_image_classification_dataset
from dataeval.config import get_device
from dataeval.data._embeddings import Embeddings
from dataeval.evaluators.drift._mmd import DriftMMD, GaussianRBF, _squared_pairwise_distance, mmd2_from_kernel_matrix
from dataeval.evaluators.drift.updates import LastSeenUpdate, ReservoirSamplingUpdate


class HiddenOutput(nn.Module):
    def __init__(
        self,
        model: nn.Module | nn.Sequential,
        layer: int = -1,
        flatten: bool = False,
    ) -> None:
        super().__init__()
        layers = list(model.children())[:layer]
        if flatten:
            layers += [nn.Flatten()]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MyModel(nn.Module):
    def __init__(self, n_shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.dense0 = nn.Flatten()
        self.dense1 = nn.Linear(math.prod(n_shape), 20)
        self.dense2 = nn.Linear(20, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.ReLU()(self.dense1(self.dense0(x)))
        return self.dense2(x)


def get_embeddings(
    n: int,
    n_shape: tuple[int, int, int],
    n_classes: int,
    model: nn.Module,
    get_mock_ic_dataset: Callable,
    RNG: np.random.Generator,
) -> Embeddings:
    images = RNG.standard_normal(n * math.prod(n_shape)).reshape(n, *n_shape).astype(np.float32)
    dataset = get_mock_ic_dataset(images, RNG.integers(n_classes, size=n).tolist())
    return Embeddings(dataset=dataset, model=model, batch_size=n, device="cpu")


@pytest.mark.required
class TestMMDDrift:
    n, n_hidden, n_classes = 100, 10, 5
    n_shape = (1, 16, 16)
    model = [nn.Identity(), HiddenOutput(MyModel(n_shape), layer=-1)]
    update_strategy = [LastSeenUpdate(750), ReservoirSamplingUpdate(750), None]
    n_permutations = [10]
    sigma = [np.array([[1, 0], [0, 1]]), None]
    tests_mmddrift = list(product(model, n_permutations, update_strategy, sigma))
    n_tests = len(tests_mmddrift)

    @pytest.fixture(scope="class")
    def mmd_params(self, request):
        return self.tests_mmddrift[request.param]

    @pytest.mark.parametrize("mmd_params", list(range(n_tests)), indirect=True)
    def test_mmd(self, mmd_params, get_mock_ic_dataset, RNG):
        model, n_permutations, update_strategy, sigma = mmd_params

        x_ref = get_embeddings(self.n, self.n_shape, self.n_classes, model, get_mock_ic_dataset, RNG)
        cd = DriftMMD(
            data=x_ref,
            p_val=0.05,
            update_strategy=update_strategy,
            sigma=sigma,
            n_permutations=n_permutations,
            device="cpu",
        )
        preds = cd.predict(x_ref)
        assert not preds.drifted and preds.p_val >= cd.p_val
        if isinstance(update_strategy, dict):
            k = list(update_strategy.keys())[0]
            assert cd.n == self.n + self.n
            assert cd._data.shape[0] == min(update_strategy[k], self.n + self.n)  # type: ignore

        preds = cd.predict(get_embeddings(self.n, self.n_shape, self.n_classes, model, get_mock_ic_dataset, RNG))
        if preds.drifted:
            assert preds.p_val < preds.threshold == cd.p_val
            assert preds.distance > preds.distance_threshold
        else:
            assert preds.p_val >= preds.threshold == cd.p_val
            assert preds.distance <= preds.distance_threshold

    def test_permutation_pvalue_not_degenerate(self, RNG):
        """Regression test: p-value should be between 0 and 1, not exactly 0 or 1."""
        x_ref = RNG.normal(size=(50, 10)).astype(np.float32)
        x_test = RNG.normal(size=(50, 10)).astype(np.float32)

        cd = DriftMMD(data=x_ref, p_val=0.05, n_permutations=100, device="cpu")
        p_val, _, _ = cd.score(x_test)

        assert 0 < p_val < 1


@pytest.mark.required
class TestSquaredPairwiseDistance:
    n_features = [2, 5]
    n_instances = [(100, 100), (100, 75)]
    tests_pairwise = list(product(n_features, n_instances))
    n_tests_pairwise = len(tests_pairwise)

    @pytest.fixture(scope="class")
    def pairwise_params(self, request):
        return self.tests_pairwise[request.param]

    @pytest.mark.parametrize("pairwise_params", list(range(n_tests_pairwise)), indirect=True)
    def test_pairwise(self, pairwise_params, RNG):
        n_features, n_instances = pairwise_params
        xshape, yshape = (n_instances[0], n_features), (n_instances[1], n_features)
        x = torch.from_numpy(RNG.random(xshape).astype("float32"))
        y = torch.from_numpy(RNG.random(yshape).astype("float32"))

        dist_xx = _squared_pairwise_distance(x, x).numpy()
        dist_xy = _squared_pairwise_distance(x, y).numpy()

        assert dist_xx.shape == (xshape[0], xshape[0])
        assert dist_xy.shape == n_instances
        np.testing.assert_almost_equal(dist_xx.trace(), 0.0, decimal=5)


@pytest.mark.required
class TestMMDKernelMatrix:
    n_features = [2, 5]
    n_instances = [(100, 100), (100, 75)]
    batch_size = [1, 5]
    tests_bckm = list(product(n_features, n_instances, batch_size))
    n_tests_bckm = len(tests_bckm)

    n = [10, 100]
    m = [10, 100]
    n_perms = [1, 0]
    zero_diag = [True, False]
    tests_mmd_from_kernel_matrix = list(product(n, m, n_perms, zero_diag))
    n_tests_mmd_from_kernel_matrix = len(tests_mmd_from_kernel_matrix)

    @pytest.fixture(scope="class")
    def mmd_from_kernel_matrix_params(self, request):
        return self.tests_mmd_from_kernel_matrix[request.param]

    @pytest.mark.parametrize(
        "mmd_from_kernel_matrix_params",
        list(range(n_tests_mmd_from_kernel_matrix)),
        indirect=True,
    )
    def test_mmd_from_kernel_matrix(self, mmd_from_kernel_matrix_params, RNG):
        n, m, n_perms, zero_diag = mmd_from_kernel_matrix_params
        n_tot = n + m
        shape = (n_tot, n_tot)
        kernel_mat = RNG.uniform(0, 1, size=shape)
        kernel_mat_2 = kernel_mat.copy()
        kernel_mat_2[-m:, :-m] = 1.0
        kernel_mat_2[:-m, -m:] = 1.0
        kernel_mat = torch.from_numpy(kernel_mat)
        kernel_mat_2 = torch.from_numpy(kernel_mat_2)
        if not zero_diag:
            kernel_mat -= torch.diag(kernel_mat.diag())
            kernel_mat_2 -= torch.diag(kernel_mat_2.diag())
        mmd = mmd2_from_kernel_matrix(kernel_mat, m, zero_diag=zero_diag, n_permutations=n_perms)
        mmd_2 = mmd2_from_kernel_matrix(kernel_mat_2, m, zero_diag=zero_diag, n_permutations=n_perms)
        if not n_perms:
            assert mmd_2.numpy() < mmd.numpy()

    # New tests for batched permutations
    n_batched = [10, 100]
    m_batched = [10, 100]
    n_permutations = [1, 10, 100]
    zero_diag_batched = [True, False]
    tests_mmd_batched = list(product(n_batched, m_batched, n_permutations, zero_diag_batched))
    n_tests_mmd_batched = len(tests_mmd_batched)

    @pytest.fixture(scope="class")
    def mmd_batched_params(self, request):
        return self.tests_mmd_batched[request.param]

    @pytest.mark.parametrize(
        "mmd_batched_params",
        list(range(n_tests_mmd_batched)),
        indirect=True,
    )
    def test_mmd_batched_permutations(self, mmd_batched_params):
        """Test batched permutation computation returns correct shape and values."""
        n, m, n_perms, zero_diag = mmd_batched_params
        n_tot = n + m

        # Create a proper kernel matrix using RBF kernel to ensure valid MMD^2 properties
        x = torch.randn(n_tot, 10)
        dist = torch.cdist(x, x, p=2) ** 2
        kernel_mat = torch.exp(-dist / (2 * 1.0**2))

        # Note: Don't zero the diagonal here - let the function handle it based on zero_diag parameter

        # Compute batched
        mmd_batched = mmd2_from_kernel_matrix(kernel_mat, m, zero_diag=zero_diag, n_permutations=n_perms)

        # Check shape
        assert mmd_batched.shape == (n_perms,)

        # MMD^2 should be close to non-negative for valid kernel matrices
        # Allow small negative values due to numerical precision and unbiased estimator
        assert (mmd_batched > -0.05).all(), (
            f"MMD^2 values should be approximately non-negative, got min={mmd_batched.min()}"
        )

    def test_mmd_batched_vs_loop(self):
        """Test batched implementation matches loop-based implementation."""
        n, m, n_perms = 50, 30, 20
        kernel_mat = torch.rand(n + m, n + m)
        kernel_mat -= torch.diag(kernel_mat.diag())

        # Set seed for reproducibility
        torch.manual_seed(42)
        mmd_batched = mmd2_from_kernel_matrix(kernel_mat, m, zero_diag=False, n_permutations=n_perms)

        # Compute with loop (old method)
        torch.manual_seed(42)
        mmd_loop = torch.tensor(
            [mmd2_from_kernel_matrix(kernel_mat, m, zero_diag=False, n_permutations=1).item() for _ in range(n_perms)]
        )

        # Should match within numerical precision
        torch.testing.assert_close(mmd_batched, mmd_loop, rtol=1e-5, atol=1e-7)

    def test_mmd_batched_device_consistency(self):
        """Test batched computation works on different devices."""
        n, m, n_perms = 20, 15, 10
        kernel_mat_cpu = torch.rand(n + m, n + m)

        # CPU computation
        mmd_cpu = mmd2_from_kernel_matrix(kernel_mat_cpu, m, zero_diag=True, n_permutations=n_perms)
        assert mmd_cpu.device.type == "cpu"

        # GPU computation (if available)
        if torch.cuda.is_available():
            kernel_mat_gpu = kernel_mat_cpu.cuda()
            mmd_gpu = mmd2_from_kernel_matrix(kernel_mat_gpu, m, zero_diag=True, n_permutations=n_perms)
            assert mmd_gpu.device.type == "cuda"

    def test_mmd_batched_different_permutations(self):
        """Test that batched permutations produce different results."""
        n, m, n_perms = 50, 30, 50
        kernel_mat = torch.rand(n + m, n + m)

        mmd_batched = mmd2_from_kernel_matrix(kernel_mat, m, zero_diag=True, n_permutations=n_perms)

        # With random permutations, should have variation in results
        # (not all identical, which would indicate a bug)
        assert mmd_batched.std() > 0


@pytest.mark.required
@pytest.mark.parametrize("device", [None, torch.device("cpu"), "cpu", "cuda"])
def test_drift_get_device(device):
    assert isinstance(get_device(device), torch.device)


@pytest.mark.required
def test_gaussianrbf_forward_valueerror():
    g = GaussianRBF(trainable=True)
    with pytest.raises(ValueError):
        g.forward(np.zeros((2, 2)), np.zeros((2, 2)), infer_sigma=True)
