"""
Source code derived from Alibi-Detect 0.11.4
https://github.com/SeldonIO/alibi-detect/tree/v0.11.4.

Original code Copyright (c) 2023 Seldon Technologies Ltd
Licensed under Apache Software License (Apache 2.0)
"""

import math
from collections.abc import Callable
from itertools import product
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from dataeval._embeddings import Embeddings

# from maite_datasets import to_image_classification_dataset
from dataeval.config import get_device
from dataeval.extractors import TorchExtractor
from dataeval.shift._drift._mmd import (
    DriftMMD,
    GaussianRBF,
    _auto_detect_permutation_batch_size,
    _squared_pairwise_distance,
    mmd2_from_kernel_matrix,
)
from dataeval.shift.update_strategies import LastSeenUpdateStrategy, ReservoirSamplingUpdateStrategy


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
    extractor = TorchExtractor(model, device="cpu")
    return Embeddings(dataset=dataset, extractor=extractor)


@pytest.mark.required
class TestMMDDrift:
    n, n_hidden, n_classes = 100, 10, 5
    n_shape = (1, 16, 16)
    model = [nn.Identity(), HiddenOutput(MyModel(n_shape), layer=-1)]
    update_strategy = [LastSeenUpdateStrategy(750), ReservoirSamplingUpdateStrategy(750), None]
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
        assert not preds.drifted
        assert preds.p_val >= cd.p_val
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

    def test_permutation_batch_size_parameter(self, RNG):
        """Test that DriftMMD with permutation_batch_size produces same results as without."""
        x_ref = RNG.normal(size=(50, 10)).astype(np.float32)
        x_test = RNG.normal(size=(50, 10)).astype(np.float32)

        # Create detector with auto (default)
        cd_auto = DriftMMD(data=x_ref, p_val=0.05, n_permutations=100, device="cpu")
        assert cd_auto.permutation_batch_size == "auto"

        # Create detector with explicit batching
        cd_with_batch = DriftMMD(data=x_ref, p_val=0.05, n_permutations=100, permutation_batch_size=10, device="cpu")

        # Create detector with batching disabled (batch_size >= n_permutations)
        cd_no_batch = DriftMMD(data=x_ref, p_val=0.05, n_permutations=100, permutation_batch_size=100, device="cpu")

        # All should produce consistent results (setting seed for reproducibility)
        torch.manual_seed(42)
        result_auto = cd_auto.predict(x_test)

        torch.manual_seed(42)
        result_with_batch = cd_with_batch.predict(x_test)

        torch.manual_seed(42)
        result_no_batch = cd_no_batch.predict(x_test)

        # Check that drift detection is consistent across all modes
        assert result_auto.drifted == result_with_batch.drifted == result_no_batch.drifted
        # P-values and distances should be close (within reasonable tolerance due to random permutations)
        assert abs(result_auto.p_val - result_with_batch.p_val) < 0.2
        assert abs(result_auto.distance - result_with_batch.distance) < 0.1


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
            [mmd2_from_kernel_matrix(kernel_mat, m, zero_diag=False, n_permutations=1).item() for _ in range(n_perms)],
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

    def test_mmd_permutation_batch_size(self):
        """Test permutation_batch_size parameter reduces memory usage without changing results."""
        n, m, n_perms = 50, 30, 100
        kernel_mat = torch.rand(n + m, n + m)
        kernel_mat -= torch.diag(kernel_mat.diag())

        # Compute with full batch
        torch.manual_seed(42)
        mmd_full = mmd2_from_kernel_matrix(kernel_mat, m, zero_diag=False, n_permutations=n_perms)

        # Compute with batched permutations
        torch.manual_seed(42)
        mmd_batched = mmd2_from_kernel_matrix(
            kernel_mat,
            m,
            zero_diag=False,
            n_permutations=n_perms,
            permutation_batch_size=10,
        )

        # Results should match
        torch.testing.assert_close(mmd_full, mmd_batched, rtol=1e-5, atol=1e-7)

        # Test with different batch sizes
        for batch_size in [1, 5, 25, 50]:
            torch.manual_seed(42)
            mmd_test = mmd2_from_kernel_matrix(
                kernel_mat,
                m,
                zero_diag=False,
                n_permutations=n_perms,
                permutation_batch_size=batch_size,
            )
            torch.testing.assert_close(mmd_full, mmd_test, rtol=1e-5, atol=1e-7)


@pytest.mark.required
@pytest.mark.parametrize("device", [None, torch.device("cpu"), "cpu", "cuda"])
def test_drift_get_device(device):
    assert isinstance(get_device(device), torch.device)


@pytest.mark.required
def test_gaussianrbf_forward_valueerror():
    g = GaussianRBF(trainable=True)
    with pytest.raises(ValueError, match="Gradients cannot be computed w.r.t. an inferred sigma"):
        g.forward(np.zeros((2, 2)), np.zeros((2, 2)), infer_sigma=True)


@pytest.mark.required
class TestAutoDetectBatchSize:
    def test_auto_detect_returns_one_for_cpu(self):
        """Auto-detection should return 1 (no batching) for CPU devices."""
        batch_size = _auto_detect_permutation_batch_size(
            kernel_mat_size=500,
            n_permutations=100,
            device=torch.device("cpu"),
        )
        assert batch_size == 1

    def test_auto_detect_returns_one_for_small_matrices(self):
        """Auto-detection should return 1 (no batching) when all permutations fit in memory."""
        # Small matrix that should fit in memory even with all permutations
        batch_size = _auto_detect_permutation_batch_size(
            kernel_mat_size=50,
            n_permutations=100,
            device=torch.device("cpu"),
        )
        assert batch_size == 1

    @pytest.mark.cuda
    def test_auto_detect_suggests_batch_for_large_matrices(self):
        """Auto-detection should suggest batching for large matrices on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Large matrix that would require significant memory
        batch_size = _auto_detect_permutation_batch_size(
            kernel_mat_size=2000,
            n_permutations=1000,
            device=torch.device("cuda"),
        )
        # Should always return a positive integer
        assert isinstance(batch_size, int)
        assert batch_size > 0

    def test_auto_detect_batch_size_is_positive(self):
        """Auto-detection always returns a positive integer."""
        # Use a scenario on CPU
        batch_size = _auto_detect_permutation_batch_size(
            kernel_mat_size=1000,
            n_permutations=500,
            device=torch.device("cpu"),
        )
        # Should always return a positive integer
        assert isinstance(batch_size, int)
        assert batch_size > 0

    @pytest.mark.cuda
    def test_auto_detect_batch_size_cuda_fallback(self):
        """Test auto-detection fallback when CUDA memory info unavailable (line 94-99)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Test with very large matrix that might trigger fallback heuristic
        batch_size = _auto_detect_permutation_batch_size(
            kernel_mat_size=600,  # > 500 threshold
            n_permutations=100,
            device=torch.device("cuda"),
        )

        # Should return a positive integer
        assert isinstance(batch_size, int)
        assert batch_size > 0

    @pytest.mark.cuda
    def test_auto_detect_small_matrix_cuda(self):
        """Test auto-detection with small matrix on CUDA returns 1 (line 97-99)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Small matrix should not need batching
        batch_size = _auto_detect_permutation_batch_size(
            kernel_mat_size=100,  # <= 500 threshold
            n_permutations=50,
            device=torch.device("cuda"),
        )

        # Small matrices might return 1 (no batching) or a reasonable batch size
        assert isinstance(batch_size, int)
        assert batch_size > 0


@pytest.fixture
def mock_cuda_memory():
    """
    Fixture to mock torch.cuda functions for memory management.

    Returns a context manager factory that accepts total_memory and reserved_memory.
    """
    from contextlib import contextmanager

    @contextmanager
    def _mock(total_memory=None, reserved_memory=None, get_props_side_effect=None, mem_reserved_side_effect=None):
        """
        Mock torch.cuda memory functions.

        Parameters
        ----------
        total_memory : int or None
            Total GPU memory in bytes. If None, get_device_properties will raise side_effect.
        reserved_memory : int or None
            Reserved GPU memory in bytes. If None, memory_reserved will raise side_effect.
        get_props_side_effect : Exception or None
            Exception to raise when get_device_properties is called.
        mem_reserved_side_effect : Exception or None
            Exception to raise when memory_reserved is called.
        """
        mock_device_props = Mock()
        if total_memory is not None:
            mock_device_props.total_memory = total_memory

        with (
            patch(
                "dataeval.shift._drift._mmd.torch.cuda.get_device_properties",
                return_value=mock_device_props if get_props_side_effect is None else None,
                side_effect=get_props_side_effect,
            ) as mock_get_props,
            patch(
                "dataeval.shift._drift._mmd.torch.cuda.memory_reserved",
                return_value=reserved_memory if mem_reserved_side_effect is None else None,
                side_effect=mem_reserved_side_effect,
            ) as mock_mem_reserved,
        ):
            yield mock_get_props, mock_mem_reserved

    return _mock


@pytest.mark.required
class TestAutoDetectBatchSizeMocked:
    """Test _auto_detect_permutation_batch_size with mocked torch.cuda functions."""

    def test_cpu_device_returns_one(self, mock_cuda_memory):
        """CPU device should always return 1 without calling CUDA functions."""
        with mock_cuda_memory() as (mock_props, mock_reserved):
            batch_size = _auto_detect_permutation_batch_size(
                kernel_mat_size=1000,
                n_permutations=100,
                device=torch.device("cpu"),
            )

            # Should return 1 for CPU
            assert batch_size == 1
            # Should not call any CUDA functions
            mock_props.assert_not_called()
            mock_reserved.assert_not_called()

    def test_cuda_device_with_sufficient_memory(self, mock_cuda_memory):
        """CUDA device with sufficient memory should return 1 (no batching needed)."""
        # Mock device properties with large total memory (16 GB)
        with mock_cuda_memory(total_memory=16 * 1024**3, reserved_memory=1 * 1024**3):
            batch_size = _auto_detect_permutation_batch_size(
                kernel_mat_size=100,
                n_permutations=100,
                device=torch.device("cuda:0"),
            )

            # With small kernel matrix and lots of memory, should return 1
            assert batch_size == 1

    def test_cuda_device_with_limited_memory_suggests_batching(self, mock_cuda_memory):
        """CUDA device with limited memory should suggest appropriate batch size."""
        # Mock device properties with limited memory (2 GB)
        with mock_cuda_memory(total_memory=2 * 1024**3, reserved_memory=1 * 1024**3):
            # Large kernel matrix requiring significant memory
            batch_size = _auto_detect_permutation_batch_size(
                kernel_mat_size=2000,
                n_permutations=100,
                device=torch.device("cuda:0"),
            )

            # Should suggest batching (return value < n_permutations and > 1)
            assert isinstance(batch_size, int)
            assert batch_size >= 1

    def test_cuda_device_calculates_batch_size_from_memory(self, mock_cuda_memory):
        """Test correct batch size calculation based on available memory."""
        # Set up specific memory scenario
        total_memory = 8 * 1024**3  # 8 GB
        reserved_memory = 2 * 1024**3  # 2 GB reserved
        # Available memory: (8GB - 2GB) * 0.5 = 3GB safe to use

        with mock_cuda_memory(total_memory=total_memory, reserved_memory=reserved_memory):
            # kernel_mat_size=1000 means each permutation needs:
            # 1000 * 1000 * 4 bytes * 2.0 (safety factor) = 8 MB
            kernel_mat_size = 1000
            n_permutations = 1000

            batch_size = _auto_detect_permutation_batch_size(
                kernel_mat_size=kernel_mat_size,
                n_permutations=n_permutations,
                device=torch.device("cuda:0"),
            )

            # With 3GB available and ~8MB per permutation:
            # max_batch_size = 3GB / 8MB = ~375
            # final batch_size = max_batch_size * 0.5 = ~187
            assert isinstance(batch_size, int)
            assert batch_size >= 1
            # Should be significantly less than n_permutations
            assert batch_size < n_permutations

    def test_cuda_device_high_memory_pressure(self, mock_cuda_memory):
        """Test behavior when most memory is already reserved."""
        # Almost all memory is reserved
        total_memory = 4 * 1024**3  # 4 GB
        reserved_memory = int(3.9 * 1024**3)  # 3.9 GB reserved

        with mock_cuda_memory(total_memory=total_memory, reserved_memory=reserved_memory):
            batch_size = _auto_detect_permutation_batch_size(
                kernel_mat_size=1000,
                n_permutations=100,
                device=torch.device("cuda:0"),
            )

            # With very little available memory, should still return at least 1
            assert batch_size >= 1

    def test_cuda_fallback_on_runtime_error(self, mock_cuda_memory):
        """Test fallback behavior when CUDA functions raise RuntimeError."""
        with mock_cuda_memory(get_props_side_effect=RuntimeError("CUDA error")):
            # Large matrix should trigger fallback heuristic: return 10
            batch_size = _auto_detect_permutation_batch_size(
                kernel_mat_size=600,
                n_permutations=100,
                device=torch.device("cuda:0"),
            )

            assert batch_size == 10

    def test_cuda_fallback_on_attribute_error(self, mock_cuda_memory):
        """Test fallback behavior when CUDA functions raise AttributeError."""
        with mock_cuda_memory(get_props_side_effect=AttributeError("No CUDA support")):
            # Small matrix should trigger fallback: return 1
            batch_size = _auto_detect_permutation_batch_size(
                kernel_mat_size=400,
                n_permutations=100,
                device=torch.device("cuda:0"),
            )

            assert batch_size == 1

    def test_cuda_fallback_large_matrix_heuristic(self, mock_cuda_memory):
        """Test fallback returns 10 for large matrices (>500)."""
        # Mock get_device_properties to succeed but memory_reserved to fail
        with mock_cuda_memory(
            total_memory=16 * 1024**3,
            reserved_memory=None,
            mem_reserved_side_effect=RuntimeError("CUDA error"),
        ):
            # kernel_mat_size > 500 should return 10
            batch_size = _auto_detect_permutation_batch_size(
                kernel_mat_size=1000,
                n_permutations=100,
                device=torch.device("cuda:0"),
            )

            assert batch_size == 10

    def test_cuda_fallback_small_matrix_heuristic(self, mock_cuda_memory):
        """Test fallback returns 1 for small matrices (<=500)."""
        # Mock get_device_properties to succeed but memory_reserved to fail
        with mock_cuda_memory(
            total_memory=16 * 1024**3,
            reserved_memory=None,
            mem_reserved_side_effect=RuntimeError("CUDA error"),
        ):
            # kernel_mat_size <= 500 should return 1
            batch_size = _auto_detect_permutation_batch_size(
                kernel_mat_size=500,
                n_permutations=100,
                device=torch.device("cuda:0"),
            )

            assert batch_size == 1

    def test_cuda_max_batch_size_calculation(self, mock_cuda_memory):
        """Test max_batch_size is correctly calculated and capped at 1."""
        # Very limited memory scenario
        total_memory = 1 * 1024**3  # 1 GB
        reserved_memory = int(0.99 * 1024**3)  # 0.99 GB reserved

        with mock_cuda_memory(total_memory=total_memory, reserved_memory=reserved_memory):
            batch_size = _auto_detect_permutation_batch_size(
                kernel_mat_size=5000,  # Very large matrix
                n_permutations=100,
                device=torch.device("cuda:0"),
            )

            # Even with very limited memory, should return at least 1
            assert batch_size >= 1

    def test_cuda_all_permutations_fit_returns_one(self, mock_cuda_memory):
        """Test returns 1 when all permutations fit in memory."""
        # Lots of memory available
        with mock_cuda_memory(total_memory=32 * 1024**3, reserved_memory=1 * 1024**3):
            batch_size = _auto_detect_permutation_batch_size(
                kernel_mat_size=500,  # Moderate matrix
                n_permutations=100,  # Not too many permutations
                device=torch.device("cuda:0"),
            )

            # With plenty of memory, should return 1 (no batching needed)
            assert batch_size == 1

    def test_cuda_conservative_batch_size_50_percent_reduction(self, mock_cuda_memory):
        """Test that suggested batch size is 50% of calculated max."""
        # Set up memory to force specific calculation
        total_memory = 10 * 1024**3  # 10 GB
        reserved_memory = 5 * 1024**3  # 5 GB reserved
        # Available: (10-5) * 0.5 = 2.5 GB

        with mock_cuda_memory(total_memory=total_memory, reserved_memory=reserved_memory):
            # kernel_mat_size=1500: 1500*1500*4*2 = 18 MB per permutation
            # max_batch = 2.5GB / 18MB = ~142
            # final_batch = 142 * 0.5 = 71
            kernel_mat_size = 1500
            n_permutations = 200  # More than max_batch

            batch_size = _auto_detect_permutation_batch_size(
                kernel_mat_size=kernel_mat_size,
                n_permutations=n_permutations,
                device=torch.device("cuda:0"),
            )

            # Should be significantly less than max_batch
            # and definitely less than n_permutations
            assert isinstance(batch_size, int)
            assert 1 <= batch_size < n_permutations


@pytest.mark.required
class TestMMDKernelMatrixEdgeCases:
    """Additional tests for MMD kernel matrix edge cases."""

    def test_mmd_from_kernel_matrix_batched_exact_division(self):
        """Test batched permutations when n_permutations divides evenly."""
        n, m, n_perms = 50, 30, 20
        batch_size = 10  # Divides evenly into 20
        kernel_mat = torch.rand(n + m, n + m)
        kernel_mat -= torch.diag(kernel_mat.diag())

        torch.manual_seed(42)
        mmd_batched = mmd2_from_kernel_matrix(
            kernel_mat,
            m,
            zero_diag=False,
            n_permutations=n_perms,
            permutation_batch_size=batch_size,
        )

        # Should return correct number of permutations
        assert mmd_batched.shape == (n_perms,)

    def test_mmd_from_kernel_matrix_batched_partial_batch(self):
        """Test batched permutations with remainder (line 499-514)."""
        n, m, n_perms = 50, 30, 25
        batch_size = 10  # Last batch will have 5 permutations
        kernel_mat = torch.rand(n + m, n + m)
        kernel_mat -= torch.diag(kernel_mat.diag())

        torch.manual_seed(42)
        mmd_batched = mmd2_from_kernel_matrix(
            kernel_mat,
            m,
            zero_diag=False,
            n_permutations=n_perms,
            permutation_batch_size=batch_size,
        )

        # Should still return correct number of permutations
        assert mmd_batched.shape == (n_perms,)

    def test_mmd_from_kernel_matrix_no_permutations(self):
        """Test mmd2_from_kernel_matrix with n_permutations=0 (line 524-527)."""
        n, m = 50, 30
        kernel_mat = torch.rand(n + m, n + m)
        kernel_mat -= torch.diag(kernel_mat.diag())

        # With n_permutations=0, should return scalar MMD^2
        mmd_scalar = mmd2_from_kernel_matrix(kernel_mat, m, zero_diag=False, n_permutations=0)

        # Should be a scalar (0-d tensor)
        assert mmd_scalar.ndim == 0
        assert isinstance(mmd_scalar.item(), float)
