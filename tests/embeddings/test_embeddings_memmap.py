"""Tests for Embeddings class memory management, caching, and persistence."""

from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from dataeval import Embeddings
from dataeval.extractors import TorchExtractor
from tests.conftest import SimpleDataset

NP_MAJOR_VERSION = tuple(int(x) for x in np.__version__.split("."))[0]


class FlattenModel(torch.nn.Module):
    """Simple model that flattens input for testing."""

    def forward(self, x):
        return x.flatten(1)


@pytest.fixture
def encoder():
    """Simple extractor for testing."""
    return TorchExtractor(FlattenModel(), device="cpu")


@pytest.fixture
def memmap_embeddings(tmp_path) -> Generator[tuple[Embeddings, NDArray[Any]], None, None]:
    """Create temporary memmap-backed embeddings for testing."""
    cache_path = tmp_path / "test_embeddings.npy"
    shape = (1000, 128)
    dtype = np.float32

    data = np.random.randn(*shape).astype(dtype)
    np.save(cache_path, data)

    # Create embeddings-only instance from memmap
    emb = Embeddings()
    emb._embeddings = np.load(cache_path, mmap_mode="r")
    emb._cached_idx = set(range(shape[0]))
    emb._dataset = data

    yield emb, data


@pytest.fixture
def lazy_embeddings(simple_dataset, tmp_path, encoder) -> Generator[tuple[Embeddings, SimpleDataset, Path], None, None]:
    """Create embeddings with lazy evaluation for testing compute()."""
    cache_path = tmp_path / "lazy_embeddings.npy"
    emb = Embeddings(simple_dataset, extractor=encoder, path=cache_path)
    yield emb, simple_dataset, cache_path


@pytest.fixture
def in_memory_embeddings(simple_dataset, encoder) -> Generator[tuple[Embeddings, SimpleDataset], None, None]:
    """Create in-memory embeddings (no path) for testing."""
    emb = Embeddings(simple_dataset, extractor=encoder, path=None)
    yield emb, simple_dataset


class TestMemmapPreservation:
    """Test suite for memmap preservation in Embeddings.__array__()."""

    def test_array_protocol_on_memmap(self, memmap_embeddings: tuple[Embeddings, NDArray[Any]]):
        """__array__() should return memmap directly for embeddings-only instances."""
        emb, _ = memmap_embeddings
        arr = emb.__array__()
        assert isinstance(arr, np.memmap), "Expected __array__() to return memmap"

    def test_asarray_converts_memmap_to_ndarray(self, memmap_embeddings: tuple[Embeddings, NDArray[Any]]):
        """np.asarray() converts memmap to ndarray (expected numpy behavior)."""
        emb, _ = memmap_embeddings
        arr = np.asarray(emb)
        assert isinstance(arr, np.ndarray), "Expected ndarray"
        assert not isinstance(arr, np.memmap), "np.asarray() should not preserve memmap"

    def test_direct_access_preserves_memmap(self, memmap_embeddings: tuple[Embeddings, NDArray[Any]]):
        """Direct access to _embeddings preserves memmap."""
        emb, _ = memmap_embeddings
        arr = emb._embeddings
        assert isinstance(arr, np.memmap), "Direct access should preserve memmap"

    def test_explicit_copy_loads_to_memory(self, memmap_embeddings: tuple[Embeddings, NDArray[Any]]):
        """Explicit copy=True should load memmap into memory."""
        emb, _ = memmap_embeddings
        arr = np.array(emb, copy=True)
        assert not isinstance(arr, np.memmap), "Expected in-memory array"

    def test_dtype_conversion_loads_to_memory(self, memmap_embeddings: tuple[Embeddings, NDArray[Any]]):
        """Dtype conversion should load memmap into memory."""
        emb, _ = memmap_embeddings
        arr = np.asarray(emb, dtype=np.float16)
        assert not isinstance(arr, np.memmap), "Expected in-memory array after dtype conversion"
        assert arr.dtype == np.float16

    @pytest.mark.xfail(
        NP_MAJOR_VERSION < 2,
        reason="numpy < 2 changes memmap/dtype/copy behavior; test expected to fail",
        strict=False,
    )
    def test_dtype_conversion_with_copy_false_raises(self, memmap_embeddings: tuple[Embeddings, NDArray[Any]]):
        """Dtype conversion with copy=False should raise ValueError."""
        emb, _ = memmap_embeddings
        with pytest.raises(ValueError, match="Cannot avoid copy when converting dtype"):
            np.array(emb, dtype=np.float64, copy=False)

    def test_data_integrity(self, memmap_embeddings: tuple[Embeddings, NDArray[Any]]):
        """Data should be identical whether memmap or in-memory."""
        emb, original_data = memmap_embeddings
        arr_memmap = emb._embeddings
        arr_memory = np.asarray(emb)
        np.testing.assert_array_equal(arr_memmap, original_data)
        np.testing.assert_array_equal(arr_memory, original_data)


class TestComputeMethod:
    """Test suite for the compute() method."""

    def test_compute_lazy_embeddings(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """compute() should compute all embeddings and cache them."""
        emb, dataset, _ = lazy_embeddings
        assert len(emb._cached_idx) == 0
        result = emb.compute()
        assert len(emb._cached_idx) == len(dataset)
        assert result is emb

    def test_compute_force_recomputes(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """compute(force=True) should recompute all embeddings."""
        emb, dataset, _ = lazy_embeddings
        emb.compute()
        first_shape = emb._embeddings.shape
        emb.compute(force=True)
        assert len(emb._cached_idx) == len(dataset)
        assert emb._embeddings.shape == first_shape

    def test_compute_partial_then_complete(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """Accessing partial data then compute() should complete caching."""
        emb, dataset, _ = lazy_embeddings
        _ = emb[0:10]
        assert len(emb._cached_idx) == 10
        emb.compute()
        assert len(emb._cached_idx) == len(dataset)

    def test_compute_idempotent(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """compute() called multiple times should be idempotent."""
        emb, _, _ = lazy_embeddings
        emb.compute()
        first_embeddings = emb._embeddings.copy()
        first_cached = emb._cached_idx.copy()
        emb.compute()
        assert emb._cached_idx == first_cached
        np.testing.assert_array_equal(emb._embeddings, first_embeddings)


class TestArrayProtocol:
    """Test suite for __array__ protocol with lazy embeddings."""

    def test_asarray_triggers_computation(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """np.asarray() on lazy embeddings should trigger computation."""
        emb, dataset, _ = lazy_embeddings
        assert len(emb._cached_idx) == 0
        arr = np.asarray(emb)
        assert len(emb._cached_idx) == len(dataset)
        assert arr.shape[0] == len(dataset)

    def test_asarray_equals_getitem_slice(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """np.asarray(emb) should equal emb[:]."""
        emb, _, _ = lazy_embeddings
        arr1 = np.asarray(emb)
        arr2 = emb[:]
        np.testing.assert_array_equal(arr1, arr2)


class TestSaveMethod:
    """Test suite for the save() method."""

    def test_save_computes_and_saves(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """save() should compute embeddings before saving."""
        emb, dataset, save_path = lazy_embeddings
        assert len(emb._cached_idx) == 0
        emb.save(save_path)
        assert len(emb._cached_idx) == len(dataset)
        assert save_path.exists()

    def test_save_without_path_raises(self, in_memory_embeddings: tuple[Embeddings, SimpleDataset]):
        """save() without path should raise ValueError."""
        emb, _ = in_memory_embeddings
        with pytest.raises(ValueError, match="No path specified"):
            emb.save()

    def test_save_uses_configured_path(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """save() without argument should use configured path."""
        emb, _, configured_path = lazy_embeddings
        emb.save()
        assert configured_path.exists()

    def test_save_chain_with_compute(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """save() should work in chain with compute()."""
        emb, _, save_path = lazy_embeddings
        emb.compute().save(save_path)
        assert save_path.exists()


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow_in_memory(self, simple_dataset: SimpleDataset, encoder):
        """Test complete workflow with in-memory caching."""
        # Create embeddings
        emb = Embeddings(simple_dataset, extractor=encoder, path=None)

        # Access some embeddings
        sample = emb[0:10]
        assert sample.shape[0] == 10

        # Compute all
        emb.compute()
        assert len(emb._cached_idx) == len(simple_dataset)

        # Access via numpy
        arr = np.asarray(emb)
        assert arr.shape[0] == len(simple_dataset)

    def test_full_workflow_with_disk_save(self, simple_dataset: SimpleDataset, tmp_path: Path, encoder):
        """Test complete workflow with disk persistence."""
        save_path = tmp_path / "embeddings.npy"

        # Create and compute embeddings
        emb = Embeddings(simple_dataset, extractor=encoder, path=save_path)

        # Compute and save
        emb.compute().save()

        assert save_path.exists()


class TestShouldUseMemmap:
    """Test suite for _should_use_memmap() method."""

    def test_should_use_memmap_no_path_returns_false(self, simple_dataset: SimpleDataset, encoder):
        """_should_use_memmap should return False when path is None."""
        emb = Embeddings(simple_dataset, extractor=encoder, path=None)
        result = emb._should_use_memmap((128,))
        assert result is False

    def test_should_use_memmap_small_data_with_path(
        self, simple_dataset: SimpleDataset, tmp_path: Path, encoder, monkeypatch
    ):
        """_should_use_memmap should return False when estimated size is below threshold."""

        class MockVirtualMemory:
            available = 10 * 1024**3  # 10 GB

        monkeypatch.setattr("psutil.virtual_memory", lambda: MockVirtualMemory())

        cache_path = tmp_path / "test.npy"
        emb = Embeddings(simple_dataset, extractor=encoder, path=cache_path, memory_threshold=0.8)
        result = emb._should_use_memmap((128,))
        assert result is False

    def test_should_use_memmap_large_data_with_path(
        self, simple_dataset: SimpleDataset, tmp_path: Path, encoder, monkeypatch
    ):
        """_should_use_memmap should return True when estimated size exceeds threshold."""

        class MockVirtualMemory:
            available = 100 * 1024**2  # 100 MB

        monkeypatch.setattr("psutil.virtual_memory", lambda: MockVirtualMemory())

        cache_path = tmp_path / "test.npy"
        emb = Embeddings(simple_dataset, extractor=encoder, path=cache_path, memory_threshold=0.8)
        result = emb._should_use_memmap((2000000,))
        assert result is True


class TestInitializeStorage:
    """Test suite for _initialize_storage method."""

    def test_initialize_storage_creates_memmap(self, simple_dataset: SimpleDataset, tmp_path, encoder, monkeypatch):
        """Test _initialize_storage creates memmap when path is set and threshold exceeded."""

        class MockVirtualMemory:
            available = 10 * 1024  # 10 KB - very small

        monkeypatch.setattr("psutil.virtual_memory", lambda: MockVirtualMemory())

        cache_path = tmp_path / "test.npy"
        embs = Embeddings(simple_dataset, extractor=encoder, path=cache_path, memory_threshold=0.01)

        sample_embedding = np.random.randn(128).astype(np.float32)
        embs._initialize_storage(sample_embedding)

        assert isinstance(embs._embeddings, np.memmap)
        assert embs._use_memmap is True
        assert cache_path.exists()

    def test_initialize_storage_creates_in_memory_array(
        self, simple_dataset: SimpleDataset, tmp_path, encoder, monkeypatch
    ):
        """Test _initialize_storage creates in-memory array when below threshold."""

        class MockVirtualMemory:
            available = 10 * 1024**3  # 10 GB

        monkeypatch.setattr("psutil.virtual_memory", lambda: MockVirtualMemory())

        cache_path = tmp_path / "test.npy"
        embs = Embeddings(simple_dataset, extractor=encoder, path=cache_path, memory_threshold=0.8)

        sample_embedding = np.random.randn(128).astype(np.float32)
        embs._initialize_storage(sample_embedding)

        assert isinstance(embs._embeddings, np.ndarray)
        assert not isinstance(embs._embeddings, np.memmap)
        assert embs._use_memmap is False


class TestBatchErrors:
    """Test suite for error cases in _batch method."""

    def test_batch_out_of_range_error(self, simple_dataset: SimpleDataset, encoder):
        """Test _batch raises IndexError for out of range indices."""
        embs = Embeddings(simple_dataset, extractor=encoder)
        out_of_range_indices = [0, 1, 100, 200]
        with pytest.raises(IndexError, match="Indices.*are out of range for dataset of size"):
            list(embs._batch(out_of_range_indices))


class TestGetitemErrors:
    """Test suite for error cases in __getitem__ method."""

    def test_getitem_sequence_invalid_element_raises(self):
        """Test __getitem__ raises TypeError for sequence with non-int elements."""
        arr = np.random.randn(10, 128)
        embs = Embeddings()
        embs._embeddings = arr
        embs._cached_idx = set(range(len(arr)))

        # Sequence with invalid element types
        with pytest.raises(TypeError, match="All indices in the sequence must be integers"):
            embs[[0, 1, "invalid", 3]]  # type: ignore


class TestProgressCallback:
    """Test suite for progress_callback functionality with memmap storage."""

    def test_progress_callback_with_memmap_storage(self, simple_dataset: SimpleDataset, tmp_path, encoder, monkeypatch):
        """Test that progress_callback works with memmap-backed embeddings."""
        callback_calls = []

        def callback(step: int, *, total: int | None = None, desc: str | None = None, extra_info: dict | None = None):
            callback_calls.append({"step": step, "total": total})

        class MockVirtualMemory:
            available = 10 * 1024  # 10 KB

        monkeypatch.setattr("psutil.virtual_memory", lambda: MockVirtualMemory())

        cache_path = tmp_path / "test.npy"
        embs = Embeddings(
            simple_dataset,
            extractor=encoder,
            path=cache_path,
            memory_threshold=0.01,
            progress_callback=callback,
        )
        embs.compute()

        assert len(callback_calls) > 0
        assert isinstance(embs._embeddings, np.memmap)
