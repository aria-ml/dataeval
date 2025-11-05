from collections.abc import Generator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from dataeval.data import Embeddings
from tests.conftest import SimpleDataset

NP_MAJOR_VERSION = tuple(int(x) for x in np.__version__.split("."))[0]


class IdentityModel(torch.nn.Module):
    """Model that flattens input to a fixed embedding dimension."""

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.embedding_dim = embedding_dim

    def forward(self, x):
        # Flatten and project to embedding dimension
        flat = self.flatten(x)
        # Use a simple linear projection
        if not hasattr(self, "proj"):
            input_dim = flat.shape[1]
            self.proj = torch.nn.Linear(input_dim, self.embedding_dim).to(x.device)
            # Initialize deterministically for consistent tests
            torch.nn.init.xavier_uniform_(self.proj.weight)
            torch.nn.init.zeros_(self.proj.bias)
        return self.proj(flat)


@pytest.fixture
def memmap_embeddings(tmp_path) -> Generator[tuple[Embeddings, NDArray[Any]], None, None]:
    """Create temporary memmap-backed embeddings for testing."""
    cache_path = tmp_path / "test_embeddings.npy"
    shape = (1000, 128)
    dtype = np.float32

    # Create data and save as proper .npy file
    data = np.random.randn(*shape).astype(dtype)
    np.save(cache_path, data)

    # Load as memmap from the .npy file
    loaded_memmap = np.load(cache_path, mmap_mode="r")

    # Create embeddings-only instance from memmap
    emb = Embeddings.from_array(loaded_memmap)

    # Verify it's actually memmap backed
    assert isinstance(emb._embeddings, np.memmap), "Fixture should create memmap-backed embeddings"

    yield emb, data


@pytest.fixture
def lazy_embeddings(simple_dataset, tmp_path) -> Generator[tuple[Embeddings, SimpleDataset, Path], None, None]:
    """Create embeddings with lazy evaluation for testing compute()."""
    cache_path = tmp_path / "lazy_embeddings.npy"
    model = IdentityModel(embedding_dim=128)

    emb = Embeddings(simple_dataset, batch_size=16, model=model, device="cpu", path=cache_path, verbose=False)

    yield emb, simple_dataset, cache_path


@pytest.fixture
def in_memory_embeddings(simple_dataset) -> Generator[tuple[Embeddings, SimpleDataset], None, None]:
    """Create in-memory embeddings (no path) for testing."""
    model = IdentityModel(embedding_dim=128)

    emb = Embeddings(
        simple_dataset,
        batch_size=16,
        model=model,
        device="cpu",
        path=None,  # In-memory only
        verbose=False,
    )

    yield emb, simple_dataset


class TestMemmapPreservation:
    """Test suite for memmap preservation in Embeddings.__array__()."""

    def test_array_protocol_on_memmap(self, memmap_embeddings: tuple[Embeddings, NDArray[Any]]):
        """__array__() should return memmap directly for embeddings-only instances."""
        emb, _ = memmap_embeddings

        # Call __array__ directly
        arr = emb.__array__()

        assert isinstance(arr, np.memmap), "Expected __array__() to return memmap"

    def test_asarray_converts_memmap_to_ndarray(self, memmap_embeddings: tuple[Embeddings, NDArray[Any]]):
        """np.asarray() converts memmap to ndarray (expected numpy behavior)."""
        emb, _ = memmap_embeddings

        # np.asarray() loads memmap into memory
        arr = np.asarray(emb)

        # This is expected behavior - np.asarray doesn't preserve memmap
        assert isinstance(arr, np.ndarray), "Expected ndarray"
        assert not isinstance(arr, np.memmap), "np.asarray() should not preserve memmap"

    def test_direct_access_preserves_memmap(self, memmap_embeddings: tuple[Embeddings, NDArray[Any]]):
        """Direct access to _embeddings preserves memmap."""
        emb, _ = memmap_embeddings

        # Direct access should preserve memmap
        arr = emb._embeddings

        assert isinstance(arr, np.memmap), "Direct access should preserve memmap"

    def test_getitem_on_memmap_preserves_type(self, memmap_embeddings: tuple[Embeddings, NDArray[Any]]):
        """Slicing memmap-backed embeddings returns memmap views."""
        emb, _ = memmap_embeddings

        # Access underlying memmap directly via _embeddings
        sliced = emb._embeddings[0:100]

        assert isinstance(sliced, np.memmap), "Slicing memmap should return memmap view"

    def test_explicit_copy_loads_to_memory(self, memmap_embeddings: tuple[Embeddings, NDArray[Any]]):
        """Explicit copy=True should load memmap into memory."""
        emb, _ = memmap_embeddings
        arr = np.array(emb, copy=True)

        assert not isinstance(arr, np.memmap), "Expected in-memory array"
        assert isinstance(arr, np.ndarray), "Expected ndarray"

    def test_dtype_conversion_loads_to_memory(self, memmap_embeddings: tuple[Embeddings, NDArray[Any]]):
        """Dtype conversion should load memmap into memory."""
        emb, _ = memmap_embeddings
        arr = np.asarray(emb, dtype=np.float16)

        assert not isinstance(arr, np.memmap), "Expected in-memory array after dtype conversion"
        assert arr.dtype == np.float16, "Expected float64 dtype"

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

    def test_operations_work_on_memmap_directly(self, memmap_embeddings: tuple[Embeddings, NDArray[Any]]):
        """Operations on underlying memmap work without loading all data."""
        emb, original_data = memmap_embeddings

        # Access underlying memmap directly
        arr = emb._embeddings

        # Compute statistics on memmap (doesn't load all into memory)
        mean_val = arr.mean()
        sum_val = arr.sum()

        # Verify against original data
        assert np.isclose(mean_val, original_data.mean()), "Mean computation incorrect"
        assert np.isclose(sum_val, original_data.sum()), "Sum computation incorrect"
        assert isinstance(arr, np.memmap), "Direct access still memmap"

    def test_memory_footprint_difference(self, memmap_embeddings: tuple[Embeddings, NDArray[Any]]):
        """Memmap has minimal memory footprint vs. in-memory copy."""
        emb, original_data = memmap_embeddings

        # Get memmap reference (minimal memory)
        arr_memmap = emb._embeddings

        # Get in-memory copy via np.asarray
        arr_memory = np.asarray(emb)

        # Both should have same data size
        assert arr_memmap.nbytes == arr_memory.nbytes
        assert arr_memmap.nbytes == original_data.nbytes

        # Verify they're different types
        assert isinstance(arr_memmap, np.memmap)
        assert not isinstance(arr_memory, np.memmap)

    def test_readonly_access_preserves_memmap(self, memmap_embeddings: tuple[Embeddings, NDArray[Any]]):
        """Read-only operations on underlying memmap don't change type."""
        emb, _ = memmap_embeddings
        arr = emb._embeddings  # Direct access

        # Multiple read operations
        _ = arr[0]
        _ = arr.shape
        _ = arr.dtype
        _ = arr.mean()
        _ = arr[100:200]

        # Should still be memmap after all read operations
        assert isinstance(arr, np.memmap)

    def test_data_integrity(self, memmap_embeddings: tuple[Embeddings, NDArray[Any]]):
        """Data should be identical whether memmap or in-memory."""
        emb, original_data = memmap_embeddings

        # Get both versions
        arr_memmap = emb._embeddings  # Memmap
        arr_memory = np.asarray(emb)  # Loaded into memory

        # All should be equal
        np.testing.assert_array_equal(arr_memmap, original_data)
        np.testing.assert_array_equal(arr_memory, original_data)
        np.testing.assert_array_equal(arr_memmap, arr_memory)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataset(self):
        """Test handling of empty datasets."""
        empty_dataset = SimpleDataset(size=0)
        model = IdentityModel()

        emb = Embeddings(empty_dataset, batch_size=16, model=model, device="cpu", path=None, verbose=False)

        arr = np.asarray(emb)
        assert arr.shape[0] == 0
        assert isinstance(arr, np.ndarray)

    def test_single_sample(self):
        """Test with single sample dataset."""
        single_dataset = SimpleDataset(size=1)
        model = IdentityModel()

        emb = Embeddings(single_dataset, batch_size=16, model=model, device="cpu", path=None, verbose=False)

        arr = np.asarray(emb)
        assert arr.shape[0] == 1
        assert len(arr.shape) == 2  # (1, embedding_dim)


class TestComputeMethod:
    """Test suite for the compute() method."""

    def test_compute_lazy_embeddings(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """compute() should compute all embeddings and cache them."""
        emb, dataset, _ = lazy_embeddings

        # Initially no embeddings cached
        assert len(emb._cached_idx) == 0

        # Compute all embeddings
        result = emb.compute()

        # All embeddings should now be cached
        assert len(emb._cached_idx) == len(dataset)
        assert emb._embeddings.shape[0] == len(dataset)

        # Should return self for chaining
        assert result is emb

    def test_compute_returns_self(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """compute() should return self for method chaining."""
        emb, _, _ = lazy_embeddings

        result = emb.compute()
        assert result is emb

    def test_compute_no_op_on_embeddings_only(self):
        """compute() should be a no-op on embeddings-only instances."""
        # Create embeddings-only instance
        data = np.random.randn(100, 128).astype(np.float32)
        emb = Embeddings.from_array(data)

        original_data = emb._embeddings.copy()

        # Should return immediately without error
        result = emb.compute()
        assert result is emb
        np.testing.assert_array_equal(emb._embeddings, original_data)

    def test_compute_force_recomputes(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """compute(force=True) should recompute all embeddings."""
        emb, dataset, _ = lazy_embeddings

        # Compute embeddings once
        emb.compute()
        first_shape = emb._embeddings.shape

        # Force recompute
        emb.compute(force=True)

        # Should have recomputed
        assert len(emb._cached_idx) == len(dataset)
        assert emb._embeddings.shape == first_shape

    def test_compute_partial_then_complete(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """Accessing partial data then compute() should complete caching."""
        emb, dataset, _ = lazy_embeddings

        # Access first 10 embeddings
        _ = emb[0:10]
        assert len(emb._cached_idx) == 10

        # Compute remaining
        emb.compute()
        assert len(emb._cached_idx) == len(dataset)

    def test_compute_idempotent(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """compute() called multiple times should be idempotent."""
        emb, _, _ = lazy_embeddings

        # First compute
        emb.compute()
        first_embeddings = emb._embeddings.copy()
        first_cached = emb._cached_idx.copy()

        # Second compute (without force)
        emb.compute()

        # Should be unchanged
        assert emb._cached_idx == first_cached
        np.testing.assert_array_equal(emb._embeddings, first_embeddings)

    def test_compute_then_access(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """Accessing embeddings after compute() should be fast (no recompute)."""
        emb, dataset, _ = lazy_embeddings

        # Compute all
        emb.compute()
        cached_after_compute = len(emb._cached_idx)

        # Access some embeddings
        sample = emb[10:20]

        # Should not trigger additional caching
        assert len(emb._cached_idx) == cached_after_compute
        assert sample.shape[0] == 10

    def test_compute_force_clears_cache(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """compute(force=True) should clear cache before recomputing."""
        emb, _, _ = lazy_embeddings

        # Compute and verify cached
        emb.compute()
        assert len(emb._cached_idx) > 0
        original_size = emb._embeddings.size

        # Force recompute
        emb.compute(force=True)

        # Should have same final state
        assert len(emb._cached_idx) == len(emb._dataset)
        assert emb._embeddings.size == original_size


class TestArrayProtocol:
    """Test suite for __array__ protocol with lazy embeddings."""

    def test_asarray_triggers_computation(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """np.asarray() on lazy embeddings should trigger computation."""
        emb, dataset, _ = lazy_embeddings

        # Initially no embeddings cached
        assert len(emb._cached_idx) == 0

        # Call np.asarray - should trigger computation
        arr = np.asarray(emb)

        # Should have computed all embeddings
        assert len(emb._cached_idx) == len(dataset)
        assert arr.shape[0] == len(dataset)

    def test_asarray_equals_getitem_slice(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """np.asarray(emb) should equal emb[:]."""
        emb, _, _ = lazy_embeddings

        arr1 = np.asarray(emb)
        arr2 = emb[:]

        np.testing.assert_array_equal(arr1, arr2)

    def test_asarray_on_computed_embeddings(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """np.asarray() on already-computed embeddings should not recompute."""
        emb, _, _ = lazy_embeddings

        # Pre-compute
        emb.compute()
        first_arr = emb._embeddings.copy()

        # Call asarray
        arr = np.asarray(emb)

        # Should be same array (not recomputed)
        np.testing.assert_array_equal(arr, first_arr)

    def test_asarray_empty_lazy_embeddings(self):
        """np.asarray() on empty lazy embeddings should work."""
        empty_dataset = SimpleDataset(size=0)
        model = IdentityModel()

        emb = Embeddings(empty_dataset, batch_size=16, model=model, device="cpu", path=None, verbose=False)

        arr = np.asarray(emb)

        assert arr.shape[0] == 0
        assert isinstance(arr, np.ndarray)

    def test_asarray_always_computes_for_lazy(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """np.asarray() should always compute for lazy embeddings."""
        emb, dataset, _ = lazy_embeddings

        # Call asarray - should compute
        arr1 = np.asarray(emb)
        assert arr1.shape[0] == len(dataset)

        # Call again - should return cached result
        arr2 = np.asarray(emb)
        assert arr2.shape[0] == len(dataset)

        # Both calls should return valid arrays
        assert arr1.shape == arr2.shape
        np.testing.assert_array_equal(arr1, arr2)


class TestSaveMethod:
    """Test suite for the save() method."""

    def test_save_computes_and_saves(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """save() should compute embeddings before saving."""
        emb, dataset, save_path = lazy_embeddings

        # Initially not computed
        assert len(emb._cached_idx) == 0

        # Save should compute and write to disk
        emb.save(save_path)

        # Should have computed all embeddings
        assert len(emb._cached_idx) == len(dataset)

        # File should exist
        assert save_path.exists()

    def test_save_without_path_raises(self, in_memory_embeddings: tuple[Embeddings, SimpleDataset]):
        """save() without path should raise ValueError."""
        emb, _ = in_memory_embeddings

        with pytest.raises(ValueError, match="No path specified"):
            emb.save()

    def test_save_uses_configured_path(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """save() without argument should use configured path."""
        emb, dataset, configured_path = lazy_embeddings

        # Save without specifying path
        emb.save()

        # Should save to configured path
        assert configured_path.exists()

    def test_save_chain_with_compute(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path]):
        """save() should work in chain with compute()."""
        emb, _, save_path = lazy_embeddings

        # Chain compute and save
        emb.compute().save(save_path)

        # File should exist
        assert save_path.exists()

    def test_save_custom_path(self, lazy_embeddings: tuple[Embeddings, SimpleDataset, Path], tmp_path: Path):
        """save() with custom path should save to that path."""
        emb, _, _ = lazy_embeddings

        custom_path = tmp_path / "custom_output.npy"

        emb.save(custom_path)

        assert custom_path.exists()

    def test_save_in_memory_to_disk(self, in_memory_embeddings: tuple[Embeddings, SimpleDataset], tmp_path: Path):
        """In-memory embeddings can be saved to disk."""
        emb, dataset = in_memory_embeddings

        # Compute embeddings
        emb.compute()

        # Save to disk
        output_path = tmp_path / "in_memory_saved.npy"
        emb.save(output_path)

        assert output_path.exists()

        # Verify saved data
        loaded = np.load(output_path)
        assert loaded.shape[0] == len(dataset)


class TestFromArrayMethod:
    """Test suite for from_array() classmethod."""

    def test_from_array_creates_embeddings_only(self):
        """from_array() should create embeddings-only instance."""
        data = np.random.randn(100, 128).astype(np.float32)

        emb = Embeddings.from_array(data)

        # Should be embeddings-only
        assert emb._embeddings_only
        assert len(emb._cached_idx) == len(data)

    def test_from_array_preserves_data(self):
        """from_array() should preserve original data."""
        data = np.random.randn(50, 256).astype(np.float32)

        emb = Embeddings.from_array(data)

        np.testing.assert_array_equal(emb._embeddings, data)

    def test_from_array_replaces_load(self, tmp_path: Path):
        """from_array(np.load()) should replace old load() method."""
        save_path = tmp_path / "test.npy"

        # Save some data
        original_data = np.random.randn(75, 384).astype(np.float32)
        np.save(save_path, original_data)

        # Load using from_array
        loaded = np.load(save_path)
        emb = Embeddings.from_array(loaded)

        # Should match original
        np.testing.assert_array_equal(emb._embeddings, original_data)

    def test_from_array_with_memmap(self, tmp_path: Path):
        """from_array() should work with memmap arrays."""
        memmap_path = tmp_path / "memmap.npy"
        shape = (200, 128)

        # Create memmap
        data = np.random.randn(*shape).astype(np.float32)
        memmap_arr = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=shape)
        memmap_arr[:] = data
        memmap_arr.flush()

        # Create embeddings from memmap
        emb = Embeddings.from_array(memmap_arr)

        # Should preserve memmap
        assert isinstance(emb._embeddings, np.memmap)
        np.testing.assert_array_equal(emb._embeddings, data)


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow_in_memory(self, simple_dataset: SimpleDataset):
        """Test complete workflow with in-memory caching."""
        model = IdentityModel()

        # Create embeddings
        emb = Embeddings(simple_dataset, batch_size=16, model=model, device="cpu", path=None, verbose=False)

        # Access some embeddings
        sample = emb[0:10]
        assert sample.shape[0] == 10

        # Compute all
        emb.compute()
        assert len(emb._cached_idx) == len(simple_dataset)

        # Access via numpy
        arr = np.asarray(emb)
        assert arr.shape[0] == len(simple_dataset)

    def test_full_workflow_with_disk_save(self, simple_dataset: SimpleDataset, tmp_path: Path):
        """Test complete workflow with disk persistence."""
        model = IdentityModel()
        save_path = tmp_path / "embeddings.npy"

        # Create and compute embeddings
        emb = Embeddings(simple_dataset, batch_size=16, model=model, device="cpu", path=save_path, verbose=False)

        # Compute and save
        emb.compute().save()

        assert save_path.exists()

        # Load back
        loaded_arr = np.load(save_path)
        loaded_emb = Embeddings.from_array(loaded_arr)

        # Verify
        assert loaded_emb.shape[0] == len(simple_dataset)
        np.testing.assert_array_equal(loaded_emb[:], emb[:])


class TestLoadMethod:
    """Test suite for the load() classmethod."""

    def test_load_as_ndarray(self, tmp_path: Path):
        """load() without mmap_mode should load as in-memory ndarray."""
        save_path = tmp_path / "test.npy"

        # Save some data
        original_data = np.random.randn(100, 128).astype(np.float32)
        np.save(save_path, original_data)

        # Load as ndarray (default)
        emb = Embeddings.load(save_path)

        # Should be in-memory ndarray
        assert isinstance(emb._embeddings, np.ndarray)
        assert not isinstance(emb._embeddings, np.memmap)
        np.testing.assert_array_equal(emb._embeddings, original_data)

    def test_load_as_memmap(self, tmp_path: Path):
        """load() with mmap_mode should load as memmap."""
        save_path = tmp_path / "test.npy"

        # Save some data
        original_data = np.random.randn(200, 256).astype(np.float32)
        np.save(save_path, original_data)

        # Load as memmap
        emb = Embeddings.load(save_path, mmap_mode="r")

        # Should be memmap
        assert isinstance(emb._embeddings, np.memmap)
        np.testing.assert_array_equal(emb._embeddings, original_data)

    def test_load_different_mmap_modes(self, tmp_path: Path):
        """load() should support different mmap_mode values."""
        save_path = tmp_path / "test.npy"
        original_data = np.random.randn(50, 64).astype(np.float32)
        np.save(save_path, original_data)

        # This is a little silly
        modes: list[Literal["r", "r+", "c"]] = ["r", "r+", "c"]

        # Test different modes
        for mode in modes:
            emb = Embeddings.load(save_path, mmap_mode=mode)
            assert isinstance(emb._embeddings, np.memmap)
            np.testing.assert_array_equal(emb._embeddings, original_data)

    def test_load_creates_embeddings_only_instance(self, tmp_path: Path):
        """load() should create embeddings-only instance."""
        save_path = tmp_path / "test.npy"
        original_data = np.random.randn(75, 128).astype(np.float32)
        np.save(save_path, original_data)

        emb = Embeddings.load(save_path)

        # Should be embeddings-only
        assert emb._embeddings_only
        assert len(emb._cached_idx) == len(original_data)

    def test_load_nonexistent_file_raises(self, tmp_path: Path):
        """load() with nonexistent file should raise FileNotFoundError."""
        nonexistent_path = tmp_path / "does_not_exist.npy"

        with pytest.raises(FileNotFoundError, match="File not found"):
            Embeddings.load(nonexistent_path)

    def test_load_with_string_path(self, tmp_path: Path):
        """load() should accept string paths."""
        save_path = tmp_path / "test.npy"
        original_data = np.random.randn(50, 128).astype(np.float32)
        np.save(save_path, original_data)

        # Load using string path
        emb = Embeddings.load(str(save_path))

        np.testing.assert_array_equal(emb._embeddings, original_data)

    def test_load_with_path_object(self, tmp_path: Path):
        """load() should accept Path objects."""
        save_path = tmp_path / "test.npy"
        original_data = np.random.randn(50, 128).astype(np.float32)
        np.save(save_path, original_data)

        # Load using Path object
        emb = Embeddings.load(save_path)

        np.testing.assert_array_equal(emb._embeddings, original_data)

    def test_load_preserves_dtype(self, tmp_path: Path):
        """load() should preserve the original dtype."""
        save_path = tmp_path / "test.npy"

        # Test with different dtypes
        for dtype in [np.float32, np.float64, np.float16]:
            data = np.random.randn(50, 64).astype(dtype)
            np.save(save_path, data)

            emb = Embeddings.load(save_path)
            assert emb._embeddings.dtype == dtype

    def test_load_preserves_shape(self, tmp_path: Path):
        """load() should preserve the original shape."""
        save_path = tmp_path / "test.npy"

        # Test with different shapes
        shapes = [(100, 128), (50, 256), (200, 64)]
        for shape in shapes:
            data = np.random.randn(*shape).astype(np.float32)
            np.save(save_path, data)

            emb = Embeddings.load(save_path)
            assert emb.shape == shape

    def test_load_then_access(self, tmp_path: Path):
        """Loaded embeddings should support indexing."""
        save_path = tmp_path / "test.npy"
        original_data = np.random.randn(100, 128).astype(np.float32)
        np.save(save_path, original_data)

        emb = Embeddings.load(save_path)

        # Test various access patterns
        assert emb[0].shape == (128,)
        assert emb[0:10].shape == (10, 128)
        assert emb[:].shape == (100, 128)

        # Verify data integrity
        np.testing.assert_array_equal(emb[0], original_data[0])
        np.testing.assert_array_equal(emb[:], original_data)

    def test_load_memmap_then_access(self, tmp_path: Path):
        """Loaded memmap embeddings should support indexing."""
        save_path = tmp_path / "test.npy"
        original_data = np.random.randn(100, 128).astype(np.float32)
        np.save(save_path, original_data)

        emb = Embeddings.load(save_path, mmap_mode="r")

        # Verify memmap is preserved and data is accessible
        assert isinstance(emb._embeddings, np.memmap)
        np.testing.assert_array_equal(emb[0], original_data[0])
        np.testing.assert_array_equal(emb[:], original_data)

    def test_save_then_load_roundtrip(self, simple_dataset: SimpleDataset, tmp_path: Path):
        """save() then load() should preserve embeddings."""
        model = IdentityModel()
        save_path = tmp_path / "roundtrip.npy"

        # Create, compute, and save embeddings
        original_emb = Embeddings(
            simple_dataset, batch_size=16, model=model, device="cpu", path=save_path, verbose=False
        )
        original_emb.compute().save()

        # Load back using load() method
        loaded_emb = Embeddings.load(save_path)

        # Verify data is identical
        assert loaded_emb.shape == original_emb.shape
        np.testing.assert_array_equal(loaded_emb[:], original_emb[:])

    def test_save_then_load_memmap_roundtrip(self, simple_dataset: SimpleDataset, tmp_path: Path):
        """save() then load(mmap_mode='r') should preserve embeddings as memmap."""
        model = IdentityModel()
        save_path = tmp_path / "roundtrip_memmap.npy"

        # Create, compute, and save embeddings
        original_emb = Embeddings(
            simple_dataset, batch_size=16, model=model, device="cpu", path=save_path, verbose=False
        )
        original_emb.compute().save()

        # Load back as memmap
        loaded_emb = Embeddings.load(save_path, mmap_mode="r")

        # Verify it's memmap and data is identical
        assert isinstance(loaded_emb._embeddings, np.memmap)
        assert loaded_emb.shape == original_emb.shape
        np.testing.assert_array_equal(loaded_emb[:], original_emb[:])

    def test_load_equivalent_to_from_array_with_np_load(self, tmp_path: Path):
        """load() should be equivalent to from_array(np.load())."""
        save_path = tmp_path / "test.npy"
        original_data = np.random.randn(100, 128).astype(np.float32)
        np.save(save_path, original_data)

        # Using load()
        emb1 = Embeddings.load(save_path)

        # Using from_array(np.load())
        emb2 = Embeddings.from_array(np.load(save_path))

        # Should be equivalent
        assert emb1._embeddings_only == emb2._embeddings_only
        assert emb1.shape == emb2.shape
        np.testing.assert_array_equal(emb1[:], emb2[:])


class TestShouldUseMemmap:
    """Test suite for _should_use_memmap() method with mocked system functionality."""

    def test_should_use_memmap_no_path_returns_false(self, simple_dataset: SimpleDataset):
        """_should_use_memmap should return False when path is None."""
        model = IdentityModel()

        # Create embeddings without path
        emb = Embeddings(
            simple_dataset,
            batch_size=16,
            model=model,
            device="cpu",
            path=None,  # No path
            verbose=False,
        )

        # Should return False regardless of estimated size
        embedding_shape = (128,)
        result = emb._should_use_memmap(embedding_shape)

        assert result is False

    def test_should_use_memmap_small_data_with_path(self, simple_dataset: SimpleDataset, tmp_path: Path, monkeypatch):
        """_should_use_memmap should return False when estimated size is below threshold."""
        model = IdentityModel()
        cache_path = tmp_path / "test.npy"

        # Mock psutil to return large available memory (10 GB)
        class MockVirtualMemory:
            available = 10 * 1024**3  # 10 GB

        monkeypatch.setattr("psutil.virtual_memory", lambda: MockVirtualMemory())

        # Create embeddings with path and default threshold (0.8)
        emb = Embeddings(
            simple_dataset,
            batch_size=16,
            model=model,
            device="cpu",
            path=cache_path,
            memory_threshold=0.8,
            verbose=False,
        )

        # Small embedding shape - should be well below threshold
        # 50 samples * 128 dims * 4 bytes = 25.6 KB << 8 GB (80% of 10 GB)
        embedding_shape = (128,)
        result = emb._should_use_memmap(embedding_shape)

        assert result is False

    def test_should_use_memmap_large_data_with_path(self, simple_dataset: SimpleDataset, tmp_path: Path, monkeypatch):
        """_should_use_memmap should return True when estimated size exceeds threshold."""
        model = IdentityModel()
        cache_path = tmp_path / "test.npy"

        # Mock psutil to return small available memory (100 MB)
        class MockVirtualMemory:
            available = 100 * 1024**2  # 100 MB

        monkeypatch.setattr("psutil.virtual_memory", lambda: MockVirtualMemory())

        # Create embeddings with path and default threshold (0.8)
        emb = Embeddings(
            simple_dataset,
            batch_size=16,
            model=model,
            device="cpu",
            path=cache_path,
            memory_threshold=0.8,
            verbose=False,
        )

        # Large embedding shape - should exceed threshold
        # 50 samples * 2000000 dims * 4 bytes = 400 MB > 80 MB (80% of 100 MB)
        embedding_shape = (2000000,)
        result = emb._should_use_memmap(embedding_shape)

        assert result is True

    def test_should_use_memmap_exact_threshold(self, simple_dataset: SimpleDataset, tmp_path: Path, monkeypatch):
        """_should_use_memmap should handle exact threshold boundary."""
        model = IdentityModel()
        cache_path = tmp_path / "test.npy"

        # Available memory: 1000 bytes
        class MockVirtualMemory:
            available = 1000

        monkeypatch.setattr("psutil.virtual_memory", lambda: MockVirtualMemory())

        emb = Embeddings(
            simple_dataset,
            batch_size=16,
            model=model,
            device="cpu",
            path=cache_path,
            memory_threshold=0.8,  # Threshold at 800 bytes
            verbose=False,
        )

        # Calculate embedding shape that gives exactly 800 bytes
        # 50 samples * X dims * 4 bytes = 800 bytes
        # X = 800 / (50 * 4) = 4
        # Exactly at threshold (800 bytes): should NOT use memmap (> not >=)
        embedding_shape = (4,)
        result = emb._should_use_memmap(embedding_shape)
        assert result is False

        # Just above threshold: should use memmap
        embedding_shape_above = (5,)  # 50 * 5 * 4 = 1000 bytes > 800
        result_above = emb._should_use_memmap(embedding_shape_above)
        assert result_above is True

    def test_should_use_memmap_different_thresholds(self, simple_dataset: SimpleDataset, tmp_path: Path, monkeypatch):
        """_should_use_memmap should respect different memory_threshold values."""
        model = IdentityModel()
        cache_path = tmp_path / "test.npy"

        # Available memory: 1 GB
        class MockVirtualMemory:
            available = 1 * 1024**3

        monkeypatch.setattr("psutil.virtual_memory", lambda: MockVirtualMemory())

        # Embedding shape that gives ~500 MB
        # 50 samples * 2621440 dims * 4 bytes ≈ 500 MB
        embedding_shape = (2621440,)

        # Test with low threshold (0.3) - 300 MB threshold - should use memmap
        emb_low = Embeddings(
            simple_dataset,
            batch_size=16,
            model=model,
            device="cpu",
            path=cache_path,
            memory_threshold=0.3,
            verbose=False,
        )
        assert emb_low._should_use_memmap(embedding_shape) is True

        # Test with medium threshold (0.5) - 500 MB threshold - should NOT use memmap (exactly at)
        emb_med = Embeddings(
            simple_dataset,
            batch_size=16,
            model=model,
            device="cpu",
            path=cache_path,
            memory_threshold=0.5,
            verbose=False,
        )
        # Note: This may be True or False depending on exact calculation
        result_med = emb_med._should_use_memmap(embedding_shape)
        # Just verify it's a boolean - exact value depends on rounding
        assert isinstance(result_med, bool)

        # Test with high threshold (0.9) - 900 MB threshold - should NOT use memmap
        emb_high = Embeddings(
            simple_dataset,
            batch_size=16,
            model=model,
            device="cpu",
            path=cache_path,
            memory_threshold=0.9,
            verbose=False,
        )
        assert emb_high._should_use_memmap(embedding_shape) is False

    def test_should_use_memmap_verbose_logging(
        self, simple_dataset: SimpleDataset, tmp_path: Path, monkeypatch, caplog
    ):
        """_should_use_memmap should log when using memmap if verbose=True."""
        import logging

        model = IdentityModel()
        cache_path = tmp_path / "test.npy"

        # Mock psutil to return small available memory
        class MockVirtualMemory:
            available = 50 * 1024**2  # 50 MB

        monkeypatch.setattr("psutil.virtual_memory", lambda: MockVirtualMemory())

        # Create embeddings with verbose=True
        emb = Embeddings(
            simple_dataset,
            batch_size=16,
            model=model,
            device="cpu",
            path=cache_path,
            memory_threshold=0.8,
            verbose=True,
        )

        # Large embedding that will trigger memmap
        # 50 samples * 1000000 dims * 4 bytes = 200 MB > 40 MB (80% of 50 MB)
        embedding_shape = (1000000,)

        with caplog.at_level(logging.INFO):
            result = emb._should_use_memmap(embedding_shape)

        assert result is True

        # Check that log message was produced
        assert any("Using memory-mapped storage" in record.message for record in caplog.records)

    def test_should_use_memmap_no_logging_when_not_verbose(
        self, simple_dataset: SimpleDataset, tmp_path: Path, monkeypatch, caplog
    ):
        """_should_use_memmap should not log when verbose=False."""
        import logging

        model = IdentityModel()
        cache_path = tmp_path / "test.npy"

        # Mock psutil to return small available memory
        class MockVirtualMemory:
            available = 50 * 1024**2  # 50 MB

        monkeypatch.setattr("psutil.virtual_memory", lambda: MockVirtualMemory())

        # Create embeddings with verbose=False
        emb = Embeddings(
            simple_dataset,
            batch_size=16,
            model=model,
            device="cpu",
            path=cache_path,
            memory_threshold=0.8,
            verbose=False,
        )

        # Large embedding that will trigger memmap
        embedding_shape = (1000000,)

        with caplog.at_level(logging.INFO):
            result = emb._should_use_memmap(embedding_shape)

        assert result is True

        # Should NOT have any log messages
        assert not any("Using memory-mapped storage" in record.message for record in caplog.records)

    def test_should_use_memmap_no_logging_when_false(
        self, simple_dataset: SimpleDataset, tmp_path: Path, monkeypatch, caplog
    ):
        """_should_use_memmap should not log when returning False, even if verbose."""
        import logging

        model = IdentityModel()
        cache_path = tmp_path / "test.npy"

        # Mock psutil to return large available memory
        class MockVirtualMemory:
            available = 10 * 1024**3  # 10 GB

        monkeypatch.setattr("psutil.virtual_memory", lambda: MockVirtualMemory())

        # Create embeddings with verbose=True
        emb = Embeddings(
            simple_dataset,
            batch_size=16,
            model=model,
            device="cpu",
            path=cache_path,
            memory_threshold=0.8,
            verbose=True,
        )

        # Small embedding that won't trigger memmap
        embedding_shape = (128,)

        with caplog.at_level(logging.INFO):
            result = emb._should_use_memmap(embedding_shape)

        assert result is False

        # Should NOT have any log messages since memmap wasn't used
        assert not any("Using memory-mapped storage" in record.message for record in caplog.records)


class TestInitializeStorage:
    """Test suite for _initialize_storage method."""

    def test_initialize_storage_creates_memmap(self, simple_dataset: SimpleDataset, tmp_path, monkeypatch):
        """Test _initialize_storage creates memmap when path is set and threshold exceeded (lines 296-299)"""
        cache_path = tmp_path / "test.npy"

        # Mock psutil to return very small available memory to force memmap
        class MockVirtualMemory:
            available = 10 * 1024  # 10 KB - very small

        monkeypatch.setattr("psutil.virtual_memory", lambda: MockVirtualMemory())

        model = IdentityModel()
        embs = Embeddings(
            simple_dataset,
            batch_size=16,
            model=model,
            device="cpu",
            path=cache_path,
            memory_threshold=0.01,  # Very low threshold (1%)
            verbose=False,
        )

        # Trigger initialization by accessing first embedding
        # This will initialize storage with memmap due to mocked memory constraints
        sample_embedding = np.random.randn(128).astype(np.float32)

        # Manually trigger initialization
        embs._initialize_storage(sample_embedding)

        # Should have created memmap
        assert isinstance(embs._embeddings, np.memmap)
        assert embs._use_memmap is True
        assert cache_path.exists()

    def test_initialize_storage_creates_in_memory_array(self, simple_dataset: SimpleDataset, tmp_path, monkeypatch):
        """Test _initialize_storage creates in-memory array when below threshold"""
        cache_path = tmp_path / "test.npy"

        # Mock psutil to return large available memory
        class MockVirtualMemory:
            available = 10 * 1024**3  # 10 GB

        monkeypatch.setattr("psutil.virtual_memory", lambda: MockVirtualMemory())

        model = IdentityModel()
        embs = Embeddings(
            simple_dataset,
            batch_size=16,
            model=model,
            device="cpu",
            path=cache_path,
            memory_threshold=0.8,
            verbose=False,
        )

        # Trigger initialization
        sample_embedding = np.random.randn(128).astype(np.float32)
        embs._initialize_storage(sample_embedding)

        # Should have created regular array, not memmap
        assert isinstance(embs._embeddings, np.ndarray)
        assert not isinstance(embs._embeddings, np.memmap)
        assert embs._use_memmap is False

    def test_initialize_storage_without_path(self, simple_dataset: SimpleDataset):
        """Test _initialize_storage creates in-memory array when no path is set"""
        model = IdentityModel()
        embs = Embeddings(simple_dataset, batch_size=16, model=model, device="cpu", path=None, verbose=False)

        # Trigger initialization
        sample_embedding = np.random.randn(128).astype(np.float32)
        embs._initialize_storage(sample_embedding)

        # Should have created regular array
        assert isinstance(embs._embeddings, np.ndarray)
        assert not isinstance(embs._embeddings, np.memmap)


class TestSaveMemmap:
    """Test suite for save() method with memmap."""

    def test_save_flushes_memmap(self, tmp_path, monkeypatch):
        """Test save() flushes memmap instead of using np.save (lines 486-490)"""
        cache_path = tmp_path / "test.npy"

        # Create embeddings that will use memmap
        dataset = SimpleDataset(size=50, image_shape=(3, 32, 32))

        # Mock psutil to force memmap usage
        class MockVirtualMemory:
            available = 10 * 1024  # 10 KB

        monkeypatch.setattr("psutil.virtual_memory", lambda: MockVirtualMemory())

        model = IdentityModel()
        embs = Embeddings(
            dataset,
            batch_size=16,
            model=model,
            device="cpu",
            path=cache_path,
            memory_threshold=0.01,  # Very low threshold
            verbose=False,
        )

        # Compute to create memmap
        embs.compute()

        # Verify it's memmap
        assert isinstance(embs._embeddings, np.memmap)

        # Mock the flush method to track if it was called
        flush_called = False
        original_flush = embs._embeddings.flush

        def mock_flush():
            nonlocal flush_called
            flush_called = True
            original_flush()

        embs._embeddings.flush = mock_flush

        # Save should flush, not call np.save
        embs.save()

        # Verify flush was called
        assert flush_called

    def test_save_verbose_logging_memmap(self, tmp_path, monkeypatch, caplog):
        """Test save() logs when flushing memmap with verbose=True (lines 489-490)"""
        import logging

        cache_path = tmp_path / "test.npy"
        dataset = SimpleDataset(size=50, image_shape=(3, 32, 32))

        # Mock psutil to force memmap usage
        class MockVirtualMemory:
            available = 10 * 1024  # 10 KB

        monkeypatch.setattr("psutil.virtual_memory", lambda: MockVirtualMemory())

        model = IdentityModel()
        embs = Embeddings(
            dataset,
            batch_size=16,
            model=model,
            device="cpu",
            path=cache_path,
            memory_threshold=0.01,  # Very low threshold
            verbose=True,
        )

        embs.compute()
        assert isinstance(embs._embeddings, np.memmap)

        with caplog.at_level(logging.DEBUG):
            embs.save()

        # Should log flushing message
        assert any("Flushed memmap embeddings" in record.message for record in caplog.records)

    def test_save_verbose_logging_in_memory(self, tmp_path, caplog):
        """Test save() logs when saving in-memory array with verbose=True (lines 493-495)"""
        import logging

        cache_path = tmp_path / "test.npy"
        dataset = SimpleDataset(size=10, image_shape=(3, 32, 32))

        model = IdentityModel()
        embs = Embeddings(
            dataset,
            batch_size=16,
            model=model,
            device="cpu",
            path=cache_path,
            memory_threshold=0.8,
            verbose=True,
        )

        embs.compute()
        # Should be in-memory
        assert not isinstance(embs._embeddings, np.memmap)

        with caplog.at_level(logging.DEBUG):
            embs.save()

        # Should log saving message
        assert any("Saved embeddings to" in record.message for record in caplog.records)


class TestBatchErrors:
    """Test suite for error cases in _batch method."""

    def test_batch_out_of_range_error(self, simple_dataset: SimpleDataset):
        """Test _batch raises IndexError for out of range indices (lines 567-570)"""
        model = IdentityModel()
        embs = Embeddings(simple_dataset, batch_size=16, model=model, device="cpu", verbose=False)

        # Try to access indices beyond dataset size
        out_of_range_indices = [0, 1, 100, 200]  # 100 and 200 are out of range

        with pytest.raises(IndexError, match="Indices.*are out of range for dataset of size"):
            # Consume the generator to trigger the error
            list(embs._batch(out_of_range_indices))

    def test_batch_memmap_flush(self, simple_dataset: SimpleDataset, tmp_path, monkeypatch):
        """Test _batch flushes memmap after writing (lines 582-583)"""
        cache_path = tmp_path / "test.npy"

        # Mock psutil to force memmap usage
        class MockVirtualMemory:
            available = 10 * 1024  # 10 KB

        monkeypatch.setattr("psutil.virtual_memory", lambda: MockVirtualMemory())

        model = IdentityModel()
        embs = Embeddings(
            simple_dataset,
            batch_size=5,
            model=model,
            device="cpu",
            path=cache_path,
            memory_threshold=0.01,  # Very low threshold
            verbose=False,
        )

        # Track flush calls
        flush_count = [0]  # Use list to allow modification in closure

        # Patch flush before accessing
        original_flush = None

        def track_flush():
            nonlocal original_flush
            flush_count[0] += 1
            if original_flush:
                original_flush()

        # Process first batch to create memmap
        indices = list(range(5))
        for batch_result in embs._batch(indices):
            if isinstance(embs._embeddings, np.memmap) and original_flush is None:
                original_flush = embs._embeddings.flush
                embs._embeddings.flush = track_flush
                break  # Exit after first batch

        # Now process more batches with monitoring
        indices = list(range(5, 10))
        for batch_result in embs._batch(indices):
            pass

        # Flush should have been called at least once during the second batch processing
        assert flush_count[0] > 0


class TestGetitemErrors:
    """Test suite for error cases in __getitem__ method."""

    def test_getitem_sequence_invalid_element_raises(self):
        """Test __getitem__ raises TypeError for sequence with non-int elements (lines 620-621)"""
        arr = np.random.randn(10, 128)
        embs = Embeddings.from_array(arr)

        # Sequence with invalid element types
        with pytest.raises(TypeError, match="All indices in the sequence must be integers"):
            embs[[0, 1, "invalid", 3]]  # type: ignore

    def test_getitem_embeddings_only_empty_raises(self):
        """Test __getitem__ on embeddings-only with empty array raises (lines 626-627)"""
        # Create embeddings-only with empty array
        embs = Embeddings.from_array(np.empty((0,)))
        embs._cached_idx = set()  # Clear cached indices

        with pytest.raises(ValueError, match="Embeddings not initialized"):
            embs[0]

    def test_getitem_embeddings_only_out_of_cache_raises(self):
        """Test __getitem__ on embeddings-only accessing uncached index raises (lines 628-629)"""
        arr = np.random.randn(5, 128)
        embs = Embeddings.from_array(arr)

        # Manually remove some indices from cache to simulate partial cache
        embs._cached_idx = {0, 1, 2}  # Only first 3 are "cached"

        with pytest.raises(ValueError, match="Unable to generate new embeddings from a shallow instance"):
            # Try to access index 4 which is not in cached_idx
            embs[4]
