import pytest
import torch

import dataeval.config as config
from dataeval.config import resolve_batch_size


class TestSeed:
    def test_seed(self):
        original_seed = config.get_seed()
        try:
            config.set_seed(42)
            assert config._config.seed == 42
            assert config.get_seed() == 42
            config.set_seed(0)
            assert config._config.seed == 0
            assert config.get_seed() == 0
        finally:
            config.set_seed(original_seed, all_generators=True)


class TestDevice:
    def test_device(self):
        original_device = config._config.device
        try:
            config.set_device("cpu")
            assert config._config.device == torch.device("cpu")
            assert config.get_device() == torch.device("cpu")
            config.set_device(None)
            assert config._config.device is None
            if hasattr(torch, "get_default_device"):
                assert config.get_device() == torch.get_default_device()
        finally:
            config._config.device = original_device

    def test_device_context(self):
        original_device = config._config.device
        try:
            config.set_device(None)
            assert config._config.device is None
            with config.use_device("cpu"):
                assert config.get_device() == torch.device("cpu")
            assert config._config.device is None
        finally:
            config._config.device = original_device


class TestBatchSize:
    def test_batch_size(self):
        original_batch_size = config._config.batch_size
        try:
            config.set_batch_size(64)
            assert config._config.batch_size == 64
            assert config.get_batch_size() == 64
            config.set_batch_size(32)
            assert config._config.batch_size == 32
            assert config.get_batch_size() == 32
        finally:
            config._config.batch_size = original_batch_size

    def test_batch_size_context(self):
        original_batch_size = config._config.batch_size
        try:
            config.set_batch_size(16)
            assert config._config.batch_size == 16
            with config.use_batch_size(128):
                assert config.get_batch_size() == 128
            assert config._config.batch_size == 16
        finally:
            config._config.batch_size = original_batch_size

    def test_batch_size_not_set(self):
        original_batch_size = config._config.batch_size
        try:
            config.set_batch_size(None)
            with pytest.raises(ValueError, match="No batch_size provided"):
                config.get_batch_size()
        finally:
            config._config.batch_size = original_batch_size

    def test_batch_size_less_than_one(self):
        original_batch_size = config._config.batch_size
        try:
            with pytest.raises(ValueError, match="batch_size must be greater than 0"):
                config.set_batch_size(0)

            with pytest.raises(ValueError, match="batch_size must be greater than 0"):
                config.set_batch_size(-1)
        finally:
            config._config.batch_size = original_batch_size


class TestMaxProcesses:
    def test_set_max_processes_zero_raises_error(self):
        """Test that setting processes to 0 raises ValueError (lines 182-183)."""
        with pytest.raises(ValueError, match="processes cannot be zero"):
            config.set_max_processes(0)

    def test_set_max_processes_valid(self):
        """Test setting valid max_processes value (lines 184-185)."""
        original_processes = config._config.max_processes
        try:
            config.set_max_processes(4)
            assert config._config.max_processes == 4
            assert config.get_max_processes() == 4

            config.set_max_processes(-1)
            assert config._config.max_processes == -1
            assert config.get_max_processes() == -1

            config.set_max_processes(None)
            assert config._config.max_processes is None
            assert config.get_max_processes() is None
        finally:
            config._config.max_processes = original_processes


class TestSetSeedAllGenerators:
    def test_set_seed_none_with_all_generators(self):
        """Test set_seed with None and all_generators=True (lines 229-230)."""
        original_seed = config.get_seed()
        try:
            # This should call torch.seed() and torch.cuda.seed_all()
            config.set_seed(None, all_generators=True)
            assert config.get_seed() is None
        finally:
            config.set_seed(original_seed, all_generators=True)

    def test_set_seed_value_with_all_generators(self):
        """Test set_seed with a value and all_generators=True (lines 226-227)."""
        original_seed = config.get_seed()
        try:
            config.set_seed(42, all_generators=True)
            assert config.get_seed() == 42
        finally:
            config.set_seed(original_seed, all_generators=True)


@pytest.mark.required
class TestResolveBatchSize:
    def test_first_non_none_wins(self):
        assert resolve_batch_size(8, 32) == 8
        assert resolve_batch_size(None, 32) == 32
        assert resolve_batch_size(None, None, 5) == 5

    def test_falls_back_to_global(self):
        original = config._config.batch_size
        config.set_batch_size(64)
        try:
            assert resolve_batch_size(None, None) == 64
        finally:
            config.set_batch_size(original)

    def test_raises_when_nothing_set(self):
        original = config._config.batch_size
        config.set_batch_size(None)
        try:
            with pytest.raises(ValueError, match="No batch_size provided"):
                resolve_batch_size(None, None)
        finally:
            config.set_batch_size(original)

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError, match="greater than 0"):
            resolve_batch_size(0, 32)
