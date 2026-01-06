import pytest
import torch

import dataeval.config as config


class TestSeed:
    def test_seed(self):
        config.set_seed(42)
        assert config._seed == 42
        assert config.get_seed() == 42
        config.set_seed(0)
        assert config._seed == 0
        assert config.get_seed() == 0


class TestDevice:
    def test_device(self):
        config.set_device("cpu")
        assert config._device == torch.device("cpu")
        assert config.get_device() == torch.device("cpu")
        config.set_device(None)
        assert config._device is None
        if hasattr(torch, "get_default_device"):
            assert config.get_device() == torch.get_default_device()

    def test_device_context(self):
        config.set_device(None)
        assert config._device is None
        with config.use_device("cpu"):
            assert config.get_device() == torch.device("cpu")
        assert config._device is None


class TestBatchSize:
    def test_batch_size(self):
        config.set_batch_size(64)
        assert config._batch_size == 64
        assert config.get_batch_size() == 64
        config.set_batch_size(32)
        assert config._batch_size == 32
        assert config.get_batch_size() == 32

    def test_batch_size_context(self):
        config.set_batch_size(16)
        assert config._batch_size == 16
        with config.use_batch_size(128):
            assert config.get_batch_size() == 128
        assert config._batch_size == 16

    def test_batch_size_not_set(self):
        config.set_batch_size(None)
        with pytest.raises(ValueError, match="No batch_size provided"):
            config.get_batch_size()

    def test_batch_size_less_than_one(self):
        config.set_batch_size(0)
        with pytest.raises(ValueError, match="Provided batch_size must be greater than 1"):
            config.get_batch_size()

        config.set_batch_size(-1)
        with pytest.raises(ValueError, match="Provided batch_size must be greater than 1"):
            config.get_batch_size()
