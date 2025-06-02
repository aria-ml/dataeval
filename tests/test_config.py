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
