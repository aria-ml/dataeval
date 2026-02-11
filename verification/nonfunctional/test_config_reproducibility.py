"""Verify global configuration and reproducibility controls.

Maps to meta repo test cases:
  - TC-11.1: Configuration and reproducibility
"""

import pytest


@pytest.mark.test_case("11-1")
class TestConfigReproducibility:
    """Verify dataeval.config functions and reproducibility."""

    def test_set_seed_and_get_seed(self):
        from dataeval import config

        config.set_seed(42)
        assert config.get_seed() == 42
        config.set_seed(None)

    def test_set_device_and_get_device(self):
        from dataeval import config

        config.set_device("cpu")
        device = config.get_device()
        assert "cpu" in str(device)
        config.set_device(None)

    def test_use_device_context_manager(self):
        from dataeval import config

        original = config.get_device()
        with config.use_device("cpu"):
            assert "cpu" in str(config.get_device())
        # Original restored
        assert str(config.get_device()) == str(original)

    def test_set_max_processes(self):
        from dataeval import config

        config.set_max_processes(2)
        assert config.get_max_processes() == 2
        config.set_max_processes(None)

    def test_use_max_processes_context_manager(self):
        from dataeval import config

        config.set_max_processes(1)
        with config.use_max_processes(4):
            assert config.get_max_processes() == 4
        assert config.get_max_processes() == 1
        config.set_max_processes(None)

    def test_seed_produces_reproducible_results(self):
        import numpy as np

        from dataeval import config
        from dataeval.core import label_stats

        labels = np.array([0, 0, 1, 1, 2, 2])

        config.set_seed(123)
        result1 = label_stats(labels)

        config.set_seed(123)
        result2 = label_stats(labels)

        assert result1 == result2
        config.set_seed(None)
