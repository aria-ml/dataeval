"""Verify global configuration and reproducibility controls.

Maps to meta repo test cases:
  - TC-11.1: Configuration and reproducibility
"""

import pytest


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

    def test_set_batch_size_and_get_batch_size(self):
        from dataeval import config

        config.set_batch_size(32)
        try:
            assert config.get_batch_size() == 32
            assert config.get_batch_size(8) == 8  # an explicit override wins over the global
        finally:
            config.set_batch_size(None)
        with pytest.raises(ValueError, match="batch_size"):
            config.get_batch_size()  # unset and no override

    def test_use_batch_size_context_manager(self):
        from dataeval import config

        config.set_batch_size(16)
        try:
            with config.use_batch_size(64):
                assert config.get_batch_size() == 64
            assert config.get_batch_size() == 16
        finally:
            config.set_batch_size(None)

    def test_use_scopes_nest_and_restore_after_error(self):
        """Device, batch size, and max processes are all restored together, even when the scope raises."""
        from dataeval import config

        config.set_batch_size(16)
        config.set_max_processes(1)
        original_device = str(config.get_device())
        try:
            try:
                with config.use_device("cpu"), config.use_batch_size(8), config.use_max_processes(3):
                    assert config.get_batch_size() == 8
                    assert config.get_max_processes() == 3
                    raise RuntimeError("boom")
            except RuntimeError:
                pass
            assert config.get_batch_size() == 16
            assert config.get_max_processes() == 1
            assert str(config.get_device()) == original_device
        finally:
            config.set_batch_size(None)
            config.set_max_processes(None)

    def test_dataeval_logging_uses_standard_logging_framework(self):
        """A plain ``logging`` handler on the ``dataeval`` logger receives DataEval's records."""
        import logging

        import numpy as np

        from dataeval.quality import Duplicates

        class Collect(logging.Handler):
            def __init__(self):
                super().__init__(logging.DEBUG)
                self.records: list[logging.LogRecord] = []

            def emit(self, record):
                self.records.append(record)

        logger = logging.getLogger("dataeval")
        handler, previous_level = Collect(), logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            Duplicates().evaluate(np.random.default_rng(0).random((20, 3, 16, 16)).astype(np.float32))
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

        assert handler.records
        names = {record.name for record in handler.records}
        assert all(name == "dataeval" or name.startswith("dataeval.") for name in names)
        assert {"dataeval.quality", "dataeval.core"} <= names  # curated subsystem loggers
        assert all(isinstance(record.getMessage(), str) for record in handler.records)

    def test_core_evaluator_falls_back_to_cpu_without_cuda(self, monkeypatch):
        """With CUDA hidden, a torch-backed evaluator runs on CPU without error."""
        torch = pytest.importorskip("torch")
        import numpy as np

        from dataeval import config
        from dataeval.shift import DriftMMD

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        config.set_device(None)
        config.set_batch_size(16)
        try:
            assert config.get_device().type == "cpu"
            rng = np.random.default_rng(0)
            ref = rng.standard_normal((50, 8)).astype(np.float32)
            detector = DriftMMD(n_permutations=20).fit(ref)
            result = detector.predict(rng.standard_normal((20, 8)).astype(np.float32) + 10.0)
            assert result.drifted is True
        finally:
            config.set_batch_size(None)
