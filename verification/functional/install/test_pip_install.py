"""Verify that DataEval installs correctly via pip/uv and core functionality is available.

Maps to meta repo test cases:
  - TC-1.1: Python version compatibility
  - TC-1.7: Package manager installation
"""

import pytest


@pytest.mark.test_case("1-1")
@pytest.mark.test_case("1-7")
class TestPipInstall:
    """Verify the package is importable and functional after pip installation."""

    def test_import_dataeval(self):
        import dataeval  # noqa: F401

    def test_version_is_set(self):
        from dataeval import __version__

        assert __version__ != "unknown"

    def test_core_modules_importable(self):
        from dataeval import config, flags, protocols, types  # noqa: F401

    def test_subpackages_importable(self):
        from dataeval import bias, core, quality, selection, shift  # noqa: F401

    def test_basic_smoke_test(self):
        """Smoke test: a simple end-to-end calculation completes without error."""
        import numpy as np

        from dataeval.core import label_stats

        labels = np.array([0, 0, 1, 1, 2, 2])
        result = label_stats(labels)
        assert result is not None
