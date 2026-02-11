"""Verify that all documented public API symbols are importable and accessible.

Maps to meta repo test cases:
  - TC-1.1: Python version compatibility (imports across versions)
"""

import importlib
import pkgutil

import pytest


@pytest.mark.test_case("1-1")
class TestPublicAPI:
    """Verify the public API surface is complete and importable."""

    def test_top_level_all_exports(self):
        """Every symbol in dataeval.__all__ is importable."""
        import dataeval

        for name in dataeval.__all__:
            obj = getattr(dataeval, name, None)
            assert obj is not None, f"dataeval.__all__ lists '{name}' but it is not accessible"

    def test_subpackages_discoverable(self):
        """All subpackages under dataeval are importable."""
        import dataeval

        package_path = dataeval.__path__
        subpackages = [
            name
            for importer, name, ispkg in pkgutil.walk_packages(package_path, prefix="dataeval.")
            if ispkg and not name.startswith("dataeval._") and not name.startswith("dataeval.core._")
        ]
        assert len(subpackages) > 0, "No public subpackages found"

        for pkg_name in subpackages:
            mod = importlib.import_module(pkg_name)
            assert mod is not None, f"Could not import subpackage '{pkg_name}'"

    def test_key_classes_exist(self):
        """Core user-facing classes are importable from their documented locations."""
        from dataeval import Embeddings, Metadata  # noqa: F401
        from dataeval.bias import Balance, Diversity, Parity  # noqa: F401
        from dataeval.quality import Duplicates, Outliers  # noqa: F401

    def test_protocols_module_exports(self):
        """The protocols module exports its documented protocol types."""
        from dataeval import protocols

        assert hasattr(protocols, "ImageClassificationDataset")
        assert hasattr(protocols, "ObjectDetectionDataset")
