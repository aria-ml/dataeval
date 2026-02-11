"""Verify type safety infrastructure is in place.

Maps to meta repo test cases:
  - TC-10.1: Type safety (PEP 561, type annotations)
"""

import importlib

import pytest


@pytest.mark.test_case("10-1")
class TestTypeSafety:
    """Verify type annotation infrastructure."""

    def test_py_typed_marker_exists(self):
        """PEP 561: py.typed must be present for downstream type checking."""
        package_dir = importlib.resources.files("dataeval")
        py_typed = package_dir / "py.typed"
        assert py_typed.is_file()

    def test_protocols_module_exports_types(self):
        from dataeval import protocols

        expected = {"Array", "FeatureExtractor", "Dataset", "AnnotatedDataset"}
        exported = set(dir(protocols))
        for name in expected:
            assert name in exported, f"protocols missing {name}"

    def test_types_module_exports_output_bases(self):
        from dataeval import types

        expected = {"Output", "DictOutput"}
        exported = set(dir(types))
        for name in expected:
            assert name in exported, f"types missing {name}"

    def test_public_modules_have_all(self):
        """Key public modules should define __all__ for type checker support."""
        import dataeval

        assert hasattr(dataeval, "__all__")
        assert len(dataeval.__all__) > 0
