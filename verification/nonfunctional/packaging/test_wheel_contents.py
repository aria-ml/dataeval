"""Verify the installed package contains all expected files and subpackages.

Maps to meta repo test cases:
  - TC-1.7: Package manager installation (wheel integrity)
"""

import importlib

import pytest


@pytest.mark.test_case("1-7")
class TestWheelContents:
    """Verify the installed package structure matches expectations."""

    def test_py_typed_marker_exists(self):
        """PEP 561: py.typed marker must be present for type checker support."""

        package_dir = importlib.resources.files("dataeval")
        py_typed = package_dir / "py.typed"
        assert py_typed.is_file(), "py.typed marker not found in installed package"

    def test_expected_subpackages_present(self):
        """All documented subpackages are installed."""
        expected = [
            "dataeval.bias",
            "dataeval.core",
            "dataeval.quality",
            "dataeval.selection",
            "dataeval.shift",
            "dataeval.extractors",
            "dataeval.utils",
        ]
        for pkg in expected:
            spec = importlib.util.find_spec(pkg)
            assert spec is not None, f"Subpackage '{pkg}' not found in installed package"

    def test_no_test_files_in_package(self):
        """Test files should not be included in the distributed package."""

        package_dir = importlib.resources.files("dataeval")
        # Walk the package tree looking for test files
        test_files = []
        for item in package_dir.iterdir():
            if hasattr(item, "name") and item.name.startswith("test_"):
                test_files.append(item.name)
        assert test_files == [], f"Test files found in package: {test_files}"
