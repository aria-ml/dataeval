"""Verify DataEval compatibility with supported Python versions.

Maps to meta repo test cases:
  - TC-1.1: Python version compatibility
"""

import sys

import pytest

SUPPORTED_VERSIONS = [(3, 10), (3, 11), (3, 12), (3, 13), (3, 14)]


@pytest.mark.test_case("1-1")
class TestPythonVersions:
    """Verify the package works on the current Python version and it is within the supported range."""

    def test_running_on_supported_version(self):
        current = (sys.version_info.major, sys.version_info.minor)
        assert current in SUPPORTED_VERSIONS, (
            f"Python {current[0]}.{current[1]} is not in the supported range: "
            f"{', '.join(f'{ma}.{mi}' for ma, mi in SUPPORTED_VERSIONS)}"
        )

    def test_import_succeeds_on_current_version(self):
        import dataeval

        assert dataeval.__version__ != "unknown"

    def test_typing_extensions_available(self):
        """typing_extensions is required for Python <3.12 backports."""
        import typing_extensions  # noqa: F401
