"""Verify package version and metadata are correctly set.

Maps to meta repo test cases:
  - TC-1.1: Python version compatibility (metadata detection)
"""

from importlib.metadata import metadata

import pytest


@pytest.mark.test_case("1-1")
class TestVersionMetadata:
    """Verify the installed package metadata matches expectations."""

    def test_version_not_unknown(self):
        from dataeval import __version__

        assert __version__ != "unknown", "__version__ should be set by hatch-vcs"

    def test_package_name(self):
        meta = metadata("dataeval")
        assert meta["Name"] == "dataeval"

    def test_requires_python(self):
        meta = metadata("dataeval")
        assert meta["Requires-Python"] is not None
        assert "3.10" in meta["Requires-Python"]

    def test_license_set(self):
        meta = metadata("dataeval")
        # PEP 639 uses License-Expression; older packaging uses License
        license_val = meta.get("License-Expression") or meta.get("License")
        assert license_val is not None, "Neither License-Expression nor License found in metadata"

    def test_project_urls_set(self):
        meta = metadata("dataeval")
        urls = meta.get_all("Project-URL")
        assert urls is not None
        assert len(urls) > 0
