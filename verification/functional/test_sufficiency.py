"""Verify that the Sufficiency analysis class is importable and configurable.

Maps to meta repo test cases:
  - TC-8.1: Data sufficiency analysis
"""

import pytest


@pytest.mark.test_case("8-1")
class TestSufficiency:
    """Verify Sufficiency class importability and configuration."""

    def test_sufficiency_importable(self):
        from dataeval.performance import Sufficiency  # noqa: F401

    def test_sufficiency_output_importable(self):
        from dataeval.performance import SufficiencyOutput  # noqa: F401

    def test_sufficiency_config_exists(self):
        from dataeval.performance import Sufficiency

        assert hasattr(Sufficiency, "Config")

    def test_sufficiency_config_has_expected_fields(self):
        from dataeval.performance import Sufficiency

        config = Sufficiency.Config()
        assert hasattr(config, "runs")
        assert hasattr(config, "substeps")
