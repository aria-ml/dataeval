import subprocess
import sys
import textwrap

import pytest

import dataeval.shift


@pytest.mark.required
class TestDriftKnownDomainsExport:
    def test_is_importable_from_the_package(self):
        from dataeval.shift import DriftKnownDomains  # noqa: F401

        assert "DriftKnownDomains" in dataeval.shift.__all__

    def test_instantiating_warns_once(self):
        # The warning is once per process, and other tests have likely spent it, so count in a fresh interpreter.
        code = textwrap.dedent(
            """
            import warnings
            from dataeval.exceptions import ExperimentalWarning
            from dataeval.shift import DriftKnownDomains

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                DriftKnownDomains()
                DriftKnownDomains()
            ours = [w for w in caught if issubclass(w.category, ExperimentalWarning)]
            assert len(ours) == 1 and "'DriftKnownDomains' is experimental" in str(ours[0].message), ours
            """
        )
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
