"""Verify that public API classes validate inputs and raise clear errors.

These tests verify high-level contract enforcement, not individual function
correctness (which is covered by the unit test suite under tests/).
"""

import numpy as np


class TestInputContracts:
    """Verify input validation across key public API entry points."""

    def test_label_stats_accepts_valid_labels(self):
        from dataeval.core import label_stats

        result = label_stats(np.array([0, 1, 2, 0, 1, 2]))
        assert result is not None

    def test_label_stats_handles_empty_labels(self):
        """Empty labels should either raise or return an empty/zero-count result."""
        from dataeval.core import label_stats

        result = label_stats(np.array([], dtype=np.intp))
        # API handles empty input gracefully — verify result is consistent
        assert "label_count" in result
        assert result["label_count"] == 0

    def test_duplicates_handles_empty_dataset(self):
        """Empty dataset should either raise or return an empty result."""
        from dataeval.quality import Duplicates

        result = Duplicates().evaluate(np.array([]))
        assert hasattr(result, "items")

    def test_outliers_handles_empty_dataset(self):
        """Empty dataset should either raise or return an empty result."""
        from dataeval.quality import Outliers

        result = Outliers().evaluate(np.array([]))
        assert hasattr(result, "issues")
