"""Verify that public API functions return outputs matching documented types.

These tests verify the output type contracts hold, not that individual results
are numerically correct (which is covered by unit tests under tests/).
"""

import numpy as np


class TestOutputContracts:
    """Verify outputs conform to documented return types and structures."""

    def test_label_stats_returns_expected_keys(self):
        from dataeval.core import label_stats

        result = label_stats(np.array([0, 0, 1, 1, 2, 2]))
        assert result is not None
        # label_stats returns a dict-like with these keys
        assert "label_counts_per_class" in result
        assert "class_count" in result
        assert "label_count" in result

    def test_duplicates_returns_items_and_targets(self):
        from dataeval.quality import Duplicates

        data = np.random.default_rng(0).random((20, 3, 16, 16)).astype(np.float32)
        result = Duplicates().evaluate(data)
        assert hasattr(result, "items")
        assert hasattr(result, "targets")
