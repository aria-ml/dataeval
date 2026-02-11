"""Verify that dataset selection operators compose and function correctly.

Maps to meta repo test cases:
  - TC-6.1: Dataset selection and filtering
"""

import numpy as np
import pytest


@pytest.mark.test_case("6-1")
class TestDatasetSelection:
    """Verify Select, Indices, Limit, Shuffle, Reverse, and ClassFilter."""

    def test_select_with_limit(self):
        from dataeval.selection import Limit, Select
        from verification.helpers import SimpleImageDataset

        images = np.random.default_rng(0).random((20, 3, 8, 8)).astype(np.float32)
        dataset = SimpleImageDataset(images)
        selected = Select(dataset, Limit(5))
        assert len(selected) == 5

    def test_select_with_indices(self):
        from dataeval.selection import Indices, Select
        from verification.helpers import SimpleImageDataset

        images = np.random.default_rng(0).random((20, 3, 8, 8)).astype(np.float32)
        dataset = SimpleImageDataset(images)
        selected = Select(dataset, Indices([0, 5, 10, 15]))
        assert len(selected) == 4

    def test_select_with_shuffle(self):
        from dataeval.selection import Select, Shuffle
        from verification.helpers import SimpleImageDataset

        images = np.random.default_rng(0).random((20, 3, 8, 8)).astype(np.float32)
        dataset = SimpleImageDataset(images)
        selected = Select(dataset, Shuffle(seed=42))
        assert len(selected) == 20

    def test_select_with_reverse(self):
        from dataeval.selection import Reverse, Select
        from verification.helpers import SimpleImageDataset

        images = np.random.default_rng(0).random((10, 3, 8, 8)).astype(np.float32)
        dataset = SimpleImageDataset(images)
        selected = Select(dataset, Reverse())
        # First item of reversed should be last item of original
        np.testing.assert_array_equal(selected[0], images[9])

    def test_select_composes_multiple_operations(self):
        from dataeval.selection import Limit, Select, Shuffle
        from verification.helpers import SimpleImageDataset

        images = np.random.default_rng(0).random((20, 3, 8, 8)).astype(np.float32)
        dataset = SimpleImageDataset(images)
        selected = Select(dataset, [Limit(10), Shuffle(seed=42)])
        assert len(selected) == 10

    def test_select_is_iterable(self):
        from dataeval.selection import Limit, Select
        from verification.helpers import SimpleImageDataset

        images = np.random.default_rng(0).random((10, 3, 8, 8)).astype(np.float32)
        dataset = SimpleImageDataset(images)
        selected = Select(dataset, Limit(3))
        items = list(selected)
        assert len(items) == 3
