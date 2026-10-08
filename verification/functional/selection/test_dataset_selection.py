"""Verify that dataset selection operators compose and function correctly.

Maps to meta repo test cases:
  - TC-6.1: Dataset selection and filtering
"""

import numpy as np


class TestDatasetSelection:
    """Verify View, Indices, Limit, Shuffle, Reverse, and ClassFilter."""

    def test_select_with_limit(self):
        from dataeval.data import Limit, View
        from verification.helpers import SimpleImageDataset

        images = np.random.default_rng(0).random((20, 3, 8, 8)).astype(np.float32)
        dataset = SimpleImageDataset(images)
        selected = View(dataset, Limit(5))
        assert len(selected) == 5

    def test_select_with_indices(self):
        from dataeval.data import Indices, View
        from verification.helpers import SimpleImageDataset

        images = np.random.default_rng(0).random((20, 3, 8, 8)).astype(np.float32)
        dataset = SimpleImageDataset(images)
        selected = View(dataset, Indices([0, 5, 10, 15]))
        assert len(selected) == 4

    def test_select_with_shuffle(self):
        from dataeval.data import Shuffle, View
        from verification.helpers import SimpleImageDataset

        images = np.random.default_rng(0).random((20, 3, 8, 8)).astype(np.float32)
        dataset = SimpleImageDataset(images)
        selected = View(dataset, Shuffle(seed=42))
        assert len(selected) == 20

    def test_select_with_reverse(self):
        from dataeval.data import Reverse, View
        from verification.helpers import SimpleImageDataset

        images = np.random.default_rng(0).random((10, 3, 8, 8)).astype(np.float32)
        dataset = SimpleImageDataset(images)
        selected = View(dataset, Reverse())
        # First item of reversed should be last item of original
        np.testing.assert_array_equal(selected[0], images[9])

    def test_select_composes_multiple_operations(self):
        from dataeval.data import Limit, Shuffle, View
        from verification.helpers import SimpleImageDataset

        images = np.random.default_rng(0).random((20, 3, 8, 8)).astype(np.float32)
        dataset = SimpleImageDataset(images)
        selected = View(dataset, [Limit(10), Shuffle(seed=42)])
        assert len(selected) == 10

    def test_select_is_iterable(self):
        from dataeval.data import Limit, View
        from verification.helpers import SimpleImageDataset

        images = np.random.default_rng(0).random((10, 3, 8, 8)).astype(np.float32)
        dataset = SimpleImageDataset(images)
        selected = View(dataset, Limit(3))
        items = list(selected)
        assert len(items) == 3

    def test_shuffle_is_reproducible_for_fixed_seed(self):
        from dataeval.data import Shuffle, View
        from verification.helpers import SimpleImageDataset

        # Each image is filled with its own index so the selection order can be read back.
        images = np.arange(20, dtype=np.float32).reshape(20, 1, 1, 1) * np.ones((20, 3, 4, 4), dtype=np.float32)
        dataset = SimpleImageDataset(images)

        def order(seed):
            view = View(dataset, Shuffle(seed=seed))
            return [int(view[i][0, 0, 0]) for i in range(len(view))]

        first, second = order(7), order(7)
        assert len(first) == 20
        assert sorted(first) == list(range(20))
        assert first != list(range(20))  # order changes
        assert first == second  # same seed, same order
        assert order(8) != first

    def test_class_filter_and_class_balance_on_labeled_dataset(self):
        from dataeval.data import ClassBalance, ClassFilter, View
        from verification.helpers import SimpleICDataset

        labels = np.array([0] * 30 + [1] * 10 + [2] * 20)
        images = np.random.default_rng(0).random((60, 3, 8, 8)).astype(np.float32)
        dataset = SimpleICDataset(images, labels)

        def classes_of(view):
            return np.array([int(np.argmax(view[i][1])) for i in range(len(view))])

        filtered = classes_of(View(dataset, ClassFilter([0, 2])))
        assert len(filtered) == 50
        assert set(filtered) == {0, 2}

        balanced = classes_of(View(dataset, ClassBalance("interclass", num_samples=30)))
        assert len(balanced) == 30
        assert np.bincount(balanced, minlength=3).tolist() == [10, 10, 10]
