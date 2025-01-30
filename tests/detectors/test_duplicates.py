import numpy as np
import pytest

from dataeval.detectors.linters.duplicates import Duplicates
from dataeval.metrics.stats._hashstats import hashstats


@pytest.mark.required
class TestDuplicates:
    def test_duplicates(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates()
        results = dupes.evaluate(np.concatenate((data, data)))
        assert len(results.exact) == 20
        assert len(results.near) == 0

    def test_near_duplicates(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates()
        results = dupes.evaluate(np.concatenate((data, data + 0.001)))
        assert len(results.exact) < 20
        assert len(results.near) > 0

    def test_duplicates_only_exact(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates(only_exact=True)
        results = dupes.evaluate(np.concatenate((data, data, data + 0.001)))
        assert len(results.exact) == 20
        assert len(results.near) == 0

    def test_duplicates_with_stats(self):
        data = np.random.random((20, 3, 16, 16))
        data = np.concatenate((data, data, data + 0.001))
        stats = hashstats(data)
        dupes = Duplicates(only_exact=True)
        results = dupes.from_stats(stats)
        assert len(results.exact) == 20
        assert len(results.near) == 0

    def test_get_duplicates_multiple_stats(self):
        ones = np.ones((1, 16, 16))
        zeros = np.zeros((1, 16, 16))
        data1 = np.concatenate((ones, zeros, ones, zeros, ones))
        data2 = np.concatenate((zeros, ones, zeros))
        data3 = np.concatenate((zeros + 0.001, ones - 0.001))
        dupes1 = hashstats(data1)
        dupes2 = hashstats(data2)
        dupes3 = hashstats(data3)

        dupes = Duplicates()
        results = dupes.from_stats((dupes1, dupes2, dupes3))
        assert len(results.exact) == 2
        assert results.exact[0] == {0: [0, 2, 4], 1: [1]}
        assert len(results.near) == 2
        assert results.near[0] == {0: [0, 2, 4], 1: [1], 2: [1]}

    def test_duplicates_invalid_stats(self):
        dupes = Duplicates()
        with pytest.raises(TypeError):
            dupes.from_stats(1234)  # type: ignore
        with pytest.raises(TypeError):
            dupes.from_stats([1234])  # type: ignore

    def test_duplicates_ignore_non_duplicate_too_small(self):
        dupes = Duplicates()
        images = [np.random.random((3, 16, 16)) for _ in range(20)]
        images[3] = np.zeros((3, 5, 5))
        images[5] = np.ones((3, 5, 5))
        results = dupes.evaluate(images)
        assert len(results.near) == 0

    def test_duplicates_ignore_duplicate_too_small(self):
        dupes = Duplicates()
        images = [np.random.random((3, 16, 16)) for _ in range(20)]
        images[3] = np.zeros((3, 5, 5))
        images[5] = np.zeros((3, 5, 5))
        results = dupes.evaluate(images)
        assert len(results.near) == 0
