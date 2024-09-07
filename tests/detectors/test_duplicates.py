import numpy as np
import pytest

from dataeval._internal.detectors.duplicates import Duplicates
from dataeval._internal.flags import ImageStat
from dataeval._internal.metrics.stats import imagestats


def get_dataset(count: int, channels: int):
    return [np.random.random((channels, 16, 16)) for _ in range(count)]


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
        stats = imagestats(data, ImageStat.ALL_HASHES)
        dupes = Duplicates(only_exact=True)
        results = dupes.evaluate(stats)
        assert len(results.exact) == 20
        assert len(results.near) == 0

    def test_duplicates_with_stats_no_xxhash(self):
        data = np.random.random((20, 3, 16, 16))
        data = np.concatenate((data, data, data + 0.001))
        stats = imagestats(data, ImageStat.PCHASH)
        dupes = Duplicates()
        with pytest.raises(ValueError):
            dupes.evaluate(stats)

    def test_duplicates_with_stats_no_pchash(self):
        data = np.random.random((20, 3, 16, 16))
        data = np.concatenate((data, data, data + 0.001))
        stats = imagestats(data, ImageStat.XXHASH)
        dupes = Duplicates()
        with pytest.raises(ValueError):
            dupes.evaluate(stats)

    def test_duplicates_with_stats_no_pchash_only_exact(self):
        data = np.random.random((20, 3, 16, 16))
        data = np.concatenate((data, data, data + 0.001))
        stats = imagestats(data, ImageStat.XXHASH)
        dupes = Duplicates(only_exact=True)
        results = dupes.evaluate(stats)
        assert len(results.exact) == 20
        assert len(results.near) == 0
