import numpy as np

from dataeval._internal.detectors.duplicates import Duplicates


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

    def test_duplicates_no_near(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates(find_near=False)
        results = dupes.evaluate(np.concatenate((data, data, data + 0.001)))
        assert len(results.exact) == 20
        assert len(results.near) == 0

    def test_duplicates_no_exact(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates(find_exact=False)
        results = dupes.evaluate(np.concatenate((data, data, data + 0.001)))
        assert len(results.exact) == 0
        assert len(results.near) > 0
