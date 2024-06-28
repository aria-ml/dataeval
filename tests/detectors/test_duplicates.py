import numpy as np

from daml._internal.detectors.duplicates import Duplicates


def get_dataset(count: int, channels: int):
    return [np.random.random((channels, 16, 16)) for _ in range(count)]


class TestDuplicates:
    def test_duplicates(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates(np.concatenate((data, data)))
        results = dupes.evaluate()
        assert len(results["exact"]) == 20
        assert len(results["near"]) == 0

    def test_near_duplicates(self):
        data = np.random.random((20, 3, 16, 16))
        dupes = Duplicates(np.concatenate((data, data + 0.001)))
        results = dupes.evaluate()
        assert len(results["exact"]) < 20
        assert len(results["near"]) > 0
