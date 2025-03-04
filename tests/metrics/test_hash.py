import numpy as np
import pytest

from dataeval.metrics.stats._hashstats import pchash, xxhash


@pytest.mark.required
class TestXxHash:
    def test_xxhash(self):
        result = xxhash(np.full((28, 28), 20))
        assert len(result) > 0

    def test_xxhash_same(self):
        result1 = xxhash(np.full((28, 28), 20))
        result2 = xxhash(np.full((28, 28), 20))
        assert result1 == result2

    def test_xxhash_diff(self):
        result1 = xxhash(np.full((28, 28), 20))
        result2 = xxhash(np.full((28, 28), 19))
        assert result1 != result2


@pytest.mark.required
class TestPcHash:
    def test_pchash(self):
        result = pchash(np.full((28, 28), 20))
        assert len(result) > 0

    def test_pchash_image_too_small(self):
        result = pchash(np.full((2, 2), 20))
        assert result == ""

    def test_pchash_same(self):
        result1 = pchash(np.full((28, 28), 20))
        result2 = pchash(np.full((28, 28), 20))
        assert result1 == result2

    def test_pchash_near_same(self):
        result1 = pchash(np.full((28, 28), 20))
        result2 = pchash(np.random.randint(19, 20, (28, 28)))
        assert result1 == result2

    def test_pchash_diff(self):
        result1 = pchash(np.random.randint(64, 255, (28, 28)))
        result2 = pchash(np.random.randint(0, 191, (28, 28)))
        assert result1 != result2
