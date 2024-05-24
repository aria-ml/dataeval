import numpy as np
import pytest

from daml._internal.metrics.hash import pchash, xxhash


def test_xxhash():
    result = xxhash(np.full((28, 28), 20))
    assert len(result) > 0


def test_xxhash_same():
    result1 = xxhash(np.full((28, 28), 20))
    result2 = xxhash(np.full((28, 28), 20))
    assert result1 == result2


def test_xxhash_diff():
    result1 = xxhash(np.full((28, 28), 20))
    result2 = xxhash(np.full((28, 28), 19))
    assert result1 != result2


def test_pchash():
    result = pchash(np.full((28, 28), 20))
    assert len(result) > 0


def test_pchash_image_too_small():
    with pytest.raises(ValueError):
        pchash(np.full((2, 2), 20))


def test_pchash_same():
    result1 = pchash(np.full((28, 28), 20))
    result2 = pchash(np.full((28, 28), 20))
    assert result1 == result2


def test_pchash_near_same():
    result1 = pchash(np.full((28, 28), 20))
    result2 = pchash(np.random.randint(19, 20, (28, 28)))
    assert result1 == result2


def test_pchash_diff():
    result1 = pchash(np.random.randint(64, 255, (28, 28)))
    result2 = pchash(np.random.randint(0, 191, (28, 28)))
    assert result1 != result2
