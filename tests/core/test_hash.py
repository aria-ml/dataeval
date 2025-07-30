import numpy as np
import pytest

from dataeval.core._hash import pchash, resize, xxhash


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

    def test_resize_pil(self):
        img = np.random.randint(255, size=(28, 28), dtype=np.uint8)
        resized = resize(img, 16)
        assert resized.shape == (16, 16)

    def test_resize_scipy(self):
        img = np.random.randint(255, size=(28, 28), dtype=np.uint8)
        resized = resize(img, 16, False)
        assert resized.shape == (16, 16)

    def test_resize_method_comparison(self):
        dim = 28
        img = np.zeros((dim, dim), dtype=np.uint8)
        for i in range(dim):
            for j in range(dim):
                img[i, j] = (i + j) / (2 * (dim - 1))
        pil_resized = resize(img, 16)
        scipy_resized = resize(img, 16, False)
        np.testing.assert_equal(pil_resized, scipy_resized)
