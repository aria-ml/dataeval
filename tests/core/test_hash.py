import numpy as np
import pytest

from dataeval.core._hash import dhash, dhash_d4, phash, phash_d4, resize, xxhash


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
class TestPHash:
    def test_phash(self):
        result = phash(np.full((28, 28), 20))
        assert len(result) > 0

    def test_phash_image_too_small(self):
        result = phash(np.full((2, 2), 20))
        assert result == ""

    def test_phash_same(self):
        result1 = phash(np.full((28, 28), 20))
        result2 = phash(np.full((28, 28), 20))
        assert result1 == result2

    def test_phash_near_same(self):
        result1 = phash(np.full((28, 28), 20))
        result2 = phash(np.random.randint(19, 20, (28, 28)))
        assert result1 == result2

    def test_phash_diff(self):
        result1 = phash(np.random.randint(64, 255, (28, 28)))
        result2 = phash(np.random.randint(0, 191, (28, 28)))
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


@pytest.mark.required
class TestDHash:
    def test_dhash(self):
        result = dhash(np.full((28, 28), 20))
        assert len(result) > 0

    def test_dhash_image_too_small(self):
        result = dhash(np.full((2, 2), 20))
        assert result == ""

    def test_dhash_same(self):
        result1 = dhash(np.full((28, 28), 20))
        result2 = dhash(np.full((28, 28), 20))
        assert result1 == result2

    def test_dhash_diff(self):
        np.random.seed(42)
        result1 = dhash(np.random.randint(64, 255, (28, 28)))
        result2 = dhash(np.random.randint(0, 191, (28, 28)))
        assert result1 != result2


@pytest.mark.required
class TestPHashD4:
    def test_phash_d4(self):
        result = phash_d4(np.full((28, 28), 20))
        assert len(result) > 0

    def test_phash_d4_image_too_small(self):
        result = phash_d4(np.full((2, 2), 20))
        assert result == ""

    def test_phash_d4_same(self):
        result1 = phash_d4(np.full((28, 28), 20))
        result2 = phash_d4(np.full((28, 28), 20))
        assert result1 == result2

    def test_phash_d4_rotation_invariant(self):
        np.random.seed(42)
        img = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        result_original = phash_d4(img)
        result_rot90 = phash_d4(np.rot90(img))
        result_rot180 = phash_d4(np.rot90(img, 2))
        result_rot270 = phash_d4(np.rot90(img, 3))
        assert result_original == result_rot90 == result_rot180 == result_rot270

    def test_phash_d4_flip_invariant(self):
        np.random.seed(42)
        img = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        result_original = phash_d4(img)
        result_fliplr = phash_d4(np.fliplr(img))
        result_flipud = phash_d4(np.flipud(img))
        assert result_original == result_fliplr == result_flipud

    def test_phash_d4_diff(self):
        np.random.seed(42)
        result1 = phash_d4(np.random.randint(64, 255, (28, 28)))
        result2 = phash_d4(np.random.randint(0, 191, (28, 28)))
        assert result1 != result2


@pytest.mark.required
class TestDHashD4:
    def test_dhash_d4(self):
        result = dhash_d4(np.full((28, 28), 20))
        assert len(result) > 0

    def test_dhash_d4_image_too_small(self):
        result = dhash_d4(np.full((2, 2), 20))
        assert result == ""

    def test_dhash_d4_same(self):
        result1 = dhash_d4(np.full((28, 28), 20))
        result2 = dhash_d4(np.full((28, 28), 20))
        assert result1 == result2

    def test_dhash_d4_rotation_invariant(self):
        np.random.seed(42)
        img = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        result_original = dhash_d4(img)
        result_rot90 = dhash_d4(np.rot90(img))
        result_rot180 = dhash_d4(np.rot90(img, 2))
        result_rot270 = dhash_d4(np.rot90(img, 3))
        assert result_original == result_rot90 == result_rot180 == result_rot270

    def test_dhash_d4_flip_invariant(self):
        np.random.seed(42)
        img = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        result_original = dhash_d4(img)
        result_fliplr = dhash_d4(np.fliplr(img))
        result_flipud = dhash_d4(np.flipud(img))
        assert result_original == result_fliplr == result_flipud

    def test_dhash_d4_diff(self):
        np.random.seed(42)
        result1 = dhash_d4(np.random.randint(64, 255, (28, 28)))
        result2 = dhash_d4(np.random.randint(0, 191, (28, 28)))
        assert result1 != result2
