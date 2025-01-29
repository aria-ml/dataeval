import numpy as np
import pytest

from dataeval.utils._image import edge_filter, get_bitdepth, normalize_image_shape, rescale


def test_get_bitdepth_negatives():
    image = np.random.random((3, 28, 28)) - 0.5
    bitdepth = get_bitdepth(image)
    assert bitdepth == (0, np.min(image), np.max(image))


def test_get_bitdepth_float():
    image = np.random.random((3, 28, 28))
    bitdepth = get_bitdepth(image)
    assert bitdepth == (1, 0, 1)


def test_get_bitdepth_8bit():
    image = (np.random.random((3, 28, 28)) * (2**8 - 1)).astype(np.uint8)
    bitdepth = get_bitdepth(image)
    assert bitdepth == (8, 0, (2**8 - 1))


def test_get_bitdepth_16bit():
    image = (np.random.random((3, 28, 28)) * (2**16 - 1)).astype(np.uint16)
    bitdepth = get_bitdepth(image)
    assert bitdepth == (16, 0, (2**16 - 1))


def test_get_bitdepth_64bit():
    image = (np.random.random((3, 28, 28)) * (2**64 - 1)).astype(np.uint64)
    bitdepth = get_bitdepth(image)
    assert bitdepth == (32, 0, (2**32 - 1))


def test_rescale_noop():
    image = np.random.random((3, 28, 28))
    scaled = rescale(image)
    assert np.min(scaled) == np.min(image)
    assert np.max(scaled) == np.max(image)


def test_rescale_8bit():
    image = np.random.random((3, 28, 28))
    scaled = rescale(image, 8)
    assert np.min(scaled) == np.min(image) * (2**8 - 1)
    assert np.max(scaled) == np.max(image) * (2**8 - 1)


def test_rescale_float():
    image = (np.random.random((3, 28, 28)) * 255).astype(np.uint8)
    scaled = rescale(image)
    assert np.min(scaled) == np.min(image) / (2**8 - 1)
    assert np.max(scaled) == np.max(image) / (2**8 - 1)


def test_normalize_image_shape_expand():
    image = np.zeros((28, 28))
    normalized = normalize_image_shape(image)
    assert normalized.shape == (1, 28, 28)


def test_normalize_image_shape_noop():
    image = np.zeros((1, 28, 28))
    normalized = normalize_image_shape(image)
    assert normalized.shape == (1, 28, 28)


def test_normalize_image_shape_slice():
    image = np.zeros((10, 3, 28, 28))
    normalized = normalize_image_shape(image)
    assert normalized.shape == (3, 28, 28)


def test_normalize_image_valueerror():
    image = np.zeros(10)
    with pytest.raises(ValueError):
        normalize_image_shape(image)


def testedge_filter():
    image = np.zeros((28, 28))
    edge = edge_filter(image, 0.5)
    np.testing.assert_array_equal(image + 0.5, edge)
