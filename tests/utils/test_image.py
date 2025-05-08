import numpy as np
import pytest

from dataeval.metrics.stats._base import BoundingBox
from dataeval.utils._image import (
    BitDepth,
    clip_and_pad,
    edge_filter,
    get_bitdepth,
    normalize_image_shape,
    rescale,
)


@pytest.mark.required
class TestImageUtils:
    def test_get_bitdepth_negatives(self):
        image = np.random.random((3, 28, 28)) - 0.5
        bitdepth = get_bitdepth(image)
        assert bitdepth == BitDepth(0, np.min(image), np.max(image))  # type: ignore

    def test_get_bitdepth_float(self):
        image = np.random.random((3, 28, 28))
        bitdepth = get_bitdepth(image)
        assert bitdepth == BitDepth(1, 0, 1)

    def test_get_bitdepth_8bit(self):
        image = (np.random.random((3, 28, 28)) * (2**8 - 1)).astype(np.uint8)
        bitdepth = get_bitdepth(image)
        assert bitdepth == BitDepth(8, 0, (2**8 - 1))

    def test_get_bitdepth_16bit(self):
        image = (np.random.random((3, 28, 28)) * (2**16 - 1)).astype(np.uint16)
        bitdepth = get_bitdepth(image)
        assert bitdepth == BitDepth(16, 0, (2**16 - 1))

    def test_get_bitdepth_64bit(self):
        image = (np.random.random((3, 28, 28)) * (2**64 - 1)).astype(np.uint64)
        bitdepth = get_bitdepth(image)
        assert bitdepth == BitDepth(32, 0, (2**32 - 1))

    def test_rescale_noop(self):
        image = np.random.random((3, 28, 28))
        scaled = rescale(image)
        assert np.min(scaled) == np.min(image)
        assert np.max(scaled) == np.max(image)

    def test_rescale_8bit(self):
        image = np.random.random((3, 28, 28))
        scaled = rescale(image, 8)
        assert np.min(scaled) == np.min(image) * (2**8 - 1)
        assert np.max(scaled) == np.max(image) * (2**8 - 1)

    def test_rescale_float(self):
        image = (np.random.random((3, 28, 28)) * 255).astype(np.uint8)
        scaled = rescale(image)
        assert np.min(scaled) == np.min(image) / (2**8 - 1)
        assert np.max(scaled) == np.max(image) / (2**8 - 1)

    def test_normalize_image_shape_expand(self):
        image = np.zeros((28, 28))
        normalized = normalize_image_shape(image)
        assert normalized.shape == (1, 28, 28)

    def test_normalize_image_shape_noop(self):
        image = np.zeros((1, 28, 28))
        normalized = normalize_image_shape(image)
        assert normalized.shape == (1, 28, 28)

    def test_normalize_image_shape_slice(self):
        image = np.zeros((10, 3, 28, 28))
        normalized = normalize_image_shape(image)
        assert normalized.shape == (3, 28, 28)

    def test_normalize_image_valueerror(self):
        image = np.zeros(10)
        with pytest.raises(ValueError):
            normalize_image_shape(image)

    def testedge_filter(self):
        image = np.zeros((28, 28))
        edge = edge_filter(image, 0.5)
        np.testing.assert_array_equal(image + 0.5, edge)


@pytest.mark.required
@pytest.mark.parametrize("image", [np.arange(25).reshape(1, 5, 5), np.arange(75).reshape(3, 5, 5)])
class TestClipAndPad:
    def test_inside(self, image):
        result = clip_and_pad(image, BoundingBox(1.2, 1.0, 3.9, 3.1).to_int())
        assert not np.isnan(result).any()

    def test_outside_right_bottom(self, image):
        result = clip_and_pad(image, BoundingBox(2.5, 2.1, 7.2, 6.7).to_int())
        print(result)
        assert np.isnan(result).any()

    def test_outside_left_top(self, image):
        result = clip_and_pad(image, BoundingBox(-2.3, -1.8, 3.0, 2.2).to_int())
        assert np.isnan(result).any()

    def test_outside(self, image):
        result = clip_and_pad(image, BoundingBox(-5.5, -5.4, -2.1, -2.6).to_int())
        assert np.isnan(result).all()
