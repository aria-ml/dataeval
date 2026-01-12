import numpy as np
import pytest

from dataeval.utils._boundingbox import BoundingBox
from dataeval.utils._image import (
    BitDepth,
    clip_and_pad,
    edge_filter,
    get_bitdepth,
    normalize_image_shape,
    rescale,
    resize,
    to_canonical_grayscale,
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
        result = clip_and_pad(image, BoundingBox(1.2, 1.0, 3.9, 3.1).xyxy_int)
        assert not np.isnan(result).any()

    def test_outside_right_bottom(self, image):
        result = clip_and_pad(image, BoundingBox(2.5, 2.1, 7.2, 6.7).xyxy_int)
        print(result)
        assert np.isnan(result).any()

    def test_outside_left_top(self, image):
        result = clip_and_pad(image, BoundingBox(-2.3, -1.8, 3.0, 2.2).xyxy_int)
        assert np.isnan(result).any()

    def test_outside(self, image):
        result = clip_and_pad(image, BoundingBox(-5.5, -5.4, -2.1, -2.6).xyxy_int)
        assert np.isnan(result).all()


@pytest.mark.required
class TestToCanonicalGrayscale:
    """Tests for to_canonical_grayscale function."""

    def test_single_channel(self):
        """Test conversion from 1-channel image."""
        image = np.random.randint(0, 256, (1, 28, 28)).astype(np.uint8)
        result = to_canonical_grayscale(image)
        assert result.shape == (28, 28)
        np.testing.assert_array_equal(result, image[0])

    def test_rgb_conversion(self):
        """Test conversion from 3-channel RGB image."""
        image = np.random.randint(0, 256, (3, 28, 28)).astype(np.uint8)
        result = to_canonical_grayscale(image)
        assert result.shape == (28, 28)
        assert result.dtype == np.uint8

    def test_rgba_to_grayscale(self):
        """Test conversion from 4-channel RGBA image (line 159-189)."""
        # Create RGBA image with mostly opaque pixels (high alpha)
        image = np.random.randint(0, 256, (4, 28, 28)).astype(np.uint8)
        image[3] = 255  # Make alpha channel fully opaque
        result = to_canonical_grayscale(image)
        assert result.shape == (28, 28)
        assert result.dtype == np.uint8

    def test_cmyk_to_grayscale(self):
        """Test conversion from 4-channel CMYK image (line 159-189)."""
        # Create CMYK-like image with high variance in K channel
        image = np.random.randint(0, 256, (4, 28, 28)).astype(np.uint8)
        # Make K channel (4th) have high variance and mid-range mean
        image[3] = np.random.randint(40, 215, (28, 28)).astype(np.uint8)
        result = to_canonical_grayscale(image)
        assert result.shape == (28, 28)
        assert result.dtype == np.uint8

    def test_arbitrary_channels(self):
        """Test conversion from image with arbitrary channels (line 191-193)."""
        # 5-channel image
        image = np.random.randint(0, 256, (5, 28, 28)).astype(np.uint8)
        result = to_canonical_grayscale(image)
        assert result.shape == (28, 28)
        assert result.dtype == np.uint8


@pytest.mark.required
class TestResize:
    """Tests for resize function."""

    def test_resize_with_pil(self):
        """Test resizing with PIL (line 123-124)."""
        image = np.random.randint(0, 256, (28, 28)).astype(np.uint8)
        result = resize(image, 64, use_pil=True)
        assert result.shape == (64, 64)
        assert result.dtype == np.uint8

    def test_resize_without_pil(self):
        """Test resizing without PIL using scipy (line 126-127)."""
        image = np.random.randint(0, 256, (28, 28)).astype(np.uint8)
        result = resize(image, 64, use_pil=False)
        assert result.shape == (64, 64)
        assert result.dtype == np.uint8
