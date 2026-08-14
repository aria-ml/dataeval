"""Tests for the Crop view operation."""

import numpy as np
import pytest

from dataeval.data import Crop, View
from dataeval.data._invalidates import invalidated_stats
from dataeval.flags import ImageStats
from dataeval.protocols import DatasetMetadata


class _ODTarget:
    def __init__(self, boxes, labels) -> None:
        self.boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        self.labels = np.asarray(labels, dtype=np.intp)
        self.scores = np.ones(len(self.labels), dtype=np.float64)


class _ODDataset:
    """Object-detection dataset whose pixels encode their (y, x) so crops are checkable."""

    def __init__(self, shape=(20, 30), boxes=None, labels=None, metadata=None, n: int = 2) -> None:
        self._shape = shape
        self._boxes = boxes if boxes is not None else [[4.0, 4.0, 8.0, 8.0]]
        self._labels = labels if labels is not None else [0] * len(self._boxes)
        self._metadata = metadata
        self._n = n
        self.metadata = DatasetMetadata(id="toy", index2label={0: "a", 1: "b"})

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int):
        height, width = self._shape
        ys, xs = np.mgrid[0:height, 0:width]
        plane = (ys * width + xs).astype(np.float32)
        image = np.broadcast_to(plane, (3, height, width)).copy()
        metadata = dict(self._metadata) if self._metadata else {}
        return image, _ODTarget(self._boxes, self._labels), {"id": index, **metadata}


def _images(shape=(20, 30), n: int = 2):
    class _Images:
        metadata = {"id": "images"}

        def __len__(self) -> int:
            return n

        def __getitem__(self, index: int):
            return np.full((3, *shape), index + 1, dtype=np.float32)

    return _Images()


@pytest.mark.required
class TestCropValidation:
    def test_rejects_malformed_region(self):
        with pytest.raises(ValueError, match="region"):
            Crop((1, 2, 3))  # type: ignore[arg-type]

    def test_rejects_negative_origin(self):
        with pytest.raises(ValueError, match="region"):
            Crop((-1, 0, 10, 10))

    def test_rejects_empty_region(self):
        with pytest.raises(ValueError, match="region"):
            Crop((5, 5, 5, 10))
        with pytest.raises(ValueError, match="region"):
            Crop((5, 5, 10, 4))


@pytest.mark.required
class TestCropImage:
    def test_output_dimensions_match_the_requested_region(self):
        view = View(_images((20, 30)), [Crop((2, 3, 12, 9))])
        assert np.asarray(view[0]).shape == (3, 6, 10)

    def test_pixels_come_from_the_requested_region(self):
        # Pixel value is y * width + x, so the crop's first pixel is (3 * 30) + 2 = 92.
        view = View(_ODDataset((20, 30)), [Crop((2, 3, 12, 9))])
        image = np.asarray(view[0][0])
        assert image[0, 0, 0] == 92.0
        assert image[0, -1, -1] == (8 * 30) + 11

    def test_region_beyond_the_image_raises_at_read(self):
        view = View(_images((20, 30)), [Crop((0, 0, 40, 10))])
        with pytest.raises(ValueError, match="region"):
            _ = view[0]


@pytest.mark.required
class TestCropBoxRewrite:
    def test_boxes_offset_by_the_region_origin(self):
        view = View(_ODDataset((20, 30), [[6.0, 5.0, 10.0, 9.0]]), [Crop((2, 3, 20, 18))])
        assert np.asarray(view[0][1].boxes).tolist() == [[4.0, 2.0, 8.0, 6.0]]

    def test_fully_outside_detection_is_dropped(self):
        boxes = [[0.0, 0.0, 3.0, 3.0], [8.0, 8.0, 12.0, 12.0]]
        view = View(_ODDataset((20, 30), boxes, [0, 1]), [Crop((5, 5, 20, 18))])
        assert np.asarray(view[0][1].boxes).tolist() == [[3.0, 3.0, 7.0, 7.0]]
        assert np.asarray(view[0][1].labels).tolist() == [1]

    def test_partially_outside_detection_is_clipped(self):
        view = View(_ODDataset((20, 30), [[3.0, 3.0, 9.0, 9.0]]), [Crop((5, 5, 20, 18))])
        assert np.asarray(view[0][1].boxes).tolist() == [[0.0, 0.0, 4.0, 4.0]]

    def test_detection_clipped_against_the_far_edge(self):
        view = View(_ODDataset((20, 30), [[6.0, 6.0, 25.0, 19.0]]), [Crop((5, 5, 15, 15))])
        # region is 10 wide, 10 tall; box maps to (1, 1, 20, 14) then clips to the canvas
        assert np.asarray(view[0][1].boxes).tolist() == [[1.0, 1.0, 10.0, 10.0]]

    def test_per_detection_metadata_stays_length_aligned(self):
        boxes = [[0.0, 0.0, 3.0, 3.0], [8.0, 8.0, 12.0, 12.0]]
        dataset = _ODDataset((20, 30), boxes, [0, 1], metadata={"track": [11, 22]})
        view = View(dataset, [Crop((5, 5, 20, 18))])
        assert view[0][2]["track"] == [22]

    def test_zero_detection_datum_does_not_raise(self):
        view = View(_ODDataset((20, 30), np.zeros((0, 4)), []), [Crop((5, 5, 15, 15))])
        assert len(np.asarray(view[0][1].boxes)) == 0

    def test_image_only_dataset_is_supported(self):
        view = View(_images((20, 30)), [Crop((0, 0, 10, 10))])
        assert np.asarray(view[0]).shape == (3, 10, 10)


@pytest.mark.required
class TestCropInvalidates:
    EXPECTED = ImageStats.DIMENSION & ~(ImageStats.DIMENSION_CHANNELS | ImageStats.DIMENSION_DEPTH)

    def test_declares_dimension_minus_channels_and_depth(self):
        assert Crop((0, 0, 10, 10)).invalidates == self.EXPECTED

    def test_channel_count_and_bit_depth_survive_a_crop(self):
        invalidates = Crop((0, 0, 10, 10)).invalidates
        assert not invalidates & ImageStats.DIMENSION_CHANNELS
        assert not invalidates & ImageStats.DIMENSION_DEPTH

    def test_pixel_and_visual_stats_are_not_invalidated(self):
        # Cropping out a HUD overlay *improves* these statistics — that is the point.
        invalidates = Crop((0, 0, 10, 10)).invalidates
        assert not invalidates & ImageStats.PIXEL
        assert not invalidates & ImageStats.VISUAL

    def test_hash_is_not_invalidated(self):
        assert not Crop((0, 0, 10, 10)).invalidates & ImageStats.HASH

    def test_constructor_override_replaces_the_computed_value(self):
        assert Crop((0, 0, 10, 10), invalidates=ImageStats.HASH).invalidates is ImageStats.HASH

    def test_constructor_override_survives_into_repr(self):
        assert "invalidates=None" in repr(Crop((0, 0, 10, 10)))
        assert "HASH" in repr(Crop((0, 0, 10, 10), invalidates=ImageStats.HASH))

    def test_override_reaches_the_invalidation_walk(self):
        view = View(_images(), [Crop((0, 0, 10, 10), invalidates=ImageStats.HASH_PHASH)])
        assert invalidated_stats(view) is ImageStats.HASH_PHASH
