"""Tests for the Resize view operation."""

import numpy as np
import pytest

from dataeval.data import ClassFilter, Resize, View
from dataeval.flags import ImageStats
from dataeval.protocols import DatasetMetadata


class _ODTarget:
    def __init__(self, boxes, labels) -> None:
        self.boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        self.labels = np.asarray(labels, dtype=np.intp)
        self.scores = np.ones(len(self.labels), dtype=np.float64)


class _ODDataset:
    """Object-detection dataset with caller-supplied image shape and per-image boxes."""

    def __init__(self, shape=(20, 40), boxes=None, labels=None, n: int = 2) -> None:
        self._shape = shape
        self._boxes = boxes if boxes is not None else [[4.0, 4.0, 8.0, 8.0]]
        self._labels = labels if labels is not None else [0] * len(self._boxes)
        self._n = n
        self.metadata = DatasetMetadata(id="toy", index2label={0: "a", 1: "b"})

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int):
        image = np.arange(3 * self._shape[0] * self._shape[1], dtype=np.float32).reshape(3, *self._shape)
        return image, _ODTarget(self._boxes, self._labels), {"id": index}


def _images(shape=(20, 40), n: int = 2):
    """Image-only dataset of the given (height, width)."""

    class _Images:
        metadata = {"id": "images"}

        def __len__(self) -> int:
            return n

        def __getitem__(self, index: int):
            return np.full((3, *shape), index + 1, dtype=np.float32)

    return _Images()


def _boxes_of(view, index: int = 0):
    return np.asarray(view[index][1].boxes).tolist()


@pytest.mark.required
class TestResizeValidation:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="mode"):
            Resize((10, 10), mode="squish")  # type: ignore[arg-type]

    def test_rejects_unknown_fill(self):
        with pytest.raises(ValueError, match="fill"):
            Resize((10, 10), mode="pad", fill="median")  # type: ignore[arg-type]

    def test_rejects_non_positive_size(self):
        with pytest.raises(ValueError, match="size"):
            Resize((0, 10))
        with pytest.raises(ValueError, match="size"):
            Resize(-4)

    def test_rejects_malformed_size(self):
        with pytest.raises(ValueError, match="size"):
            Resize((10, 10, 10))  # type: ignore[arg-type]

    def test_rejects_single_element_size_tuple(self):
        # Would otherwise construct and only fail on the `height, width = target` unpack.
        with pytest.raises(ValueError, match="size"):
            Resize((10,))  # type: ignore[arg-type]


@pytest.mark.required
class TestResizeOutputDimensions:
    @pytest.mark.parametrize("mode", ["stretch", "pad", "crop"])
    @pytest.mark.parametrize("shape", [(20, 40), (40, 20), (20, 20)])
    def test_exact_size_is_honored_for_every_mode_and_aspect(self, mode, shape):
        view = View(_images(shape), [Resize((10, 12), mode=mode)])
        assert np.asarray(view[0]).shape == (3, 10, 12)

    def test_int_size_scales_the_shortest_side_on_a_landscape_source(self):
        # 20x40 (hxw): shortest side is the height -> h=10, w = 40 * 10/20 = 20
        view = View(_images((20, 40)), [Resize(10)])
        assert np.asarray(view[0]).shape == (3, 10, 20)

    def test_int_size_scales_the_shortest_side_on_a_portrait_source(self):
        view = View(_images((40, 20)), [Resize(10)])
        assert np.asarray(view[0]).shape == (3, 20, 10)

    def test_int_size_on_a_square_source(self):
        view = View(_images((20, 20)), [Resize(10)])
        assert np.asarray(view[0]).shape == (3, 10, 10)


@pytest.mark.required
class TestResizeBoxRewrite:
    """Box coordinates are hand-worked from a 20x40 (hxw) source into a 10x10 frame."""

    def test_stretch_scales_each_axis_independently(self):
        # sx = 10/40 = 0.25, sy = 10/20 = 0.5
        view = View(_ODDataset((20, 40), [[4.0, 4.0, 8.0, 8.0]]), [Resize((10, 10), mode="stretch")])
        assert _boxes_of(view) == [[1.0, 2.0, 2.0, 4.0]]

    def test_pad_scales_uniformly_then_offsets_by_the_letterbox(self):
        # s = min(10/40, 10/20) = 0.25 -> content 5x10, so dy = (10-5)//2 = 2, dx = 0
        view = View(_ODDataset((20, 40), [[4.0, 4.0, 8.0, 8.0]]), [Resize((10, 10), mode="pad")])
        assert _boxes_of(view) == [[1.0, 3.0, 2.0, 4.0]]

    def test_crop_offsets_clips_and_drops(self):
        # s = max(10/40, 10/20) = 0.5 -> scaled 10x20, center crop takes x in [5, 15)
        # box A: x 2..4 -> -3..-1, fully left of the frame -> dropped
        # box B: x 12..16 -> 7..11, clipped to 10; y 2..4 unchanged
        boxes = [[4.0, 4.0, 8.0, 8.0], [24.0, 4.0, 32.0, 8.0]]
        view = View(_ODDataset((20, 40), boxes, [0, 1]), [Resize((10, 10), mode="crop")])
        assert _boxes_of(view) == [[7.0, 2.0, 10.0, 4.0]]
        assert np.asarray(view[0][1].labels).tolist() == [1]

    def test_per_detection_metadata_follows_a_dropped_box(self):
        class _Dataset(_ODDataset):
            def __getitem__(self, index: int):
                image, target, _ = super().__getitem__(index)
                return image, target, {"id": index, "track": [100, 200]}

        boxes = [[4.0, 4.0, 8.0, 8.0], [24.0, 4.0, 32.0, 8.0]]
        view = View(_Dataset((20, 40), boxes, [0, 1]), [Resize((10, 10), mode="crop")])
        assert view[0][2]["track"] == [200]

    def test_pad_clips_an_overhanging_box_to_the_content_not_the_bars(self):
        # 20x40 source into 20x20: s = 0.5, content is 10 rows centered at dy = 5, so the
        # real pixels are rows 5..15 and rows 0..5 / 15..20 are synthetic fill. A source
        # annotation running off the top and bottom of the frame must clip to the content.
        view = View(_ODDataset((20, 40), [[5.0, -3.0, 15.0, 25.0]]), [Resize((20, 20), mode="pad")])
        assert _boxes_of(view) == [[2.5, 5.0, 7.5, 15.0]]

    def test_image_classification_target_is_untouched(self):
        target = np.array([0.0, 1.0])

        class _ICDataset:
            metadata = {"id": "ic"}

            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int):
                return np.zeros((3, 20, 40), dtype=np.float32), target, {"id": index}

        view = View(_ICDataset(), [Resize((10, 10))])
        assert np.asarray(view[0][1]).tolist() == [0.0, 1.0]


@pytest.mark.required
class TestResizePadFill:
    def test_zero_fill_pads_with_zeros(self):
        # 20x40 source into 10x10: content is 5 rows tall, padded 2 rows top and 3 bottom.
        view = View(_images((20, 40)), [Resize((10, 10), mode="pad", fill="zero")])
        image = np.asarray(view[0])
        assert np.all(image[:, :2, :] == 0)
        assert np.all(image[:, 7:, :] == 0)
        assert not np.all(image[:, 2:7, :] == 0)

    def test_mean_fill_pads_with_the_per_channel_mean(self):
        view = View(_images((20, 40)), [Resize((10, 10), mode="pad", fill="mean")])
        image = np.asarray(view[0])
        content = image[:, 2:7, :]
        expected = content.reshape(3, -1).mean(axis=1)
        assert np.allclose(image[:, 0, 0], expected)
        assert np.allclose(image[:, 9, 9], expected)

    def test_stretch_ignores_fill_and_leaves_no_padding(self):
        view = View(_images((20, 40)), [Resize((10, 10), mode="stretch", fill="zero")])
        assert not np.any(np.asarray(view[0]) == 0)


@pytest.mark.required
class TestResizeInvalidates:
    BASE = (ImageStats.DIMENSION & ~ImageStats.DIMENSION_CHANNELS) | ImageStats.VISUAL_SHARPNESS
    PAD_EXTRA = (
        ImageStats.PIXEL_ZEROS
        | ImageStats.PIXEL_HISTOGRAM
        | ImageStats.PIXEL_ENTROPY
        | ImageStats.PIXEL_STD
        | ImageStats.PIXEL_VAR
        | ImageStats.PIXEL_SKEW
        | ImageStats.PIXEL_KURTOSIS
        | ImageStats.VISUAL_BRIGHTNESS
        | ImageStats.VISUAL_DARKNESS
        | ImageStats.VISUAL_CONTRAST
        | ImageStats.VISUAL_PERCENTILES
    )

    @pytest.mark.parametrize("mode", ["stretch", "crop"])
    def test_non_pad_modes_declare_the_base_set(self, mode):
        assert Resize((10, 10), mode=mode).invalidates == self.BASE

    def test_pad_adds_the_padding_sensitive_stats(self):
        assert Resize((10, 10), mode="pad").invalidates == self.BASE | self.PAD_EXTRA

    def test_zero_fill_pad_also_invalidates_the_mean(self):
        # Letterboxing a 2:1 source into a square canvas with zeros halves PIXEL_MEAN.
        assert Resize((10, 10), mode="pad", fill="zero").invalidates & ImageStats.PIXEL_MEAN

    def test_mean_fill_pad_preserves_the_mean(self):
        # fill="mean" pads with the content's own mean, so the mean is unchanged.
        assert not Resize((10, 10), mode="pad", fill="mean").invalidates & ImageStats.PIXEL_MEAN

    def test_fill_is_the_only_difference_between_the_two_pad_declarations(self):
        zero = Resize((10, 10), mode="pad", fill="zero").invalidates
        mean = Resize((10, 10), mode="pad", fill="mean").invalidates
        assert zero == mean | ImageStats.PIXEL_MEAN

    @pytest.mark.parametrize("mode", ["stretch", "pad", "crop"])
    def test_hash_is_never_invalidated(self, mode):
        # Resize-then-phash is a better near-duplicate check across heterogeneous
        # source resolutions, not a worse one.
        assert not Resize((10, 10), mode=mode).invalidates & ImageStats.HASH

    @pytest.mark.parametrize("mode", ["stretch", "pad", "crop"])
    def test_channel_count_is_never_invalidated(self, mode):
        assert not Resize((10, 10), mode=mode).invalidates & ImageStats.DIMENSION_CHANNELS

    @pytest.mark.parametrize("mode", ["stretch", "pad", "crop"])
    def test_bit_depth_is_invalidated(self, mode):
        # Bilinear resize promotes uint8 to float.
        assert Resize((10, 10), mode=mode).invalidates & ImageStats.DIMENSION_DEPTH

    def test_stretch_does_not_invalidate_pixel_stats(self):
        assert not Resize((10, 10), mode="stretch").invalidates & ImageStats.PIXEL

    def test_constructor_override_replaces_the_computed_value(self):
        op = Resize((10, 10), mode="pad", invalidates=ImageStats.HASH)
        assert op.invalidates is ImageStats.HASH

    def test_constructor_override_survives_into_repr(self):
        assert "invalidates=None" in repr(Resize((10, 10)))
        assert "HASH" in repr(Resize((10, 10), invalidates=ImageStats.HASH))

    def test_override_reaches_the_invalidation_walk(self):
        from dataeval.data._invalidates import invalidated_stats

        view = View(_images(), [Resize((10, 10), invalidates=ImageStats.HASH_PHASH)])
        assert invalidated_stats(view) is ImageStats.HASH_PHASH


@pytest.mark.required
class TestResizeComposition:
    def test_composes_with_classfilter_in_either_order(self):
        boxes = [[4.0, 4.0, 8.0, 8.0], [12.0, 4.0, 16.0, 8.0]]
        before = View(_ODDataset((20, 40), boxes, [0, 1], n=3), [Resize((10, 10)), ClassFilter(classes=[0])])
        after = View(_ODDataset((20, 40), boxes, [0, 1], n=3), [ClassFilter(classes=[0]), Resize((10, 10))])
        assert before.selection == after.selection
        assert _boxes_of(before) == _boxes_of(after)

    def test_repeated_resize_applies_in_order(self):
        view = View(_images((20, 40)), [Resize((10, 20)), Resize((5, 5))])
        assert np.asarray(view[0]).shape == (3, 5, 5)


@pytest.mark.required
class TestResizeDocstring:
    def test_carries_the_resolution_heterogeneity_caveat(self):
        doc = Resize.__doc__ or ""
        assert "resolution" in doc.lower()
        assert "aspect_ratio" in doc
        assert "Outliers" in doc
