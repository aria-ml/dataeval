"""Tests for the SelectChannels view operation."""

import numpy as np
import pytest

from dataeval.data import SelectChannels, View
from dataeval.data._invalidates import invalidated_stats
from dataeval.flags import ImageStats
from dataeval.protocols import DatasetMetadata

# Rec. 601 luma coefficients, matching utils.preprocessing.to_canonical_grayscale
LUMINANCE = (0.299, 0.587, 0.114)


class _ODTarget:
    def __init__(self, boxes, labels) -> None:
        self.boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        self.labels = np.asarray(labels, dtype=np.intp)
        self.scores = np.ones(len(self.labels), dtype=np.float64)


class _ODDataset:
    """Object-detection dataset whose channel c is filled with the constant c + 1."""

    def __init__(self, channels: int = 3, n: int = 2) -> None:
        self._channels = channels
        self._n = n
        self.metadata = DatasetMetadata(id="toy", index2label={0: "a"})

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int):
        image = np.stack([np.full((4, 5), c + 1, dtype=np.float32) for c in range(self._channels)])
        return image, _ODTarget([[1.0, 1.0, 3.0, 3.0]], [0]), {"id": index}


def _images(channels: int = 3, n: int = 2):
    class _Images:
        metadata = {"id": "images"}

        def __len__(self) -> int:
            return n

        def __getitem__(self, index: int):
            return np.stack([np.full((4, 5), c + 1, dtype=np.float32) for c in range(channels)])

    return _Images()


@pytest.mark.required
class TestSelectChannelsValidation:
    def test_rejects_unknown_keyword(self):
        with pytest.raises(ValueError, match="channels"):
            SelectChannels("hsv")  # type: ignore[arg-type]

    def test_rejects_empty_selection(self):
        with pytest.raises(ValueError, match="channels"):
            SelectChannels([])

    def test_rejects_negative_index(self):
        with pytest.raises(ValueError, match="channels"):
            SelectChannels([0, -1])

    def test_rejects_non_integer_index(self):
        with pytest.raises(ValueError, match="channels"):
            SelectChannels([0, 1.5])  # type: ignore[list-item]

    def test_rejects_boolean_index(self):
        # bool is an int subclass, but numpy reads a list of bools as a mask, so this
        # would silently select channels 0 and 2 rather than indices 1, 0, 1.
        with pytest.raises(ValueError, match="channels"):
            SelectChannels([True, False, True])  # type: ignore[list-item]

    def test_out_of_range_index_errors_clearly_at_read(self):
        view = View(_images(3), [SelectChannels([0, 7])])
        with pytest.raises(IndexError, match="channel"):
            _ = view[0]


@pytest.mark.required
class TestSelectChannelsIndices:
    def test_selection_returns_channels_in_the_requested_order(self):
        view = View(_images(3), [SelectChannels([2, 0])])
        image = np.asarray(view[0])
        assert image.shape == (2, 4, 5)
        assert image[0, 0, 0] == 3.0
        assert image[1, 0, 0] == 1.0

    def test_single_index_yields_one_channel(self):
        view = View(_images(3), [SelectChannels([1])])
        image = np.asarray(view[0])
        assert image.shape == (1, 4, 5)
        assert np.all(image == 2.0)

    def test_repeated_index_is_allowed(self):
        view = View(_images(3), [SelectChannels([0, 0, 0])])
        assert np.asarray(view[0]).shape == (3, 4, 5)

    def test_multispectral_source_narrows_to_the_bands_requested(self):
        view = View(_images(8), [SelectChannels([1, 3, 5])])
        image = np.asarray(view[0])
        assert image.shape == (3, 4, 5)
        assert [image[i, 0, 0] for i in range(3)] == [2.0, 4.0, 6.0]


@pytest.mark.required
class TestSelectChannelsGray:
    def test_gray_produces_a_single_channel_by_luminance(self):
        view = View(_images(3), [SelectChannels("gray")])
        image = np.asarray(view[0])
        assert image.shape == (1, 4, 5)
        expected = sum(w * (c + 1) for c, w in enumerate(LUMINANCE))
        assert np.allclose(image, expected)

    def test_gray_from_a_single_channel_source_is_a_passthrough(self):
        view = View(_images(1), [SelectChannels("gray")])
        image = np.asarray(view[0])
        assert image.shape == (1, 4, 5)
        assert np.allclose(image, 1.0)

    def test_gray_rejects_a_source_that_is_neither_one_nor_three_channel(self):
        view = View(_images(4), [SelectChannels("gray")])
        with pytest.raises(ValueError, match="gray"):
            _ = view[0]


@pytest.mark.required
class TestSelectChannelsRgb:
    def test_rgb_broadcasts_a_single_channel_source(self):
        view = View(_images(1), [SelectChannels("rgb")])
        image = np.asarray(view[0])
        assert image.shape == (3, 4, 5)
        assert np.all(image == 1.0)

    def test_rgb_leaves_a_three_channel_source_alone(self):
        view = View(_images(3), [SelectChannels("rgb")])
        image = np.asarray(view[0])
        assert image.shape == (3, 4, 5)
        assert [image[i, 0, 0] for i in range(3)] == [1.0, 2.0, 3.0]

    def test_rgb_rejects_a_source_that_is_neither_one_nor_three_channel(self):
        view = View(_images(4), [SelectChannels("rgb")])
        with pytest.raises(ValueError, match="rgb"):
            _ = view[0]


@pytest.mark.required
class TestSelectChannelsTargets:
    def test_targets_are_byte_identical(self):
        source = _ODDataset(3)
        view = View(source, [SelectChannels([0])])
        original = source[0][1]
        _, target, metadata = view[0]
        assert np.array_equal(np.asarray(target.boxes), np.asarray(original.boxes))
        assert np.array_equal(np.asarray(target.labels), np.asarray(original.labels))
        assert np.array_equal(np.asarray(target.scores), np.asarray(original.scores))
        assert metadata == {"id": 0}

    def test_target_object_is_passed_through_unwrapped(self):
        class _ICDataset:
            metadata = {"id": "ic"}
            target = np.array([1.0, 0.0])

            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int):
                return np.zeros((3, 4, 5), dtype=np.float32), self.target, {"id": index}

        source = _ICDataset()
        view = View(source, [SelectChannels("gray")])
        assert view[0][1] is source.target


@pytest.mark.required
class TestSelectChannelsInvalidates:
    def test_index_selection_invalidates_only_the_channel_count(self):
        assert SelectChannels([0, 1]).invalidates is ImageStats.DIMENSION_CHANNELS

    def test_rgb_invalidates_only_the_channel_count(self):
        assert SelectChannels("rgb").invalidates is ImageStats.DIMENSION_CHANNELS

    def test_gray_also_invalidates_pixel_and_visual_stats(self):
        # A luminance mix moves PIXEL_MEAN and every visual statistic.
        expected = ImageStats.DIMENSION_CHANNELS | ImageStats.PIXEL | ImageStats.VISUAL
        assert SelectChannels("gray").invalidates == expected

    def test_index_selection_does_not_invalidate_pixel_or_visual(self):
        invalidates = SelectChannels([0]).invalidates
        assert not invalidates & ImageStats.PIXEL
        assert not invalidates & ImageStats.VISUAL

    def test_geometry_stats_are_never_invalidated(self):
        for op in (SelectChannels([0]), SelectChannels("gray"), SelectChannels("rgb")):
            assert not op.invalidates & ImageStats.DIMENSION_WIDTH
            assert not op.invalidates & ImageStats.DIMENSION_HEIGHT

    def test_constructor_override_replaces_the_computed_value(self):
        assert SelectChannels("gray", invalidates=ImageStats.HASH).invalidates is ImageStats.HASH

    def test_constructor_override_survives_into_repr(self):
        assert "invalidates=None" in repr(SelectChannels([0]))
        assert "HASH" in repr(SelectChannels([0], invalidates=ImageStats.HASH))

    def test_override_reaches_the_invalidation_walk(self):
        view = View(_images(), [SelectChannels("gray", invalidates=ImageStats.HASH_PHASH)])
        assert invalidated_stats(view) is ImageStats.HASH_PHASH


@pytest.mark.required
class TestSharedIndexValidation:
    """SelectChannels and ChannelGroup share what an index is, not what a selection may be."""

    def test_selection_may_repeat_and_reorder(self):
        """A view transform duplicates and reorders bands deliberately."""
        assert SelectChannels([0, 0, 1]).channels == [0, 0, 1]
        assert SelectChannels([2, 1, 0]).channels == [2, 1, 0]

    def test_group_may_not_repeat(self):
        """A group is reduced over jointly, so a repeat would double-weight a band."""
        from dataeval.utils.preprocessing import ChannelGroup

        with pytest.raises(ValueError, match="must not repeat"):
            ChannelGroup([0, 0, 1])

    def test_both_reject_the_same_non_indices(self):
        """The shared half: bools are a mask to numpy, and negatives name nothing."""
        from dataeval.utils.preprocessing import ChannelGroup

        for bad in ([True, False, True], [-1], []):
            with pytest.raises(ValueError, match="non-negative ints"):
                SelectChannels(bad)
            with pytest.raises(ValueError, match="non-negative ints"):
                ChannelGroup(bad)
