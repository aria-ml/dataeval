"""Verify that a frame's provenance survives every view transform around the frame view.

``SequenceFrames`` reframes a multi-object-tracking dataset as an object-detection dataset
of frames, so a ``View`` can sit either side of it: above, renumbering and transforming the
frames it presents, and below, renumbering the videos it was built over. A statistic
measured over the outermost view comes back addressed by that view's numbering, and the
whole point of the reframing is that the address still names one frame of one video.

Maps to meta repo test cases:
  - TC-6.1: Dataset selection and filtering
"""

from typing import Any, cast

import numpy as np
import pytest

from dataeval.data import SequenceFrames, SourceLocator
from dataeval.types import SourceIndex


def _located(frames: SequenceFrames, locator: SourceLocator, address: SourceIndex | int) -> tuple[int, int]:
    """Resolve one address, measured over `locator`'s view, to a (sequence, frame) pair."""
    sequence, frame = frames.frame_map[locator[address].source_item_index].tolist()
    return sequence, frame


def _meta(frames: SequenceFrames) -> list[dict[str, Any]]:
    """Every selected frame's metadata, widened to a plain dict for assertion."""
    return [cast(dict[str, Any], meta) for _, _, meta in frames.stream()]


@pytest.mark.test_case("6-1")
class TestFrameViewProvenance:
    """Verify SequenceFrames provenance through View transforms, above it and below it."""

    def test_every_frame_records_the_video_it_came_from(self):
        from verification.helpers import make_tracking_dataset

        frames = SequenceFrames(make_tracking_dataset((4, 3)))
        recorded = [(meta["source_id"], meta["sequence"], meta["frame"]) for meta in _meta(frames)]
        assert recorded == [
            ("video-0", 0, 0),
            ("video-0", 0, 1),
            ("video-0", 0, 2),
            ("video-0", 0, 3),
            ("video-1", 1, 0),
            ("video-1", 1, 1),
            ("video-1", 1, 2),
        ]

    def test_statistics_over_the_frame_view_address_one_frame_each(self):
        from dataeval.core import compute_stats
        from dataeval.data import Stride
        from dataeval.flags import ImageStats
        from verification.helpers import make_tracking_dataset

        frames = SequenceFrames(make_tracking_dataset((6, 5)), Stride(2))
        result = compute_stats(frames, stats=ImageStats.DIMENSION_WIDTH, per_target=False)
        addresses = [index.item for index in result["source_index"]]
        assert addresses == list(range(len(frames)))
        assert frames.frame_map.tolist() == [[0, 0], [0, 2], [0, 4], [1, 0], [1, 2], [1, 4]]

    def test_a_reordering_view_above_the_frame_view_is_undone_by_the_locator(self):
        from dataeval.data import Reverse, View
        from verification.helpers import make_tracking_dataset

        frames = SequenceFrames(make_tracking_dataset((4, 3)))
        locator = SourceLocator(View(frames, Reverse()))
        assert _located(frames, locator, SourceIndex(0)) == (1, 2)
        assert _located(frames, locator, SourceIndex(6)) == (0, 0)

    def test_a_filtering_view_above_the_frame_view_is_undone_by_the_locator(self):
        from dataeval.data import Indices, View
        from verification.helpers import make_tracking_dataset

        frames = SequenceFrames(make_tracking_dataset((4, 3)))
        locator = SourceLocator(View(frames, Indices([5, 1])))
        assert _located(frames, locator, SourceIndex(0)) == (1, 1)
        assert _located(frames, locator, SourceIndex(1)) == (0, 1)

    def test_the_retrieved_frame_is_the_frame_the_address_names(self):
        from dataeval.data import Reverse, View
        from verification.helpers import make_tracking_dataset

        dataset = make_tracking_dataset((4, 3))
        frames = SequenceFrames(dataset)
        found = SourceLocator(View(frames, Reverse()))[SourceIndex(0)]
        np.testing.assert_array_equal(found.pixels, dataset.frame_pixels(1, 2))

    def test_a_transform_above_the_frame_view_is_visible_between_the_two_reads(self):
        from dataeval.data import Crop, View
        from verification.helpers import make_tracking_dataset

        dataset = make_tracking_dataset((4,), shape=(3, 16, 16))
        found = SourceLocator(View(SequenceFrames(dataset), Crop((0, 0, 8, 8))))[SourceIndex(1)]
        assert found.pixels.shape == (3, 8, 8)
        np.testing.assert_array_equal(found.source_pixels, dataset.frame_pixels(0, 1))

    def test_a_transform_above_the_frame_view_carries_track_ids_with_the_boxes(self):
        """A detection the crop puts out of frame takes its track id with it."""
        from dataeval.data import Crop, View
        from verification.helpers import make_tracking_dataset

        view = View(SequenceFrames(make_tracking_dataset((3,), shape=(3, 16, 16))), Crop((0, 0, 8, 8)))
        _, target, _ = view[0]
        assert np.asarray(target.boxes).shape[0] == 1
        assert np.asarray(cast(Any, target).track_ids).tolist() == [0]

    def test_a_filtered_source_below_the_frame_view_is_named_by_id(self):
        """`sequence` numbers what SequenceFrames was handed; `source_id` says which video it is."""
        from dataeval.data import Indices, View
        from verification.helpers import make_tracking_dataset

        frames = SequenceFrames(cast(Any, View(make_tracking_dataset((4, 3, 5)), Indices([2, 0]))))
        recorded = {(meta["sequence"], meta["source_id"]) for meta in _meta(frames)}
        assert recorded == {(0, "video-2"), (1, "video-0")}
        assert len(frames) == 9

    def test_an_invalidating_operation_below_the_frame_view_stays_visible_above_it(self):
        """The invalidation walk crosses the frame view rather than stopping at it.

        A tracking datum's image is a stream rather than a raster, so the shipped pixel
        operations cannot run below the frame view; what has to hold is that a declaration
        made down there still reaches a consumer reading the view on top.
        """
        from dataeval.data import Operation, Reverse, View
        from dataeval.data._invalidates import invalidated_stats
        from dataeval.flags import ImageStats
        from verification.helpers import make_tracking_dataset

        class Declares(Operation):
            @property
            def invalidates(self) -> ImageStats:
                return ImageStats.VISUAL_SHARPNESS

            def apply(self, view: View) -> None:
                pass

        below = View(make_tracking_dataset((3,)), Declares())
        assert invalidated_stats(View(SequenceFrames(cast(Any, below)), Reverse())) == ImageStats.VISUAL_SHARPNESS

    def test_an_outlier_frame_resolves_through_the_frame_view(self):
        """`Outliers` addresses a frame by its flat position in the frame view."""
        from dataeval.quality import Outliers
        from verification.helpers import make_tracking_dataset

        dataset = make_tracking_dataset((6, 5))
        # Plant one frame that no statistic can miss: saturated white among random noise.
        dataset.plant_frame(0, 3, np.full((3, 16, 16), 255, dtype=np.uint8))

        frames = SequenceFrames(dataset)
        flagged = cast(list[Any], list(Outliers().evaluate(frames).outliers))
        assert len(flagged) == 1

        locator = SourceLocator(frames)
        assert _located(frames, locator, flagged[0]) == (0, 3)
        np.testing.assert_array_equal(locator[flagged[0]].pixels, dataset.frame_pixels(0, 3))

    def test_a_duplicate_frame_is_reported_in_the_source_videos_coordinates(self):
        """`Duplicates` reads `frame_map` and reports (sequence, frame), not a view position.

        This is the second of two address spaces that meet at the frame view, and the pair
        is deliberately pinned side by side with the `Outliers` case above: the same object,
        measured by two evaluators, hands back addresses that resolve against two different
        datasets. A group member here is a bare tuple rather than a `SourceIndex`, so it
        cannot be handed to a locator as it stands -- it has to be spelled as an address
        against the *source* tracking dataset, where a frame is a keyed `unit` row.
        """
        from dataeval.quality import Duplicates
        from verification.helpers import make_tracking_dataset

        dataset = make_tracking_dataset((6, 5))
        # Plant one exact duplicate across two videos: frame 4 of video 1 is frame 2 of video 0.
        dataset.plant_frame(1, 4, dataset.frame_pixels(0, 2).copy())

        frames = SequenceFrames(dataset)
        groups = cast(list[list[Any]], Duplicates(hash_radius=6).evaluate(frames).items.exact)
        assert len(groups) == 1
        assert set(groups[0]) == {(0, 2), (1, 4)}

        # The frame view cannot resolve them: at its item level a key names no row.
        with pytest.raises(ValueError, match="carries a key at 'unit' level"):
            SourceLocator(frames)[SourceIndex(0, 2, "unit")]

        # The source tracking dataset can, which is where these coordinates were read.
        source = SourceLocator(frames.source)
        assert source.levels == ("sequence", "unit", "track", "instance")
        for sequence, frame in groups[0]:
            found = source[SourceIndex(int(sequence), int(frame), "unit")]
            assert found.level == "unit"
            np.testing.assert_array_equal(found.pixels, dataset.frame_pixels(int(sequence), int(frame)))
