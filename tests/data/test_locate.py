"""Retrieving the thing a SourceIndex names — the round trip back to the source.

An address is only useful if it can be followed. These tests are the other half of
``tests/quality/test_level_addressing.py``: that one pins what the evaluators *emit*, this
one pins that what they emit lands back on the data it was measured over.

Every fixture is built so a wrong answer looks wrong. Pixels are gradients rather than
zeros, boxes differ per detection, and the video's frame numbers deliberately disagree
with their positions in the stream.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np
import pytest
from numpy.typing import NDArray

from dataeval import Metadata
from dataeval.data import DetectionCrops, Indices, Resize, Reverse, SourceLocator, View
from dataeval.data._crops import CropPolicy
from dataeval.types import SourceIndex
from tests.embeddings.test_embeddings import MockDataset, ObjectDetectionTarget

_T = TypeVar("_T")


def present(value: _T | None) -> _T:
    """Narrow an accessor that is optional by level, asserting this one carries a value."""
    assert value is not None
    return value


def _image(item: int) -> NDArray[np.float32]:
    """A (3, 16, 16) image whose every pixel encodes the item and its own position."""
    ys, xs = np.mgrid[0:16, 0:16]
    return np.stack([np.full((16, 16), item), ys, xs]).astype(np.float32)


def _od_dataset(counts: Sequence[int] = (2, 1, 3)) -> MockDataset:
    """An object detection dataset whose boxes are all different sizes and places."""
    images, targets = [], []
    for item, count in enumerate(counts):
        images.append(_image(item))
        boxes = np.array([[j, j, j + 2 + j, j + 3 + j] for j in range(count)], dtype=np.float64)
        targets.append(ObjectDetectionTarget(boxes, np.arange(count) % 3, np.linspace(0.1, 0.9, count or 1)[:count]))
    return MockDataset(images, targets)


class _Frame:
    """One decoded frame. ``frame_index`` is the stream's own numbering, not a position."""

    def __init__(self, frame_index: int, item: int) -> None:
        self.frame_index = frame_index
        self.time_s = frame_index * 0.5
        self.pts = frame_index * 1000
        self.pixels = np.stack([np.full((8, 8), item), np.full((8, 8), frame_index), np.zeros((8, 8))]).astype(
            np.float32
        )


class _FrameTracks:
    """One frame's detections, each box distinguishable by its track id."""

    def __init__(self, track_ids: Sequence[int]) -> None:
        self.track_ids = np.asarray(track_ids, dtype=np.intp)
        count = len(track_ids)
        self.boxes = np.array([[t, t, t + 2, t + 2] for t in track_ids], dtype=np.float64)
        self.labels = np.asarray([abs(t) % 3 for t in track_ids])
        self.scores = np.full(count, 0.5)


@dataclass
class _MOTTarget:
    frame_tracks: list[Any]


@dataclass
class _Stream:
    """A video stream that is iterable only, as the MAITE protocol requires."""

    frames: list[_Frame] = field(default_factory=list)

    def __iter__(self):
        return iter(self.frames)


# Two videos, with a gap, an empty frame and an untracked detection.
#
#   item 0, frame 0: tracks 5, 9       item 1, frame 0: track 7
#   item 0, frame 1: (none)            item 1, frame 1: tracks 7, 3
#   item 0, frame 2: track 9, untracked
#
# Track ids ascend nowhere, and a detection's key is not its index within its frame, so a
# positional shortcut anywhere below the item fails.
_SHAPES = [[[5, 9], [], [9, -1]], [[7], [7, 3]]]


def _mot_dataset(shapes=_SHAPES, subscriptable: bool = False, offset: int = 0) -> MockDataset:
    """A tracking dataset over `shapes`.

    ``offset`` numbers the frames ``offset, offset + 2, ...`` instead of by position,
    producing the non-conforming stream :class:`TestAStreamThatNumbersItsOwnFrames`
    describes. MAITE numbers a yielded frame by its position, so the default conforms.
    """
    streams, targets = [], []
    for item, per_frame in enumerate(shapes):
        numbering = (lambda p: offset + 2 * p) if offset else (lambda p: p)
        frames = [_Frame(numbering(position), item) for position in range(len(per_frame))]
        streams.append(frames if subscriptable else _Stream(frames))
        targets.append(_MOTTarget([_FrameTracks(ids) for ids in per_frame]))
    return MockDataset(streams, targets)


@pytest.mark.required
class TestTheTaskDecidesWhichLevelsExist:
    """A locator reads its dataset's task, and refuses an address from another one."""

    def test_an_image_task_sits_at_unit_level(self):
        assert SourceLocator(_od_dataset()).item_level == "unit"

    def test_a_tracking_task_sits_at_sequence_level(self):
        assert SourceLocator(_mot_dataset()).item_level == "sequence"

    def test_an_image_task_has_two_levels(self):
        assert SourceLocator(_od_dataset()).levels == ("unit", "instance")

    def test_a_tracking_task_has_all_four(self):
        assert SourceLocator(_mot_dataset()).levels == ("sequence", "unit", "track", "instance")

    def test_a_track_address_against_images_is_refused(self):
        locator = SourceLocator(_od_dataset())
        with pytest.raises(ValueError, match="'track'-level data, but this dataset's levels"):
            locator[SourceIndex(0, 1, "track")]

    def test_the_refusal_lists_the_levels_it_does_have(self):
        locator = SourceLocator(_od_dataset())
        with pytest.raises(ValueError, match="'unit', 'instance'"):
            locator[SourceIndex(0, None, "sequence")]

    def test_an_item_it_does_not_have_is_refused(self):
        locator = SourceLocator(_od_dataset())
        with pytest.raises(IndexError, match="names item 9, but the dataset has 3 items"):
            locator[SourceIndex(9)]

    def test_an_empty_dataset_constructs(self):
        """Nothing can be retrieved either way; refusing would be refusing a len() check."""
        assert len(SourceLocator(MockDataset([], []))) == 0


@pytest.mark.required
class TestUnstatedLevelsResolveAgainstTheTask:
    """The producer's minimal spelling is the one that has to work without help."""

    @pytest.mark.parametrize(
        ("address", "expected"),
        [(SourceIndex(1), "unit"), (SourceIndex(1, 0), "instance")],
    )
    def test_an_image_address(self, address, expected):
        assert SourceLocator(_od_dataset())[address].level == expected

    @pytest.mark.parametrize(
        ("address", "expected"),
        [(SourceIndex(1), "sequence"), (SourceIndex(1, 0), "instance")],
    )
    def test_the_same_address_on_video(self, address, expected):
        """One tuple, two datasets, two levels — which is the whole point of resolve()."""
        assert SourceLocator(_mot_dataset())[address].level == expected

    def test_both_spellings_of_one_detection_retrieve_alike(self):
        locator = SourceLocator(_od_dataset())
        minimal, explicit = locator[SourceIndex(2, 1)], locator[SourceIndex(2, 1, "instance")]
        assert np.array_equal(present(minimal.box), present(explicit.box))
        assert minimal.label == explicit.label

    def test_both_spellings_of_one_item_retrieve_alike(self):
        locator = SourceLocator(_od_dataset())
        minimal, explicit = locator[SourceIndex(2)], locator[SourceIndex(2, None, "unit")]
        assert np.array_equal(present(minimal.pixels), present(explicit.pixels))

    def test_a_plain_int_is_an_item(self):
        """What Outliers keys on when nothing below the item was measured."""
        locator = SourceLocator(_od_dataset())
        assert locator[2].address == SourceIndex(2)
        assert locator[2].level == "unit"


@pytest.mark.required
class TestRetrievingFromAnImageDataset:
    def test_an_item_gives_its_image(self):
        found = SourceLocator(_od_dataset())[SourceIndex(1)]
        assert np.array_equal(found.pixels, _image(1))

    @pytest.mark.parametrize("accessor", ["box", "label", "score"])
    def test_an_item_names_no_detection(self, accessor):
        """Raising, not None: a null box propagating into a plot is a worse answer."""
        found = SourceLocator(_od_dataset())[SourceIndex(1)]
        with pytest.raises(TypeError, match="names no single detection"):
            getattr(found, accessor)

    def test_a_detection_gives_its_box(self):
        found = SourceLocator(_od_dataset())[SourceIndex(2, 2)]
        assert found.box.tolist() == [2.0, 2.0, 6.0, 7.0]

    def test_a_detection_gives_its_label_and_score(self):
        found = SourceLocator(_od_dataset())[SourceIndex(2, 1)]
        assert found.label == 1
        assert found.score == pytest.approx(0.5)

    def test_a_detection_gives_the_whole_image_it_sits_in(self):
        """Context, not the cut-out — the crop is what crop() is for."""
        found = SourceLocator(_od_dataset())[SourceIndex(2, 1)]
        assert np.array_equal(found.pixels, _image(2))

    def test_a_detection_that_is_not_there_is_refused(self):
        locator = SourceLocator(_od_dataset())
        with pytest.raises(IndexError, match="names detection 7, but there are 2 to name"):
            _ = locator[SourceIndex(0, 7)].box

    @pytest.mark.parametrize(
        ("accessor", "message"),
        [("stream", "images rather than video"), ("frame", "the item is the frame"), ("track", "no tracks")],
    )
    def test_an_image_task_has_no_stream_or_frame_or_track(self, accessor, message):
        found = SourceLocator(_od_dataset())[SourceIndex(0, 1)]
        with pytest.raises(TypeError, match=message):
            getattr(found, accessor)

    def test_the_datum_metadata_comes_back(self):
        found = SourceLocator(_od_dataset())[SourceIndex(1)]
        assert found.datum_metadata["id"] == 1


@pytest.mark.required
class TestCropsMatchDetectionCrops:
    """The crop a user eyeballs and the crop an embedding saw are the same pixels."""

    def test_the_object_region_is_the_box(self):
        found = SourceLocator(_od_dataset())[SourceIndex(2, 2)]
        crop = found.crop()
        # Box [2, 2, 6, 7]: four columns and five rows of the source image.
        assert crop.shape == (3, 5, 4)
        assert np.array_equal(crop[2], _image(2)[2][2:7, 2:6])

    def test_it_agrees_with_the_dataset_view_pixel_for_pixel(self):
        """Addressed positionally: a crop's ``source_id`` is its datum's id, not an index."""
        counts = (2, 1, 3)
        dataset = _od_dataset(counts)
        crops = DetectionCrops(dataset, square="off", min_size=0)
        addresses = [SourceIndex(item, target) for item, count in enumerate(counts) for target in range(count)]
        assert len(crops) == len(addresses)
        located = SourceLocator(dataset)
        for index, address in enumerate(addresses):
            pixels, _, meta = crops[index]
            assert meta.get("target", -1) == address.key
            assert np.array_equal(located[address].crop(), pixels)

    def test_a_policy_carries_through(self):
        dataset = _od_dataset()
        crops = DetectionCrops(dataset, region="context", padding=0.5, square="expand", min_size=0)
        # Crop 3 is item 2's first detection: items 0 and 1 contribute 2 and 1 crops.
        found = SourceLocator(dataset)[SourceIndex(2, 0)]
        assert np.array_equal(found.crop(region="context", padding=0.5, square="expand"), crops[3][0])

    def test_the_defaults_of_the_two_deliberately_differ(self):
        """`crop()` keeps the detection's own aspect ratio; DetectionCrops squares for a model."""
        dataset = _od_dataset()
        # Crops run in item then detection order, so item 2's last detection is crop 5.
        found = SourceLocator(dataset)[SourceIndex(2, 2)]
        squared = DetectionCrops(dataset, min_size=0)[5][0]
        assert found.crop().shape != squared.shape
        assert np.array_equal(found.crop(square="expand"), squared)

    def test_cropping_an_item_is_refused(self):
        locator = SourceLocator(_od_dataset())
        with pytest.raises(TypeError, match="names no box, so there is nothing to crop"):
            locator[SourceIndex(1)].crop()

    def test_an_impossible_policy_is_refused(self):
        locator = SourceLocator(_od_dataset())
        with pytest.raises(ValueError, match="region='surround' requires padding"):
            locator[SourceIndex(0, 0)].crop(region="surround")


@pytest.mark.required
class TestRetrievingFromAVideoDataset:
    """The levels an image dataset cannot reach, on a stream that is iterable only."""

    def test_a_sequence_gives_no_raster(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0)]
        assert found.level == "sequence"
        with pytest.raises(TypeError, match="names no single raster — it is a whole video"):
            _ = found.pixels

    def test_a_sequence_gives_its_stream(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0)]
        assert len(list(found.stream)) == 3

    def test_a_frame_is_found_by_its_number(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0, 2, "unit")]
        assert found.frame.frame_index == 2
        assert found.pixels[1][0][0] == 2

    def test_a_frame_number_that_does_not_exist_is_refused(self):
        locator = SourceLocator(_mot_dataset())
        with pytest.raises(IndexError, match="no frame numbered 9"):
            _ = locator[SourceIndex(0, 9, "unit")].frame

    def test_a_track_gives_its_observations(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0, 9, "track")]
        assert present(found.track).track_id == 9
        assert present(found.track).frame_indices.tolist() == [0, 2]

    def test_a_track_that_is_not_there_is_refused(self):
        locator = SourceLocator(_mot_dataset())
        with pytest.raises(IndexError, match="names track 3, but item 0 has tracks"):
            _ = locator[SourceIndex(0, 3, "track")].track

    def test_a_track_names_no_raster(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0, 9, "track")]
        with pytest.raises(TypeError, match="names no single raster — it is frames"):
            _ = found.pixels


@pytest.mark.required
class TestAVideoDetectionKnowsWhereItSits:
    """The diamond, walked from below: a detection carries its frame and its track."""

    def test_target_index_counts_across_the_whole_sequence(self):
        """Item 0 holds detections 0 and 1 in frame 0, then 2 and 3 in frame 2 — frame 1 is empty."""
        locator = SourceLocator(_mot_dataset())
        assert [present(locator[SourceIndex(0, k)].frame).frame_index for k in range(4)] == [0, 0, 2, 2]

    def test_a_detection_gives_the_box_its_track_id_encodes(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0, 1)]
        assert found.box.tolist() == [9.0, 9.0, 11.0, 11.0]

    def test_a_detection_gives_the_frame_it_was_seen_in(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0, 2)]
        assert found.frame.frame_index == 2
        assert np.array_equal(found.pixels, present(found.frame.pixels))

    def test_a_detection_gives_the_track_it_belongs_to(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0, 2)]
        assert present(found.track).track_id == 9

    def test_an_untracked_detection_belongs_to_no_track(self):
        """track_id -1 has no track row in Metadata either, for the same reason."""
        found = SourceLocator(_mot_dataset())[SourceIndex(0, 3)]
        assert found.box.tolist() == [-1.0, -1.0, 1.0, 1.0]
        assert found.track is None

    def test_a_detection_past_the_end_of_the_sequence_is_refused(self):
        locator = SourceLocator(_mot_dataset())
        with pytest.raises(IndexError, match="names detection 4, but item 0 holds 4"):
            _ = locator[SourceIndex(0, 4)].box

    def test_the_second_sequence_numbers_its_own_detections(self):
        """Keys restart per item; nothing about item 1 depends on item 0's counts."""
        locator = SourceLocator(_mot_dataset())
        assert [present(locator[SourceIndex(1, k)].track).track_id for k in range(3)] == [7, 7, 3]


@pytest.mark.required
class TestViewsAndCoordinateSpaces:
    """Give the locator what you measured; it reports the position underneath."""

    def test_an_address_is_read_in_the_view_it_was_measured_over(self):
        dataset = _od_dataset()
        view = View(dataset, Reverse())
        assert np.array_equal(SourceLocator(view)[SourceIndex(0)].pixels, _image(2))

    def test_the_source_position_undoes_the_view(self):
        view = View(_od_dataset(), Reverse())
        assert SourceLocator(view)[SourceIndex(0)].source_item_index == 2

    def test_a_plain_dataset_is_its_own_source(self):
        assert SourceLocator(_od_dataset())[SourceIndex(1)].source_item_index == 1

    def test_a_filtered_view_renumbers_and_the_locator_follows(self):
        view = View(_od_dataset(), Indices([2, 0]))
        located = SourceLocator(view)
        assert located[SourceIndex(0)].source_item_index == 2
        assert np.array_equal(present(located[SourceIndex(0)].pixels), _image(2))


@pytest.mark.required
class TestBatches:
    def test_gather_returns_one_item_per_address_in_order(self):
        locator = SourceLocator(_od_dataset())
        addresses = [SourceIndex(2, 1), SourceIndex(0), SourceIndex(1, 0)]
        assert [found.address for found in locator.gather(addresses)] == addresses

    def test_gather_accepts_the_mapping_an_evaluator_returns(self):
        locator = SourceLocator(_od_dataset())
        findings = {SourceIndex(0, 1): ["contrast"], SourceIndex(2, 0): ["brightness"]}
        assert [found.level for found in locator.gather(findings)] == ["instance", "instance"]

    def test_gather_keeps_duplicates(self):
        locator = SourceLocator(_od_dataset())
        assert len(locator.gather([SourceIndex(0), SourceIndex(0)])) == 2

    def test_gather_agrees_with_one_at_a_time(self):
        """Two locators, because one would answer the second read out of caches the first warmed."""
        addresses = [SourceIndex(1, 2), SourceIndex(0, 0), SourceIndex(1, 0)]
        gathered = SourceLocator(_mot_dataset()).gather(addresses)
        one_at_a_time = SourceLocator(_mot_dataset())
        assert [present(f.frame).frame_index for f in gathered] == [
            present(one_at_a_time[a].frame).frame_index for a in addresses
        ]


@pytest.mark.required
class TestTheHandleItself:
    def test_two_retrievals_of_one_address_are_equal(self):
        """The locator is excluded from equality, so findings can be compared."""
        locator = SourceLocator(_od_dataset())
        assert locator[SourceIndex(1, 0)] == locator[SourceIndex(1, 0)]

    def test_the_repr_does_not_print_the_dataset(self):
        found = SourceLocator(_od_dataset())[SourceIndex(1, 0)]
        assert "MockDataset" not in repr(found)
        assert "SourceIndex(1, 0)" in repr(found)

    def test_the_datum_is_the_escape_hatch(self):
        found = SourceLocator(_od_dataset())[SourceIndex(1, 0)]
        image, target, _ = found.datum
        assert np.array_equal(image, _image(1))
        assert len(target.boxes) == 1

    def test_an_address_keyed_where_it_must_not_be_says_so(self):
        """A track address with no key resolves to track level and cannot name a row."""
        locator = SourceLocator(_mot_dataset())
        with pytest.raises(ValueError, match="whose rows are named by a key, but it states none"):
            _ = locator[SourceIndex(0, None, "track")].track


@pytest.mark.required
class TestAStreamThatNumbersItsOwnFrames:
    """A stream whose frame numbers are not its positions, which MAITE says cannot happen.

    MAITE numbers a yielded frame by its position in the stream, so for conforming data a
    frame's number, its position, and the ``frame_indices`` a :class:`~dataeval.types.Track`
    reports are one and the same. DataEval does not rely on that anywhere — the structuring
    walk takes min/max over frame numbers rather than trusting their order — and neither
    does this, so these pin what happens when a stream disagrees.
    """

    def test_a_frame_is_found_by_its_number_not_its_position(self):
        """Frame 2 is the second frame here; subscripting the stream would give frame 4."""
        found = SourceLocator(_mot_dataset(offset=10))[SourceIndex(0, 12, "unit")]
        assert found.frame.frame_index == 12
        assert found.pixels[1][0][0] == 12

    def test_a_subscriptable_stream_is_addressed_by_number_too(self):
        """Being indexable is no licence to treat the key as a position."""
        found = SourceLocator(_mot_dataset(subscriptable=True, offset=10))[SourceIndex(0, 14, "unit")]
        assert found.frame.frame_index == 14

    def test_a_track_reports_positions_where_a_unit_address_uses_numbers(self):
        """The one place the two coordinate systems are visibly different.

        :func:`~dataeval.data.build_tracks` is handed the target alone and never sees the
        stream, so it can only number a track's observations by their position in
        ``frame_tracks``. :class:`~dataeval.Metadata` records the stream's own number as
        ``unit_index``, which is what a ``unit``-level address keys on. They agree for
        every conforming stream and part company here.
        """
        locator = SourceLocator(_mot_dataset(offset=10))
        track = locator[SourceIndex(0, 9, "track")].track
        assert present(track).frame_indices.tolist() == [0, 2]

        # Followed as frame numbers they reach frames the stream does not have.
        with pytest.raises(IndexError, match="no frame numbered 0"):
            _ = locator[SourceIndex(0, 0, "unit")].frame

        # The detection itself is unaffected: it carries the frame it was seen in.
        assert present(locator[SourceIndex(0, 2)].frame).frame_index == 14


@pytest.mark.required
class TestTheUntransformedSource:
    """What was measured, and what sits underneath it, are both reachable."""

    def test_a_plain_dataset_reads_the_same_pixels_either_way(self):
        found = SourceLocator(_od_dataset())[SourceIndex(1)]
        assert np.array_equal(found.source_pixels, found.pixels)

    def test_a_transform_shows_up_between_the_two(self):
        view = View(_od_dataset(), Resize(8))
        found = SourceLocator(view)[SourceIndex(1)]
        assert found.pixels.shape == (3, 8, 8)
        assert found.source_pixels.shape == (3, 16, 16)
        assert np.array_equal(found.source_pixels, _image(1))

    def test_the_source_target_comes_back_too(self):
        """The whole tuple, not just its pixels — a resize rewrites boxes as well."""
        view = View(_od_dataset(), Resize(8))
        found = SourceLocator(view)[SourceIndex(2, 2)]
        _, source_target, _ = found.source_datum
        assert source_target.boxes[2].tolist() == [2.0, 2.0, 6.0, 7.0]
        assert found.box.tolist() != source_target.boxes[2].tolist()

    def test_selection_and_transform_together(self):
        view = View(_od_dataset(), [Indices([2, 0]), Resize(8)])
        found = SourceLocator(view)[SourceIndex(0)]
        assert found.source_item_index == 2
        assert np.array_equal(found.source_pixels, _image(2))

    def test_a_chain_of_views_is_walked_all_the_way_down(self):
        """resolve_indices steps one link; "no transforms applied" means the bottom."""
        view = View(View(_od_dataset(), Reverse()), Indices([0, 1]))
        found = SourceLocator(view)[SourceIndex(0)]
        assert found.source_item_index == 2
        assert np.array_equal(found.source_pixels, _image(2))

    def test_an_explicit_source_overrides_the_root(self):
        """The replacement's pixels are distinguishable from the root's, or this proves nothing."""
        replacement = _od_dataset()
        replacement.data = [image + 100 for image in replacement.data]
        view = View(_od_dataset(), Resize(8))
        found = SourceLocator(view, source=replacement)[SourceIndex(1)]
        assert np.array_equal(found.source_pixels, _image(1) + 100)
        assert not np.array_equal(found.source_pixels, _image(1))

    def test_the_source_is_reported(self):
        dataset = _od_dataset()
        assert SourceLocator(View(dataset, Reverse())).source is dataset

    def test_a_video_has_no_source_raster(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0)]
        with pytest.raises(TypeError, match="a stream rather than a raster"):
            _ = found.source_pixels
        assert len(list(found.source_datum[0])) == 3


@pytest.mark.required
class TestClimbingToTheContainingRow:
    """``at()`` — from a finding to the thing it sits in."""

    def test_a_detection_climbs_to_its_item_on_an_image_task(self):
        found = SourceLocator(_od_dataset())[SourceIndex(2, 1)]
        assert found.at("unit").address == SourceIndex(2)

    def test_a_detection_climbs_to_its_frame(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0, 2)]
        assert found.at("unit").address == SourceIndex(0, 2, "unit")

    def test_a_detection_climbs_to_its_track(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0, 2)]
        assert found.at("track").address == SourceIndex(0, 9, "track")

    def test_a_detection_climbs_to_its_sequence(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0, 2)]
        assert found.at("sequence").address == SourceIndex(0)

    def test_the_climb_lands_on_a_usable_item(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0, 2)]
        assert present(found.at("track").track).track_id == 9
        assert found.at("unit").frame.frame_index == 2

    def test_its_own_level_is_itself(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0, 2)]
        assert found.at("instance") == found

    def test_a_frame_climbs_to_its_sequence(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0, 2, "unit")]
        assert found.at("sequence").address == SourceIndex(0)

    def test_a_sibling_is_refused_with_the_route(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0, 2, "unit")]
        with pytest.raises(ValueError, match="siblings.*Reach one from the other through 'instance'"):
            found.at("track")

    def test_descending_through_at_is_refused(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0)]
        with pytest.raises(ValueError, match="'instance' does not sit above 'sequence'"):
            found.at("instance")

    def test_an_untracked_detection_has_no_track_to_climb_to(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0, 3)]
        with pytest.raises(ValueError, match="no tracker linked"):
            found.at("track")

    def test_a_level_the_dataset_lacks_is_refused(self):
        found = SourceLocator(_od_dataset())[SourceIndex(2, 1)]
        with pytest.raises(ValueError, match="'track' is not a level of this dataset"):
            found.at("track")


@pytest.mark.required
class TestDescendingToTheRowsInside:
    """``within()`` — the fan-out one address cannot express."""

    def test_an_image_item_holds_its_detections(self):
        found = SourceLocator(_od_dataset())[SourceIndex(2)]
        assert [inside.address for inside in found.within("instance")] == [
            SourceIndex(2, 0),
            SourceIndex(2, 1),
            SourceIndex(2, 2),
        ]

    def test_the_detections_are_usable(self):
        found = SourceLocator(_od_dataset())[SourceIndex(2)]
        assert [inside.label for inside in found.within("instance")] == [0, 1, 2]

    def test_an_item_with_no_detections_holds_nothing(self):
        found = SourceLocator(_od_dataset((0, 1)))[SourceIndex(0)]
        assert found.within("instance") == []

    def test_a_sequence_holds_its_frames(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0)]
        assert [inside.address for inside in found.within("unit")] == [
            SourceIndex(0, 0, "unit"),
            SourceIndex(0, 1, "unit"),
            SourceIndex(0, 2, "unit"),
        ]

    def test_a_sequence_holds_its_tracks(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0)]
        assert [inside.address for inside in found.within("track")] == [
            SourceIndex(0, 5, "track"),
            SourceIndex(0, 9, "track"),
        ]

    def test_a_sequence_holds_every_detection_including_untracked_ones(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0)]
        assert len(found.within("instance")) == 4

    def test_a_frame_holds_the_detections_seen_in_it(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0, 0, "unit")]
        assert [inside.address for inside in found.within("instance")] == [
            SourceIndex(0, 0),
            SourceIndex(0, 1),
        ]

    def test_an_empty_frame_holds_nothing(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0, 1, "unit")]
        assert found.within("instance") == []

    def test_a_track_holds_its_observations(self):
        """Track 9 is seen in frames 0 and 2, with a gap — detections 1 and 2."""
        found = SourceLocator(_mot_dataset())[SourceIndex(0, 9, "track")]
        assert [inside.address for inside in found.within("instance")] == [
            SourceIndex(0, 1),
            SourceIndex(0, 2),
        ]

    def test_a_track_reaches_the_frames_it_appears_in(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0, 9, "track")]
        frames = [inside.at("unit").address for inside in found.within("instance")]
        assert frames == [SourceIndex(0, 0, "unit"), SourceIndex(0, 2, "unit")]

    def test_a_track_never_leaves_its_item(self):
        """Track 7 exists in item 1 only; item 0's tracks are 5 and 9."""
        locator = SourceLocator(_mot_dataset())
        assert {inside.item_index for inside in locator[SourceIndex(1, 7, "track")].within("instance")} == {1}

    def test_climbing_back_out_returns_where_it_started(self):
        locator = SourceLocator(_mot_dataset())
        track = locator[SourceIndex(0, 9, "track")]
        assert [inside.at("track") for inside in track.within("instance")] == [track, track]

    def test_a_sibling_is_refused_with_the_route(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0, 9, "track")]
        with pytest.raises(ValueError, match="siblings.*intersect"):
            found.within("unit")

    def test_its_own_level_is_not_inside_itself(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0)]
        with pytest.raises(ValueError, match="'sequence' does not sit below 'sequence'"):
            found.within("sequence")

    def test_climbing_where_there_is_nothing_below_is_refused(self):
        found = SourceLocator(_od_dataset())[SourceIndex(0, 0)]
        with pytest.raises(ValueError, match="does not sit below 'instance'"):
            found.within("unit")


@pytest.mark.required
class TestAStreamOfBareFrames:
    """A stream whose frames declare no ``frame_index``, which dispatch permits.

    MAITE declares ``frame_index`` on a frame, but DataEval duck-types the *target* to
    decide a dataset's task and never requires the whole frame protocol — a stream of bare
    arrays reaches :class:`~dataeval.Metadata`, which numbers such frames by decode order.
    A locator has to agree, or a ``unit`` address built against that metadata names a
    different frame than it resolves to here.
    """

    @staticmethod
    def _bare() -> MockDataset:
        streams, targets = [], []
        for item, per_frame in enumerate(_SHAPES):
            streams.append([
                np.full((3, 8, 8), item + position, dtype=np.float32) for position in range(len(per_frame))
            ])
            targets.append(_MOTTarget([_FrameTracks(ids) for ids in per_frame]))
        return MockDataset(streams, targets)

    def test_frames_are_numbered_by_decode_order(self):
        found = SourceLocator(self._bare())[SourceIndex(0)]
        assert [inside.address for inside in found.within("unit")] == [
            SourceIndex(0, 0, "unit"),
            SourceIndex(0, 1, "unit"),
            SourceIndex(0, 2, "unit"),
        ]

    def test_a_frame_address_resolves(self):
        found = SourceLocator(self._bare())[SourceIndex(0, 2, "unit")]
        assert found.pixels[0][0][0] == 2

    def test_a_detection_climbs_to_the_frame_it_was_seen_in(self):
        found = SourceLocator(self._bare())[SourceIndex(0, 2)]
        assert found.at("unit").address == SourceIndex(0, 2, "unit")

    def test_it_agrees_with_what_metadata_records(self):
        """The contract: a unit address from Metadata names the frame the locator returns."""
        dataset = self._bare()
        metadata = Metadata(dataset)
        keys = metadata.rows_at("unit").select("item_index", "unit_index").rows()
        located = SourceLocator(dataset)
        for item_index, unit_index in keys:
            assert located[SourceIndex(item_index, unit_index, "unit")].level == "unit"
        assert [k for k in keys if k[0] == 0] == [(0, 0), (0, 1), (0, 2)]


@pytest.mark.required
class TestTheKeyIsCarried:
    """``key`` sits alongside ``item_index``: both halves of the address, unpacked."""

    def test_an_item_has_no_key(self):
        assert SourceLocator(_od_dataset())[SourceIndex(1)].key is None

    def test_a_detection_carries_its_target_index(self):
        assert SourceLocator(_od_dataset())[SourceIndex(1, 0)].key == 0

    def test_a_frame_carries_its_number(self):
        assert SourceLocator(_mot_dataset())[SourceIndex(0, 2, "unit")].key == 2

    def test_a_track_carries_its_id(self):
        assert SourceLocator(_mot_dataset())[SourceIndex(0, 9, "track")].key == 9

    def test_it_agrees_with_the_address(self):
        found = SourceLocator(_mot_dataset())[SourceIndex(0, 2)]
        assert (found.item_index, found.key) == (found.address.item, found.address.key)


@pytest.mark.required
class TestAStreamThatCannotBeRewound:
    """A stream that is an iterator rather than a re-iterable iterable.

    MAITE asks a stream for nothing but ``__iter__``, and a lazily decoded video is
    exactly a one-shot iterator: ``iter()`` hands back the same, already-advanced object.
    Walking such a stream twice answers the second walk with different frames, so it is
    walked once and the frames it passed are held alongside the datum they came from.
    """

    @staticmethod
    def _one_shot() -> MockDataset:
        """A tracking dataset whose stream can be iterated exactly once."""
        streams = [
            iter([_Frame(position, item) for position in range(len(per_frame))])
            for item, per_frame in enumerate(_SHAPES)
        ]
        targets = [_MOTTarget([_FrameTracks(ids) for ids in per_frame]) for per_frame in _SHAPES]
        return MockDataset(streams, targets)

    def test_every_detection_reaches_the_frame_it_was_seen_in(self):
        """Reading them in order used to resume the iterator and answer with later frames."""
        locator = SourceLocator(self._one_shot())
        assert [present(locator[SourceIndex(0, k)].frame).frame_index for k in range(4)] == [0, 0, 2, 2]

    def test_a_frame_address_still_resolves_after_the_numbers_were_read(self):
        """Numbering the stream consumes it, so the frames have to be kept while it is."""
        locator = SourceLocator(self._one_shot())
        assert [inside.key for inside in locator[SourceIndex(0)].within("unit")] == [0, 1, 2]
        assert present(locator[SourceIndex(0, 2, "unit")].frame).frame_index == 2

    def test_reading_one_frame_twice_gives_the_same_frame(self):
        locator = SourceLocator(self._one_shot())
        found = locator[SourceIndex(0, 2)]
        assert present(found.frame).frame_index == present(found.frame).frame_index == 2


@pytest.mark.required
class TestTheStreamIsWalkedOnce:
    """Every frame of one item is read at most once, however many addresses ask for it."""

    class _CountingStream:
        walks = 0

        def __init__(self, frames) -> None:
            self.frames = frames

        def __iter__(self):
            type(self).walks += 1
            return iter(self.frames)

    def _dataset(self, frames: int = 12) -> MockDataset:
        shapes = [[t] for t in range(frames)]
        stream = self._CountingStream([_Frame(position, 0) for position in range(frames)])
        return MockDataset([stream], [_MOTTarget([_FrameTracks(ids) for ids in shapes])])

    def test_many_detections_walk_the_stream_once(self):
        """Restarting the walk per detection made this quadratic in the frame count."""
        dataset = self._dataset()
        self._CountingStream.walks = 0
        locator = SourceLocator(dataset)
        for key in range(12):
            _ = locator[SourceIndex(0, key)].frame
        assert self._CountingStream.walks == 1

    def test_one_unit_address_walks_the_stream_once(self):
        """Numbering the frames and then finding one used to be two separate walks."""
        dataset = self._dataset()
        self._CountingStream.walks = 0
        _ = SourceLocator(dataset)[SourceIndex(0, 11, "unit")].pixels
        assert self._CountingStream.walks == 1


@pytest.mark.required
class TestFramesAreFoundByNumberNotByCacheState:
    """One address answers the same however the locator was warmed up first."""

    @staticmethod
    def _negatively_numbered() -> MockDataset:
        """A stream whose frame numbers are negative and disagree with their positions."""
        frames = [_Frame(number, 0) for number in (-3, -1, -2)]
        target = _MOTTarget([_FrameTracks([1]), _FrameTracks([2]), _FrameTracks([3])])
        return MockDataset([frames], [target])

    def test_a_negative_frame_number_is_not_a_negated_position(self):
        locator = SourceLocator(self._negatively_numbered())
        assert present(locator[SourceIndex(0, 0)].frame).frame_index == -3
        assert present(locator[SourceIndex(0, -1, "unit")].frame).frame_index == -1

    def test_the_answer_does_not_depend_on_what_was_read_before(self):
        cold = SourceLocator(self._negatively_numbered())
        warm = SourceLocator(self._negatively_numbered())
        _ = warm[SourceIndex(0, 0)].frame
        assert (
            present(cold[SourceIndex(0, -1, "unit")].frame).frame_index
            == present(warm[SourceIndex(0, -1, "unit")].frame).frame_index
        )


@pytest.mark.required
class TestAFrameThatHoldsNoDetections:
    """A detection-free frame target is allowed to omit ``track_ids``, as Metadata allows."""

    class _EmptyFrameTracks:
        """A frame with nothing in it, carrying no track ids at all."""

        boxes = np.zeros((0, 4), dtype=np.float64)
        labels = np.zeros(0, dtype=np.intp)
        scores = np.zeros(0, dtype=np.float64)

    def _dataset(self) -> MockDataset:
        target = _MOTTarget([_FrameTracks([5]), self._EmptyFrameTracks(), _FrameTracks([9])])
        return MockDataset([[_Frame(p, 0) for p in range(3)]], [target])

    def test_the_detections_around_it_are_still_reachable(self):
        locator = SourceLocator(self._dataset())
        assert [present(locator[SourceIndex(0, k)].track).track_id for k in range(2)] == [5, 9]

    def test_it_holds_nothing(self):
        found = SourceLocator(self._dataset())[SourceIndex(0, 1, "unit")]
        assert found.within("instance") == []

    def test_the_numbering_matches_what_metadata_records(self):
        dataset = self._dataset()
        keys = Metadata(dataset).rows_at("instance").select("item_index", "target_index").rows()
        assert keys == [(0, 0), (0, 1)]
        located = SourceLocator(dataset)
        assert [located[SourceIndex(*key)].box.tolist() for key in keys] == [
            [5.0, 5.0, 7.0, 7.0],
            [9.0, 9.0, 11.0, 11.0],
        ]


@pytest.mark.required
class TestATaskWithNoBoxes:
    """A classification dataset has instance rows, but no detection to retrieve for one."""

    @staticmethod
    def _dataset() -> MockDataset:
        labels = [np.eye(3, dtype=np.float32)[item] for item in range(3)]
        return MockDataset([_image(item) for item in range(3)], labels)

    def test_its_images_are_reachable(self):
        assert np.array_equal(SourceLocator(self._dataset())[SourceIndex(1)].pixels, _image(1))

    @pytest.mark.parametrize("accessor", ["box", "label", "score"])
    def test_a_detection_accessor_says_why_there_is_none(self, accessor):
        """A message, not an AttributeError leaking out of a missing `boxes` attribute."""
        found = SourceLocator(self._dataset())[SourceIndex(1, 0)]
        with pytest.raises(TypeError, match="targets carry no boxes"):
            getattr(found, accessor)

    def test_descending_to_detections_says_the_same(self):
        found = SourceLocator(self._dataset())[SourceIndex(1)]
        with pytest.raises(TypeError, match="targets carry no boxes"):
            found.within("instance")


@pytest.mark.required
class TestTheHandleIsAboutOneDataset:
    """Two items are equal when they name one row *of one dataset*."""

    def test_one_address_over_two_datasets_gives_two_items(self):
        """A findings batch and the same batch read through a view must not collapse."""
        dataset = _od_dataset()
        measured = SourceLocator(dataset)[SourceIndex(1, 0)]
        transformed = SourceLocator(View(dataset, Resize(8)))[SourceIndex(1, 0)]
        assert measured != transformed
        assert len({measured, transformed}) == 2

    def test_two_retrievals_from_one_locator_are_still_equal(self):
        locator = SourceLocator(_od_dataset())
        assert len({locator[SourceIndex(1, 0)], locator[SourceIndex(1, 0)]}) == 1


@pytest.mark.required
class TestIndexingWithWhatNumpyHandsBack:
    """An index read out of an array is a NumPy integer, and names an item all the same."""

    def test_a_numpy_integer_is_an_item(self):
        locator = SourceLocator(_od_dataset())
        assert locator[np.int64(2)].address == SourceIndex(2)

    def test_gather_takes_them_too(self):
        locator = SourceLocator(_od_dataset())
        assert [found.item_index for found in locator.gather(np.array([2, 0]))] == [2, 0]


@pytest.mark.required
class TestAnUnusableSourceSaysSo:
    """The position a view resolves to has to be a row of `source`."""

    def test_a_source_numbered_unlike_the_chain_is_refused(self):
        """The failure names the locator rather than leaking the container's own error."""
        dataset = _od_dataset()
        view = View(dataset, Indices([2, 1]))
        found = SourceLocator(view, source=_od_dataset((1,)))[SourceIndex(0)]
        with pytest.raises(IndexError, match="numbered like the bottom of this locator's view chain"):
            _ = found.source_datum

    def test_a_position_the_view_does_not_have_is_refused(self):
        locator = SourceLocator(View(_od_dataset(), Indices([2, 0])))
        with pytest.raises(IndexError, match="does not resolve to a row of the source"):
            locator.source_item_index(-1)


@pytest.mark.required
class TestTheCropPolicyIsHandedOverRatherThanRestated:
    """The extraction's promise, made reachable: one policy, both callers."""

    def test_a_views_policy_reproduces_its_crops(self):
        dataset = _od_dataset()
        crops = DetectionCrops(dataset, region="context", padding=0.5, square="expand", min_size=0)
        located = SourceLocator(dataset)
        for index in range(len(crops)):
            pixels, _, meta = crops[index]
            found = located[SourceIndex(int(meta.get("source_id", -1)), int(meta.get("target", -1)))]
            assert np.array_equal(found.crop(policy=crops.policy), pixels)

    def test_the_two_defaults_differ_and_the_policy_is_how_you_bridge_them(self):
        """Documented divergence: a crop to look at keeps its aspect ratio, one to embed squares."""
        dataset = _od_dataset()
        crops = DetectionCrops(dataset, min_size=0)
        index = next(
            i for i in range(len(crops)) if (crops[i][2].get("source_id", -1), crops[i][2].get("target", -1)) == (2, 2)
        )
        found = SourceLocator(dataset)[SourceIndex(2, 2)]
        assert found.crop().shape != crops[index][0].shape
        assert np.array_equal(found.crop(policy=crops.policy), crops[index][0])

    def test_a_policy_and_loose_parameters_together_are_refused(self):
        found = SourceLocator(_od_dataset())[SourceIndex(0, 0)]
        with pytest.raises(TypeError, match="pass policy= or the individual crop parameters, not both"):
            found.crop(policy=CropPolicy(), square="pad")

    def test_a_policy_refuses_to_be_built_from_a_combination_that_masks_everything(self):
        """`region="surround"` with no padding masks the whole crop; the view already refused it."""
        with pytest.raises(ValueError, match="region='surround' requires padding"):
            CropPolicy(region="surround")

    def test_a_box_outside_its_raster_is_named_rather_than_returned_empty(self):
        dataset = _od_dataset()
        _, target, _ = dataset[0]
        target.boxes[0] = [100.0, 100.0, 110.0, 110.0]
        found = SourceLocator(dataset)[SourceIndex(0, 0)]
        with pytest.raises(ValueError, match="lies outside its 16x16 raster"):
            found.crop()


@pytest.mark.required
class TestRastersAreNotWritable:
    """A finding is looked at, not edited: the raster handed back is the dataset's own."""

    def test_pixels_cannot_be_written_through(self):
        found = SourceLocator(_od_dataset())[SourceIndex(1)]
        with pytest.raises(ValueError, match="read-only"):
            found.pixels[0, 0, 0] = -999

    def test_source_pixels_cannot_be_written_through(self):
        view = View(_od_dataset(), Resize(8))
        found = SourceLocator(view)[SourceIndex(1)]
        with pytest.raises(ValueError, match="read-only"):
            found.source_pixels[0, 0, 0] = -999

    def test_the_dataset_is_unchanged_by_the_attempt(self):
        dataset = _od_dataset()
        found = SourceLocator(dataset)[SourceIndex(1)]
        with pytest.raises(ValueError, match="read-only"):
            found.pixels[0, 0, 0] = -999
        assert dataset[1][0][0, 0, 0] == 1

    def test_a_crop_is_a_fresh_array_and_stays_writable(self):
        """Cutting allocates, so anything that needs to be drawn on has somewhere to go."""
        crop = SourceLocator(_od_dataset())[SourceIndex(2, 1)].crop()
        crop[0, 0, 0] = -999
        assert crop[0, 0, 0] == -999
