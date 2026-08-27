"""Unit tests for structurer selection and the reserved column schema."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray

from dataeval import Metadata
from dataeval._metadata._structurers import (
    LEGACY_COLUMNS,
    ICStructurer,
    MOTStructurer,
    ODImageStructurer,
    RowBlock,
    StructuredData,
    Structurer,
    reserved_block_columns,
    select_structurer,
)
from dataeval.types import FactorInfo, FactorLevelSchema
from tests.embeddings.test_embeddings import MockDataset, ObjectDetectionTarget


def _od_target(count: int = 2) -> ObjectDetectionTarget:
    return ObjectDetectionTarget(
        np.tile(np.array([[1.0, 1.0, 2.0, 2.0]]), (count, 1)),
        np.arange(count),
        np.full(count, 0.5),
    )


class _Frame:
    """One decoded video frame, declaring its position and timings within its sequence."""

    def __init__(self, frame_index: int) -> None:
        self.frame_index = frame_index
        self.time_s = frame_index * 0.5
        self.pts = frame_index * 1000
        self.pixels = np.zeros((3, 4, 4), dtype=np.float32)


@dataclass
class _BareFrame:
    """A frame declaring none of frame_index, time_s or pts, as a duck-typed stream may."""

    pixels: NDArray[np.float32] = field(default_factory=lambda: np.zeros((3, 4, 4), dtype=np.float32))


class _FrameTracks:
    """One frame's detections, with a track id per detection."""

    def __init__(self, track_ids: Sequence[int]) -> None:
        count = len(track_ids)
        self.boxes = np.tile(np.array([[1.0, 1.0, 2.0, 2.0]]), (count, 1))
        self.labels = np.arange(count) % 3
        self.scores = np.full(count, 0.5)
        self.track_ids = np.asarray(track_ids, dtype=np.intp)


@dataclass
class _MOTTarget:
    """One sequence's per-frame detections."""

    frame_tracks: list[_FrameTracks]


def _track_ids(per_frame: int | Sequence[int]) -> Sequence[int]:
    """A detection count means ids 0..n-1; an explicit sequence is used as given."""
    return range(per_frame) if isinstance(per_frame, int) else per_frame


def _mot_dataset(shapes, metadata=None, bare_frames: bool = False):
    """Build a tracking dataset from per-sequence, per-frame detection descriptions.

    Each frame is either a detection count — ids ``0..n-1``, so a track of a given id
    continues across frames — or an explicit list of track ids, ``-1`` for untracked.
    ``shapes=[[2, 0], [[3, -1]]]`` is two videos: the first has two frames holding two
    detections and none, the second one frame holding a detection on track 3 and an
    unlinked one.
    """
    items = [[_BareFrame() if bare_frames else _Frame(j) for j in range(len(per_frame))] for per_frame in shapes]
    targets = [_MOTTarget([_FrameTracks(_track_ids(frame)) for frame in per_frame]) for per_frame in shapes]
    return MockDataset(items, targets, metadata)


@pytest.mark.required
class TestStructurerSelection:
    """Dispatch reads the target only; MAITE places no constraint on the item."""

    def test_str_item_with_od_target(self):
        """An item that is a path string still dispatches to object detection."""
        dataset = MockDataset(["/data/0.png", "/data/1.png"], [_od_target(), _od_target(1)])
        assert isinstance(select_structurer(dataset), ODImageStructurer)

    def test_str_item_with_od_target_structures_end_to_end(self):
        """The whole pipeline runs without ever loading the item."""
        dataset = MockDataset(["/data/0.png", "/data/1.png"], [_od_target(), _od_target(1)])
        md = Metadata(dataset)
        assert md.rows_at("instance").height == 3

    def test_arbitrary_item_object_with_od_target(self):
        """A lazy loader, a PIL handle — anything that is not an Array is fine."""
        dataset = MockDataset([object(), object()], [_od_target(), _od_target()])
        assert isinstance(select_structurer(dataset), ODImageStructurer)

    def test_mock_backed_od_dataset(self):
        """Mock auto-vivifies every attribute, including ``track_ids``."""
        target = Mock(spec=ObjectDetectionTarget)
        target.boxes = np.array([[1.0, 1.0, 2.0, 2.0]])
        target.labels = np.array([0])
        target.scores = np.array([0.5])
        dataset = MockDataset([np.zeros((3, 4, 4))], [target])

        assert isinstance(target, ObjectDetectionTarget)
        assert isinstance(select_structurer(dataset), ODImageStructurer)

    def test_arraylike_target_dispatches_to_classification(self):
        dataset = MockDataset(np.zeros((2, 3, 4, 4)), np.eye(2))
        assert isinstance(select_structurer(dataset), ICStructurer)

    def test_tracking_target_dispatches_to_mot(self):
        assert isinstance(select_structurer(_mot_dataset([[2, 1], [1]])), MOTStructurer)

    def test_bare_mock_target_is_not_mistaken_for_tracking(self):
        """A bare Mock answers every hasattr, so presence alone cannot decide the task.

        MAITE declares ``frame_tracks`` as a sequence and leaves
        ``MultiobjectTrackingTarget`` without ``@runtime_checkable``, so dispatch has to
        duck-type — and has to check the attribute's *type*, not just its existence, or
        every attribute-fabricating stand-in lands on the tracking structurer.
        """
        target = Mock()
        target.boxes = np.array([[1.0, 1.0, 2.0, 2.0]])
        target.labels = np.array([0])
        target.scores = np.array([0.5])
        dataset = MockDataset([np.zeros((3, 4, 4))], [target])

        assert isinstance(select_structurer(dataset), ODImageStructurer)

    def test_detection_target_does_not_dispatch_to_mot(self):
        """The tracking predicate is checked first, so it must answer False here."""
        dataset = MockDataset([np.zeros((3, 4, 4))], [_od_target()])
        assert isinstance(select_structurer(dataset), ODImageStructurer)

    def test_explicit_mot_task_overrides_inference(self):
        dataset = MockDataset([np.zeros((3, 4, 4))], ["not a target"])
        assert isinstance(select_structurer(dataset, task="MOT"), MOTStructurer)

    def test_unrecognized_target_names_only_the_target(self):
        dataset = MockDataset([np.zeros((3, 4, 4))], ["not a target"])
        with pytest.raises(TypeError, match="Unable to infer a task from target type str"):
            select_structurer(dataset)

    def test_explicit_task_overrides_inference(self):
        dataset = MockDataset([np.zeros((3, 4, 4))], ["not a target"])
        assert isinstance(select_structurer(dataset, task="OD"), ODImageStructurer)

    def test_unknown_task_raises(self):
        dataset = MockDataset([np.zeros((3, 4, 4))], [_od_target()])
        with pytest.raises(ValueError, match="Unknown task"):
            select_structurer(dataset, task="segmentation")  # type: ignore

    def test_empty_dataset_falls_back_to_classification(self):
        """Silently here; Metadata is what warns, from the frame that knows the caller."""
        assert isinstance(select_structurer(MockDataset([], [])), ICStructurer)


@pytest.mark.required
class TestReservedBlockColumns:
    """One producer of the reserved column layout, so it cannot drift."""

    def test_unsupplied_legacy_columns_are_null(self):
        columns = reserved_block_columns("unit", 2, item_index=[0, 1])
        assert set(columns) == {"level", *LEGACY_COLUMNS}
        assert columns["level"] == ["unit", "unit"]
        assert columns["item_index"] == [0, 1]
        assert columns["box"] == [None, None]

    def test_level_key_columns_are_omitted_unless_supplied(self):
        assert "instance_index" not in reserved_block_columns("unit", 1, item_index=[0])
        assert reserved_block_columns("instance", 1, instance_index=[0])["instance_index"] == [0]

    def test_ndarrays_are_passed_through_untouched(self):
        """Arrays reach polars as arrays; listifying them here is the cost we removed."""
        supplied = np.array([3, 4], dtype=np.intp)
        columns = reserved_block_columns("unit", 2, class_label=supplied)
        assert columns["class_label"] is supplied
        assert isinstance(columns["class_label"], np.ndarray)

    def test_non_arrays_are_normalized_to_lists(self):
        columns = reserved_block_columns("unit", 2, class_label=(3, 4))
        assert columns["class_label"] == [3, 4]

    def test_non_reserved_name_raises(self):
        with pytest.raises(ValueError, match="are not reserved columns"):
            reserved_block_columns("unit", 1, brightness=[0.5])


@pytest.mark.required
class TestReservedColumnParity:
    """from_factors and the dataset path must agree on the reserved schema."""

    @staticmethod
    def _reserved(md: Metadata) -> list[str]:
        reserved = set(LEGACY_COLUMNS) | {"level"}
        return [name for name in md.dataframe.columns if name in reserved]

    def test_from_factors_matches_ic_dataset_schema(self):
        factors = {"weather": np.array(["sun", "rain", "sun"])}
        labels = np.array([0, 1, 0])

        from_factors = Metadata.from_factors(factors, labels)
        from_dataset = Metadata(
            MockDataset(
                np.zeros((3, 3, 4, 4)),
                np.eye(3, 2)[labels],
                [{"weather": value} for value in factors["weather"].tolist()],
            ),
        )

        # Same reserved columns, in the same order, from both constructors.
        assert self._reserved(from_factors) == self._reserved(from_dataset)
        assert set(self._reserved(from_factors)) == set(LEGACY_COLUMNS) | {"level"}

        # Target-row content agrees too, except ``score``, which raw factors never carry.
        # The two differ in structure by design: a dataset knows which items exist and so
        # carries an item level as well, while raw factor arrays are only the target rows.
        for name in ("item_index", "target_index", "class_label", "box"):
            assert (
                from_factors.rows_at(from_factors.label_level)[name].to_list()
                == from_dataset.rows_at(from_dataset.label_level)[name].to_list()
            )

        # Each tags its rows with its own target level, which is what rows_at(label_level) returns.
        assert from_factors.rows_at(from_factors.label_level)["level"].to_list() == ["unit"] * 3
        assert from_dataset.rows_at(from_dataset.label_level)["level"].to_list() == ["instance"] * 3

    def test_from_factors_at_instance_level_keeps_the_same_reserved_columns(self):
        md = Metadata.from_factors({"iou": np.array([0.1, 0.2])}, np.array([0, 1]), level="instance")
        assert self._reserved(md) == self._reserved(Metadata.from_factors({"a": np.array([1, 2])}))


def _two_level_blocks() -> list[RowBlock]:
    """A unit block of 2 and an instance block of 3, wired for propagation."""
    parents = np.array([0, 0, 1], dtype=np.intp)
    return [
        RowBlock("unit", 2, reserved_block_columns("unit", 2, item_index=[0, 1]), {"unit": np.arange(2)}),
        RowBlock(
            "instance",
            3,
            reserved_block_columns("instance", 3, item_index=parents),
            {"unit": parents, "instance": np.arange(3)},
        ),
    ]


@pytest.mark.required
class TestOneNameOneLevel:
    """A factor is one column, and a column belongs to exactly one level."""

    def test_a_name_at_two_levels_is_rejected(self):
        with pytest.raises(ValueError, match="declared at both the 'unit' and 'instance' levels"):
            StructuredData(
                _two_level_blocks(),
                {"unit": {"timestamp": [0.0, 1.0]}, "instance": {"timestamp": [0.0, 0.0, 1.0]}},
            )

    def test_the_error_suggests_qualified_names(self):
        with pytest.raises(ValueError, match="'unit_timestamp' and 'instance_timestamp'"):
            StructuredData(
                _two_level_blocks(),
                {"unit": {"timestamp": [0.0, 1.0]}, "instance": {"timestamp": [0.0, 0.0, 1.0]}},
            )

    def test_distinct_names_per_level_are_fine(self):
        data = StructuredData(
            _two_level_blocks(),
            {"unit": {"weather": ["sun", "rain"]}, "instance": {"iou": [0.1, 0.2, 0.3]}},
        )
        rows = data.to_rows()
        # The unit factor propagates onto instance rows; the instance factor is null above.
        assert rows["weather"] == ["sun", "rain", "sun", "sun", "rain"]
        assert rows["iou"] == [None, None, 0.1, 0.2, 0.3]

    def test_the_real_structurers_do_not_trip_it(self):
        """IC and OD both run two metadata merges that can produce the same name."""
        shared = [{"weather": "sun"}, {"weather": "rain"}, {"weather": "sun"}]
        Metadata(MockDataset(np.zeros((3, 3, 16, 16)), np.eye(3)[[0, 1, 0]], shared))._structure()
        Metadata(MockDataset(np.zeros((3, 3, 16, 16)), [_od_target(2)] * 3, shared))._structure()

    def test_mot_does_not_trip_it_either(self):
        """MOT runs the same two merges, at the sequence and unit levels."""
        shared = [{"weather": "sun"}, {"weather": "rain"}]
        Metadata(_mot_dataset([[2, 1], [1]], shared))._structure()


@pytest.mark.required
class TestToFrame:
    """The levelled build must produce exactly the frame the flat builder did."""

    @staticmethod
    def _bundles():
        shared = [{"weather": "sun"}, {"weather": "rain"}, {"weather": "sun"}]
        images = np.zeros((3, 3, 16, 16))
        return {
            "IC": MockDataset(images, np.eye(3)[[0, 1, 0]], shared),
            "OD": MockDataset(images, [_od_target(2)] * 3, shared),
            "OD-no-detections": MockDataset(images, [_od_target(0)] * 3, shared),
            "MOT": _mot_dataset(_SHAPES, shared[:2]),
        }

    def test_values_and_column_order_match_the_flat_builder(self):
        for name, dataset in self._bundles().items():
            data = select_structurer(dataset, None).build(dataset)
            flat, frame = pl.DataFrame(data.to_rows()), data.to_frame()
            assert frame.columns == flat.columns, name
            assert frame.height == flat.height, name
            for column in flat.columns:
                assert frame[column].to_list() == flat[column].to_list(), f"{name}: {column}"

    def test_column_order_is_reserved_then_factors(self):
        data = StructuredData(
            _two_level_blocks(),
            {"unit": {"weather": ["sun", "rain"]}, "instance": {"iou": [0.1, 0.2, 0.3]}},
        )
        order = data.column_order
        assert order[: len(LEGACY_COLUMNS) + 1] == ("level", *LEGACY_COLUMNS)
        assert order[-2:] == ("weather", "iou")
        assert data.to_frame().columns == list(order)

    def test_a_column_null_at_every_level_survives_as_null(self):
        """``box`` exists for classification only as a column of nothing."""
        dataset = MockDataset(np.zeros((3, 3, 16, 16)), np.eye(3)[[0, 1, 0]], [{}] * 3)
        frame = select_structurer(dataset, None).build(dataset).to_frame()
        assert frame["box"].dtype == pl.Null
        assert frame["box"].to_list() == [None] * frame.height

    def test_arrays_keep_their_own_dtype(self):
        """Reaching polars as arrays, not Python floats, keeps boxes and scores narrow."""
        dataset = MockDataset(np.zeros((3, 3, 16, 16)), [_od_target(2)] * 3, [{}] * 3)
        frame = select_structurer(dataset, None).build(dataset).to_frame()
        assert frame["box"].dtype == pl.Array(pl.Float32, 4)

    def test_factor_propagation_and_sibling_nulls_are_preserved(self):
        data = StructuredData(
            _two_level_blocks(),
            {"unit": {"weather": ["sun", "rain"]}, "instance": {"iou": [0.1, 0.2, 0.3]}},
        )
        frame = data.to_frame()
        assert frame["weather"].to_list() == ["sun", "rain", "sun", "sun", "rain"]
        assert frame["iou"].to_list() == [None, None, 0.1, 0.2, 0.3]

    def test_a_nested_factor_nulls_the_row_with_no_ancestor(self):
        """polars cannot scatter into a List column, so the null has to come from the gather."""
        unit, instance = _two_level_blocks()
        orphaned = RowBlock(
            "instance",
            3,
            instance.columns,
            {"unit": np.array([0, -1, 1], dtype=np.intp), "instance": np.arange(3)},
        )
        data = StructuredData([unit, orphaned], {"unit": {"tags": [[1, 2], [3, 4]]}})
        assert data.to_frame()["tags"].to_list() == pl.DataFrame(data.to_rows())["tags"].to_list()

    def test_blocks_that_are_all_empty_stay_empty(self):
        """Every column dropped leaves no height to broadcast a literal against."""
        block = RowBlock("unit", 0, reserved_block_columns("unit", 0, item_index=[]), {"unit": np.empty(0, np.intp)})
        assert StructuredData([block], {}).to_frame().height == 0


@pytest.mark.required
class TestOneBlockPerLevel:
    """A level's rows are one block, because the layout is keyed by level."""

    def test_two_blocks_at_one_level_are_rejected(self):
        unit, instance = _two_level_blocks()
        with pytest.raises(ValueError, match=r"Level\(s\) \['unit'\] have more than one row block"):
            StructuredData([unit, unit, instance], {"unit": {}, "instance": {}})

    def test_the_error_says_to_concatenate(self):
        unit, instance = _two_level_blocks()
        with pytest.raises(ValueError, match="one block per level"):
            StructuredData([unit, unit, instance], {"unit": {}, "instance": {}})

    def test_one_block_per_level_is_accepted(self):
        data = StructuredData(_two_level_blocks(), {"unit": {}, "instance": {}})
        assert list(data.layout.counts) == ["unit", "instance"]

    def test_every_structurer_emits_one_block_per_level(self):
        """The guard states an assumption the real structurers already satisfy."""
        shared = [{"weather": "sun"}, {"weather": "rain"}, {"weather": "sun"}]
        for metadata in (
            Metadata(MockDataset(np.zeros((3, 3, 16, 16)), np.eye(3)[[0, 1, 0]], shared)),
            Metadata(MockDataset(np.zeros((3, 3, 16, 16)), [_od_target(2)] * 3, shared)),
            Metadata(_mot_dataset([[2, 1], [1]], shared[:2])),
        ):
            metadata._structure()
            levels = [level for level, _, _ in metadata._layout.blocks]
            assert len(levels) == len(set(levels))


# Two videos: the first has three frames holding 2, 0 and 1 detections, the second two
# frames holding 1 and 3. Deliberately uneven, and deliberately not equal to the frame
# count, so that per-frame and per-instance row counts cannot be confused.
_SHAPES = [[2, 0, 1], [1, 3]]


@pytest.mark.required
class TestMOTStructurer:
    """Tracking is the first task whose item level is not ``unit``."""

    def test_schema_is_four_levels_deep(self):
        structurer = MOTStructurer()
        assert structurer.levels.levels == ("sequence", "unit", "track", "instance")
        assert structurer.item_level == "sequence"
        assert structurer.label_level == "instance"
        assert structurer.multi_target is True

    def test_an_instance_hangs_off_both_its_frame_and_its_track(self):
        assert MOTStructurer().levels.parents_of("instance") == ("unit", "track")
        assert MOTStructurer().levels.ancestors("instance") == ("unit", "track", "sequence")

    def test_row_counts_are_per_level(self):
        # Track ids restart at 0 in each frame, so sequence 0 spans tracks {0, 1} and
        # sequence 1 spans {0, 1, 2}: five tracks over seven detections.
        md = Metadata(_mot_dataset(_SHAPES))
        assert md.level_counts == {"sequence": 2, "unit": 5, "track": 5, "instance": 7}

    def test_a_sequence_with_no_detections_keeps_its_rows(self):
        """Its frames and its item-level factors survive; only instance rows are absent."""
        md = Metadata(_mot_dataset([[2, 1], [0, 0]]))
        assert md.level_counts == {"sequence": 2, "unit": 4, "track": 2, "instance": 3}
        assert md.rows_at("instance")["item_index"].to_list() == [0, 0, 0]

    def test_the_compound_key_is_unique(self):
        """instance_index counts within a frame, so it repeats across a sequence."""
        instances = Metadata(_mot_dataset(_SHAPES)).rows_at("instance")
        key = instances.select("item_index", "unit_index", "instance_index")
        assert key.n_unique() == instances.height
        # Without unit_index it would not be: sequence 0 has an instance 0 in two frames.
        assert instances.select("item_index", "instance_index").n_unique() < instances.height

    def test_instance_index_counts_within_the_frame(self):
        instances = Metadata(_mot_dataset(_SHAPES)).rows_at("instance")
        assert instances["instance_index"].to_list() == [0, 1, 0, 0, 0, 1, 2]

    def test_target_index_counts_within_the_item(self):
        """The legacy spelling keeps its meaning: position within the dataset item."""
        instances = Metadata(_mot_dataset(_SHAPES)).rows_at("instance")
        assert instances["target_index"].to_list() == [0, 1, 2, 0, 1, 2, 3]

    def test_unit_index_joins_an_instance_to_its_frame(self):
        md = Metadata(_mot_dataset(_SHAPES))
        instances, frames = md.rows_at("instance"), md.rows_at("unit")
        joined = instances.join(frames, on=["item_index", "unit_index"], how="inner")
        assert joined.height == instances.height
        # Sequence 0's third detection was observed in its third frame, not its second.
        assert instances["unit_index"].to_list() == [0, 0, 2, 0, 1, 1, 1]

    def test_frame_index_falls_back_to_decode_order(self):
        """A frame that does not declare frame_index still gets a usable key."""
        frames = Metadata(_mot_dataset(_SHAPES, bare_frames=True)).rows_at("unit")
        assert frames["unit_index"].to_list() == [0, 1, 2, 0, 1]

    def test_track_id_is_a_reserved_column_not_a_factor(self):
        """A track number is an identifier; binning it into bias analysis is meaningless."""
        md = Metadata(_mot_dataset(_SHAPES))
        assert "track_id" not in md.factor_names
        assert md.rows_at("instance")["track_id"].to_list() == [0, 1, 0, 0, 0, 1, 2]

    def test_per_frame_metadata_lands_at_the_unit_level(self):
        """A video's list-valued metadata is per frame, never per detection."""
        metadata = [
            {"id": 0, "weather": "sun", "temp": [10.0, 11.0, 12.0]},
            {"id": 1, "weather": "rain", "temp": [20.0, 21.0]},
        ]
        md = Metadata(_mot_dataset(_SHAPES, metadata))

        assert md.dropped_factors == {}
        assert md.rows_at("unit")["temp"].to_list() == [10.0, 11.0, 12.0, 20.0, 21.0]
        # And propagates down to the instances observed in each frame.
        assert md.rows_at("instance")["temp"].to_list() == [10.0, 10.0, 12.0, 20.0, 21.0, 21.0, 21.0]

    def test_item_metadata_stays_at_the_sequence_level(self):
        metadata = [{"id": 0, "weather": "sun"}, {"id": 1, "weather": "rain"}]
        md = Metadata(_mot_dataset(_SHAPES, metadata))
        assert md.rows_at("sequence")["weather"].to_list() == ["sun", "rain"]
        assert md.rows_at("instance")["weather"].to_list() == ["sun"] * 3 + ["rain"] * 4

    def test_more_frames_than_targets_is_raised_not_truncated(self):
        """Truncating would drop real detections, or misattribute them, with no signal."""
        dataset = _mot_dataset([[2, 1]])
        dataset.targets[0].frame_tracks = dataset.targets[0].frame_tracks[:1]
        with pytest.raises(ValueError, match="yields more frames than"):
            Metadata(dataset)._structure()

    def test_fewer_frames_than_targets_is_raised_too(self):
        dataset = _mot_dataset([[2, 1]])
        dataset.data[0] = dataset.data[0][:1]
        with pytest.raises(ValueError, match="video stream yielded only 1"):
            Metadata(dataset)._structure()

    def test_a_dataset_with_no_detections_at_all_still_structures(self):
        md = Metadata(_mot_dataset([[0], [0, 0]]))
        assert md.level_counts == {"sequence": 2, "unit": 3, "track": 0, "instance": 0}
        assert md.class_labels.tolist() == []

    def test_class_labels_align_with_the_instance_rows(self):
        md = Metadata(_mot_dataset(_SHAPES))
        assert len(md.class_labels) == md.level_counts["instance"]
        assert md.item_indices.tolist() == [0, 0, 0, 1, 1, 1, 1]


# One video, three frames. Track 7 is seen in frames 0 and 1; track 12 in frames 0 and 2,
# so it has a gap. A second video sees track 7 again — a different object — plus one
# detection no tracker linked.
_TRACKED = [[[7, 12], [7], [12]], [[7, -1]]]


@pytest.mark.required
class TestTrackLevel:
    """A track is one object across a sequence; each instance is one observation of it."""

    def test_tracks_are_scoped_to_their_sequence(self):
        """The same id in two videos is two objects, so it is two rows."""
        tracks = Metadata(_mot_dataset(_TRACKED)).rows_at("track")
        assert tracks.select("item_index", "track_id").rows() == [(0, 7), (0, 12), (1, 7)]

    def test_track_index_is_dense_within_the_sequence(self):
        """The dataset's own ids may be sparse or arbitrary; the level's key is not."""
        tracks = Metadata(_mot_dataset(_TRACKED)).rows_at("track")
        assert tracks["track_id"].to_list() == [7, 12, 7]
        assert tracks["track_index"].to_list() == [0, 1, 0]

    def test_an_instance_carries_its_tracks_key(self):
        instances = Metadata(_mot_dataset(_TRACKED)).rows_at("instance")
        assert instances["track_index"].to_list() == [0, 1, 0, 1, 0, -1]
        assert instances["track_id"].to_list() == [7, 12, 7, 12, 7, -1]

    def test_derived_factors_distinguish_length_from_span(self):
        """They differ exactly when a track has gaps, which is what makes the pair useful."""
        tracks = Metadata(_mot_dataset(_TRACKED)).rows_at("track")
        assert tracks["track_length"].to_list() == [2, 2, 1]
        assert tracks["frame_span"].to_list() == [2, 3, 1]

    def test_duration_spans_first_to_last_observation(self):
        """_Frame puts frames half a second apart, so track 12 spans frames 0 to 2."""
        tracks = Metadata(_mot_dataset(_TRACKED)).rows_at("track")
        assert tracks["duration_s"].to_list() == [0.5, 1.0, 0.0]

    def test_track_factors_propagate_down_to_their_observations(self):
        instances = Metadata(_mot_dataset(_TRACKED)).rows_at("instance")
        assert instances["track_length"].to_list() == [2, 2, 2, 2, 1, None]
        assert instances["frame_span"].to_list() == [2, 3, 2, 3, 1, None]

    def test_an_untracked_detection_has_no_track_ancestor(self):
        """Null rather than a singleton track: the data says the detection has none."""
        md = Metadata(_mot_dataset(_TRACKED))
        untracked = md.rows_at("instance").filter(pl.col("track_id") == -1)
        assert untracked.height == 1
        assert untracked["track_index"].to_list() == [-1]
        assert untracked.select("track_length", "frame_span", "duration_s").rows() == [(None, None, None)]

    def test_a_partly_null_track_factor_leaves_factor_analysis_at_that_view(self):
        """It cannot be binned there, but reads in full at its own level."""
        md = Metadata(_mot_dataset(_TRACKED))
        assert "track_length" not in md.factor_names
        assert "track_length" in md.at("track").factor_names
        # Excluded from analysis, still present in the dataframe.
        assert "track_length" in md.dataframe.columns

    def test_track_factors_are_analysable_when_everything_is_tracked(self):
        md = Metadata(_mot_dataset([[[7, 12], [7], [12]]]))
        assert "track_length" in md.factor_names
        assert md.factor_data.shape[0] == md.level_counts["instance"]

    def test_frame_and_track_factors_are_invisible_to_each_other(self):
        """Siblings under the sequence, so neither propagates to the other."""
        md = Metadata(_mot_dataset([[[7, 12], [7], [12]]]))
        assert "time_s" in md.at("unit").factor_names
        assert "time_s" not in md.at("track").factor_names
        assert "track_length" in md.at("track").factor_names
        assert "track_length" not in md.at("unit").factor_names

    def test_a_dataset_with_nothing_tracked_has_an_empty_track_level(self):
        md = Metadata(_mot_dataset([[[-1, -1], [-1]]]))
        assert md.level_counts["track"] == 0
        assert md.rows_at("instance")["track_index"].to_list() == [-1, -1, -1]
        assert "track_length" not in md.factor_names
        # The instance projection still works; only the track factors are absent.
        assert md.factor_data.shape[0] == 3

    def test_factors_can_be_added_at_the_track_level(self):
        """The point of the level: organize metadata by track rather than by frame."""
        md = Metadata(_mot_dataset([[[7, 12], [7], [12]]]))
        md.add_factors({"mean_iou": np.array([0.8, 0.4])}, level="track")
        assert md.rows_at("track")["mean_iou"].to_list() == [0.8, 0.4]
        # And it reaches every observation of that track.
        assert md.rows_at("instance")["mean_iou"].to_list() == [0.8, 0.4, 0.8, 0.4]

    def test_frame_timings_become_unit_level_factors(self):
        md = Metadata(_mot_dataset(_TRACKED))
        assert md.rows_at("unit")["time_s"].to_list() == [0.0, 0.5, 1.0, 0.0]
        assert md.rows_at("unit")["pts"].to_list() == [0, 1000, 2000, 0]

    def test_a_stream_without_timings_produces_no_timing_factors(self):
        """All-or-nothing: a partly populated numeric factor cannot be binned."""
        md = Metadata(_mot_dataset(_TRACKED, bare_frames=True))
        assert "time_s" not in md.dataframe.columns
        assert "pts" not in md.dataframe.columns
        assert "duration_s" not in md.dataframe.columns
        # The rest of the level survives.
        assert md.at("track").rows_at("track")["track_length"].to_list() == [2, 2, 1]

    def test_frame_dimensions_become_unit_level_factors(self):
        """The walk is the only thing that ever holds a decoded frame, so it reads the size."""
        md = Metadata(_mot_dataset(_TRACKED))
        assert md.rows_at("unit")["width"].to_list() == [4, 4, 4, 4]
        assert md.rows_at("unit")["height"].to_list() == [4, 4, 4, 4]

    def test_frame_dimensions_survive_a_stream_declaring_nothing_else(self):
        """Pixels are the one thing a frame must carry, unlike the optional timings."""
        md = Metadata(_mot_dataset(_TRACKED, bare_frames=True))
        assert "time_s" not in md.dataframe.columns
        assert md.rows_at("unit")["width"].to_list() == [4, 4, 4, 4]

    def test_a_derived_name_displaces_a_metadata_key_of_the_same_name(self):
        """The structurer reads it off the dataset itself, which outranks a dict key."""
        metadata = [{"id": 0, "track_length": 99}, {"id": 1, "track_length": 98}]
        md = Metadata(_mot_dataset(_TRACKED, metadata))
        assert md.rows_at("track")["track_length"].to_list() == [2, 2, 1]
        assert md.rows_at("sequence")["track_length"].to_list() == [None, None]


@pytest.mark.required
class TestUnitType:
    """``unit_type`` names the medium one ``unit`` row holds."""

    def test_ic_units_are_images(self):
        dataset = MockDataset(np.zeros((4, 3, 16, 16)), np.eye(4, 2)[[0, 1, 0, 1]])
        assert Metadata(dataset).unit_type == "image"

    def test_od_units_are_images(self):
        dataset = MockDataset(["/data/0.png", "/data/1.png"], [_od_target(), _od_target(1)])
        assert Metadata(dataset).unit_type == "image"

    def test_mot_units_are_frames(self):
        assert Metadata(_mot_dataset(_SHAPES)).unit_type == "frame"

    def test_factors_only_units_are_items(self):
        md = Metadata.from_factors({"a": np.array([0, 1, 0])})
        assert md.unit_type == "item"

    def test_str_names_the_unit_type(self):
        md = Metadata(_mot_dataset(_SHAPES))
        md._structure()
        assert "units=frame" in str(md)

    def test_unit_type_is_a_plain_str(self):
        """Not a Literal or Enum — a new modality must not require a type edit."""
        assert type(Metadata(_mot_dataset(_SHAPES)).unit_type) is str


@pytest.mark.required
class TestUnitLevelVocabulary:
    """The media-unit level is named ``unit``, on every task."""

    def test_hierarchy_names_unit(self):
        from dataeval.types._factors import _FACTOR_LEVEL_HIERARCHY

        assert list(_FACTOR_LEVEL_HIERARCHY) == ["sequence", "unit", "track", "instance"]
        assert _FACTOR_LEVEL_HIERARCHY["unit"] == ("sequence",)
        assert _FACTOR_LEVEL_HIERARCHY["instance"] == ("unit", "track")

    def test_ic_levels(self):
        dataset = MockDataset(np.zeros((4, 3, 16, 16)), np.eye(4, 2)[[0, 1, 0, 1]])
        md = Metadata(dataset)
        assert md.levels == ("unit", "instance")
        assert md.item_level == "unit"
        assert md.label_level == "instance"

    def test_od_levels(self):
        dataset = MockDataset(["/data/0.png", "/data/1.png"], [_od_target(), _od_target(1)])
        md = Metadata(dataset)
        assert md.levels == ("unit", "instance")
        assert md.item_level == "unit"

    def test_mot_levels(self):
        md = Metadata(_mot_dataset(_SHAPES))
        assert md.levels == ("sequence", "unit", "track", "instance")
        assert md.item_level == "sequence"
        assert md.levels.__class__ is tuple

    def test_mot_instance_parents_are_unit_and_track(self):
        assert MOTStructurer().levels.parents_of("instance") == ("unit", "track")
        assert MOTStructurer().levels.ancestors("instance") == ("unit", "track", "sequence")

    def test_level_column_never_says_image(self):
        md = Metadata(_mot_dataset(_SHAPES))
        values = set(md.dataframe["level"].unique().to_list())
        assert "unit" in values
        assert "image" not in values

    def test_rows_at_unit(self):
        md = Metadata(_mot_dataset(_SHAPES))
        assert md.rows_at("unit").height == md.level_counts["unit"]

    def test_image_is_no_longer_a_schema_level(self):
        with pytest.raises(ValueError, match="Unknown level"):
            FactorLevelSchema.of("image")  # type: ignore[arg-type]

    def test_factor_info_level_default_is_unit(self):
        assert FactorInfo("categorical").level == "unit"


@pytest.mark.required
class TestUnitIndexColumn:
    """The per-sequence frame position column is keyed to the unit level."""

    def test_mot_unit_rows_carry_unit_index(self):
        md = Metadata(_mot_dataset(_SHAPES))
        units = md.rows_at("unit")
        assert "unit_index" in units.columns
        assert "image_index" not in md.dataframe.columns

    def test_mot_instance_rows_carry_unit_index(self):
        md = Metadata(_mot_dataset(_SHAPES))
        instances = md.rows_at("instance")
        assert instances["unit_index"].null_count() == 0

    def test_unit_index_is_reserved(self):
        from dataeval._metadata._structurers import RESERVED_COLUMNS, safe_column_name

        assert "unit_index" in RESERVED_COLUMNS
        assert "image_index" not in RESERVED_COLUMNS
        assert safe_column_name("unit_index") == "metadata_unit_index"
        assert safe_column_name("image_index") == "image_index"

    def test_the_companion_namespace_is_reserved_at_the_tail(self):
        """A reserved column is collided with head-on and escaped by prefix; the namespace
        binning writes into is reached by a name's tail, so it is escaped there instead."""
        from dataeval._metadata._structurers import safe_column_name

        assert safe_column_name("w#") == "w#_metadata"
        assert safe_column_name("w↕") == "w↕_metadata"
        # Only the tail is reserved: the characters are ordinary anywhere else in a name.
        assert safe_column_name("w#ish") == "w#ish"
        assert safe_column_name("weather") == "weather"


@pytest.mark.required
class TestStructurerDeclarationChecks:
    """The three level declarations are interdependent, so a mismatch is a class-creation error."""

    def test_item_level_outside_the_schema_is_rejected(self):
        with pytest.raises(TypeError, match=r"item_level is 'frame'.*not one of its declared levels"):

            class _BadItemLevel(Structurer):
                levels = FactorLevelSchema.of("unit")
                item_level = "frame"  # type: ignore[assignment]

    def test_label_level_outside_the_schema_is_rejected(self):
        with pytest.raises(TypeError, match=r"label_level is 'box'.*not one of its declared levels"):

            class _BadLabelLevel(Structurer):
                levels = FactorLevelSchema.of("unit")
                label_level = "box"  # type: ignore[assignment]


@pytest.mark.required
class TestRowBlockAncestry:
    def test_a_level_absent_from_the_mapping_reads_as_no_ancestor(self):
        """A missing level is the block-wide statement that no row has such an ancestor."""
        block = RowBlock("unit", 3, reserved_block_columns("unit", 3, item_index=[0, 1, 2]), {"unit": np.arange(3)})
        np.testing.assert_array_equal(block.positions_at("sequence"), np.full(3, -1, dtype=np.intp))


@pytest.mark.required
class TestStructuredDataEmptyFrame:
    def test_to_frame_with_no_blocks_keeps_the_column_order(self):
        data = StructuredData((), {})
        frame = data.to_frame()
        assert frame.height == 0
        assert tuple(frame.columns) == tuple(data.column_order)


@pytest.mark.required
class TestFactorsStructurerEntryPoints:
    def test_build_from_source_index_needs_one(self):
        """The two builders are not interchangeable: this one has no rows to place onto."""
        from dataeval._metadata._structurers._factors import FactorsStructurer

        with pytest.raises(ValueError, match="requires a structurer built with a source index"):
            FactorsStructurer().build_from_source_index({"a": np.array([1, 2])}, None)


@pytest.mark.required
class TestRowLayoutPartialAncestry:
    """Whether every row at one level records an ancestor at another."""

    @staticmethod
    def _layout(positions):
        from dataeval._metadata._structurers._layout import RowLayout

        # A unit block first, so the scan has a block to step past before the match.
        return RowLayout((
            ("unit", 2, {"unit": np.arange(2, dtype=np.intp)}),
            ("instance", 3, {"unit": np.asarray(positions, dtype=np.intp)}),
        ))

    def test_a_level_with_no_rows_reports_no_gap(self):
        from dataeval._metadata._structurers._layout import RowLayout

        assert RowLayout(()).partial_ancestry("unit", "instance") is False

    def test_every_row_recording_an_ancestor_is_not_partial(self):
        assert self._layout([0, 0, 1]).partial_ancestry("unit", "instance") is False

    def test_a_negative_position_marks_a_missing_ancestor(self):
        assert self._layout([0, -1, 1]).partial_ancestry("unit", "instance") is True

    def test_a_level_no_block_records_is_not_a_gap(self):
        assert self._layout([0, 0, 1]).partial_ancestry("sequence", "instance") is False
