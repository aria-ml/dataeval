import pickle
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import pytest

from dataeval import Metadata
from dataeval.core import compute_stats
from dataeval.data import Indices, View
from dataeval.exceptions import MaiteShapeError
from dataeval.flags import ImageStats
from dataeval.protocols import DatumMetadata
from dataeval.types import RemovalPlan, SourceIndex

VOCABULARY = {0: "car", 1: "person"}


@pytest.mark.required
class TestItemAddresses:
    """An item address does what an int does."""

    def test_an_item_address_selects_like_an_int(self, od_dataset):
        ds = od_dataset([[0], [1], [0, 1]], VOCABULARY)
        assert View(ds, Indices([SourceIndex(2), 0])).selection == [2, 0]

    def test_an_item_address_is_removed_like_an_int(self, od_dataset):
        ds = od_dataset([[0], [1], [0, 1]], VOCABULARY)
        assert View(ds, Indices([SourceIndex(1)], exclude=True)).selection == [0, 2]

    def test_numpy_integers_are_ints(self, od_dataset):
        ds = od_dataset([[0], [1], [0, 1]], VOCABULARY)
        assert View(ds, Indices(np.array([2, 0]))).selection == [2, 0]

    def test_a_one_shot_iterator_is_kept_for_the_repr(self, od_dataset):
        ds = od_dataset([[0], [1], [0, 1]], VOCABULARY)
        op = Indices(i for i in [2, 0])
        assert View(ds, op).selection == [2, 0]
        assert "indices=[2, 0]" in repr(op)

    def test_a_removal_plan_is_accepted(self, od_dataset):
        ds = od_dataset([[0], [1], [0, 1]], VOCABULARY)
        view = View(ds, Indices(RemovalPlan([1, SourceIndex(2, 0)]), exclude=True))
        assert view.selection == [0, 2]
        assert view[1][1].labels.tolist() == [1]

    def test_requires_follows_the_addresses(self):
        assert Indices([1]).requires is None
        assert Indices([SourceIndex(0, 1)], exclude=True).requires == "any_target"
        assert Indices([SourceIndex(0, 1, "track")], exclude=True).requires == "multiobject_tracking"


@pytest.mark.required
class TestDetectionAddresses:
    """With exclude=True, a detection address removes that detection."""

    def test_a_detection_address_drops_that_detection(self, od_dataset):
        ds = od_dataset([[0, 1, 0], [1]], VOCABULARY)
        target = View(ds, Indices([SourceIndex(0, 1)], exclude=True))[0][1]
        assert target.labels.tolist() == [0, 0]
        assert len(target.boxes) == 2
        assert len(target.scores) == 2

    def test_per_class_scores_stay_aligned(self, od_dataset):
        ds = od_dataset([[0, 1, 0]], VOCABULARY, per_class=True)
        target = View(ds, Indices([SourceIndex(0, 1)], exclude=True))[0][1]
        assert target.scores.shape == (2, 2)
        assert target.scores.argmax(axis=1).tolist() == target.labels.tolist()

    def test_other_items_are_left_alone(self, od_dataset):
        ds = od_dataset([[0, 1, 0], [1]], VOCABULARY)
        assert View(ds, Indices([SourceIndex(0, 1)], exclude=True))[1][1].labels.tolist() == [1]

    def test_an_emptied_image_stays(self, od_dataset):
        ds = od_dataset([[0]], VOCABULARY)
        view = View(ds, Indices([SourceIndex(0, 0)], exclude=True))
        assert len(view) == 1
        assert view[0][1].labels.tolist() == []

    def test_a_detection_inside_a_removed_item_has_no_effect(self, od_dataset):
        ds = od_dataset([[0, 1], [1]], VOCABULARY)
        view = View(ds, Indices([0, SourceIndex(0, 0)], exclude=True))
        assert view.selection == [1]
        assert view[0][1].labels.tolist() == [1]

    def test_a_key_the_item_does_not_hold_is_ignored(self, od_dataset):
        ds = od_dataset([[0, 1], [1]], VOCABULARY)
        view = View(ds, Indices([SourceIndex(1, 5)], exclude=True))
        assert view[1][1].labels.tolist() == [1]
        assert view[0][1].labels.tolist() == [0, 1]

    def test_metadata_through_the_view_renumbers_the_detections(self, od_dataset):
        ds = od_dataset([[0, 1, 0], [1]], VOCABULARY)
        md = Metadata(View(ds, Indices([SourceIndex(0, 1)], exclude=True)))
        rows = md.rows_at(md.label_level).filter(pl.col("item_index") == 0)
        assert rows["target_index"].to_list() == [0, 1]

    def test_statistics_through_the_view_measure_one_detection_fewer(self, od_dataset):
        ds = od_dataset([[0, 1, 0], [1]], VOCABULARY)
        view = View(ds, Indices([SourceIndex(0, 1)], exclude=True))
        assert _detections_measured(ds) == {0: 3, 1: 1}
        assert _detections_measured(view) == {0: 2, 1: 1}

    def test_the_view_pickles(self, od_dataset):
        ds = od_dataset([[0, 1, 0], [1]], VOCABULARY)
        view = View(ds, Indices(RemovalPlan([1, SourceIndex(0, 1)]), exclude=True))
        copy = pickle.loads(pickle.dumps(view))
        assert copy.selection == [0]
        assert copy[0][1].labels.tolist() == view[0][1].labels.tolist() == [0, 0]


def _detections_measured(dataset: Any) -> dict[int, int]:
    """How many detections compute_stats measured in each item."""
    stats = compute_stats(dataset, stats=ImageStats.DIMENSION_WIDTH, per_target=True)
    return dict(Counter(index.item for index in stats["source_index"] if index.key is not None))


@dataclass
class _SegmentationTarget:
    mask: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


class _Segmentations:
    """A segmentation dataset with one mask per instance. Instance ``i`` fills its mask with ``i + 1``."""

    def __init__(self, labels: list[list[int]]) -> None:
        self._labels = labels
        self.metadata = {"id": "segmentations", "index2label": dict(VOCABULARY)}

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> tuple[Any, _SegmentationTarget, DatumMetadata]:
        n = len(self._labels[index])
        target = _SegmentationTarget(
            mask=np.arange(1, n + 1, dtype=np.intp)[:, None, None] * np.ones((1, 4, 4), dtype=np.intp),
            labels=np.array(self._labels[index], dtype=np.intp),
            scores=np.arange(n, dtype=np.float32),
        )
        return np.zeros((3, 4, 4), dtype=np.float32), target, {"id": index}


@pytest.mark.required
class TestSegmentationAddresses:
    """With exclude=True, a detection address removes that instance's mask, label and score."""

    def test_the_named_instance_is_gone_from_each_array(self):
        target = View(_Segmentations([[0, 1, 0], [1]]), Indices([SourceIndex(0, 1)], exclude=True))[0][1]
        assert target.mask[:, 0, 0].tolist() == [1, 3]
        assert target.labels.tolist() == [0, 0]
        assert target.scores.tolist() == [0.0, 2.0]

    def test_other_items_are_left_alone(self):
        target = View(_Segmentations([[0, 1, 0], [1]]), Indices([SourceIndex(0, 1)], exclude=True))[1][1]
        assert target.mask[:, 0, 0].tolist() == [1]
        assert target.labels.tolist() == [1]


@pytest.mark.required
class TestRefusals:
    """Addresses Indices cannot act on are refused when the operation or the view is built."""

    def test_keep_mode_takes_whole_items_only(self):
        with pytest.raises(ValueError, match="whole items"):
            Indices([SourceIndex(0, 1)])

    def test_a_frame_address_is_refused_with_the_way_to_leave_it_out(self):
        plan = RemovalPlan([SourceIndex(0, 3, "unit"), SourceIndex(0, 2)])
        with pytest.raises(ValueError, match="frame") as refusal:
            Indices(plan, exclude=True)
        assert 'RemovalPlan(a for a in plan if a.kind != "unit")' in str(refusal.value)
        assert Indices(RemovalPlan(a for a in plan if a.kind != "unit"), exclude=True).requires == "any_target"

    def test_track_minus_one_is_refused(self):
        with pytest.raises(ValueError, match="-1"):
            Indices([SourceIndex(0, -1, "track")], exclude=True)

    def test_a_track_needs_a_tracking_dataset(self, od_dataset):
        ds = od_dataset([[0, 1]], VOCABULARY)
        with pytest.raises(MaiteShapeError):
            View(ds, Indices([SourceIndex(0, 1, "track")], exclude=True))

    def test_a_classification_dataset_has_no_detections(self, ic_dataset):
        ds = ic_dataset([0, 1], VOCABULARY)
        with pytest.raises(ValueError, match="classification"):
            View(ds, Indices([SourceIndex(0, 0)], exclude=True))


@dataclass
class _Frame:
    frame_index: int
    pixels: np.ndarray


@dataclass
class _FrameTarget:
    labels: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    track_ids: np.ndarray


@dataclass
class _VideoTarget:
    frame_tracks: list[Any]


def _frame_target(detections: list[tuple[int, int]]) -> _FrameTarget:
    """One frame's detections, each a ``(label, track_id)`` pair."""
    n = len(detections)
    return _FrameTarget(
        labels=np.array([label for label, _ in detections], dtype=np.intp),
        boxes=np.tile(np.array([1.0, 1.0, 4.0, 4.0], dtype=np.float32), (n, 1)),
        scores=np.ones(n, dtype=np.float32),
        track_ids=np.array([track for _, track in detections], dtype=np.intp),
    )


class _Videos:
    """A tracking dataset whose frames hold the given detections."""

    def __init__(self, sequences: list[list[list[tuple[int, int]]]]) -> None:
        self._sequences = sequences
        self.metadata = {"id": "videos", "index2label": dict(VOCABULARY)}

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, index: int) -> tuple[Any, Any, DatumMetadata]:
        frames = self._sequences[index]
        stream = [_Frame(i, np.zeros((3, 8, 8), dtype=np.uint8)) for i in range(len(frames))]
        return stream, _VideoTarget([_frame_target(d) for d in frames]), {"id": f"v{index}"}


# Sequence 0: target_index 0 and 1 in frame 0, 2 in frame 1, 3 and 4 in frame 2. Track 2 spans frames 0 and 2.
VIDEOS = [
    [[(0, 1), (1, 2)], [(0, 1)], [(1, 2), (0, -1)]],
    [[(0, 7)], [(0, 7)]],
]


def _track_ids(view: View, item: int) -> list[list[int]]:
    return [frame.track_ids.tolist() for frame in view[item][1].frame_tracks]


@pytest.mark.required
class TestTrackingAddresses:
    """On a tracking dataset, detections and tracks are removed frame by frame, and every frame stays."""

    def test_a_detection_is_found_by_its_count_across_the_sequence(self):
        view = View(_Videos(VIDEOS), Indices([SourceIndex(0, 2)], exclude=True))
        assert _track_ids(view, 0) == [[1, 2], [], [2, -1]]

    def test_a_detection_in_a_later_frame(self):
        view = View(_Videos(VIDEOS), Indices([SourceIndex(0, 4)], exclude=True))
        assert _track_ids(view, 0) == [[1, 2], [1], [2]]

    def test_a_track_is_removed_from_every_frame(self):
        view = View(_Videos(VIDEOS), Indices([SourceIndex(0, 2, "track")], exclude=True))
        assert _track_ids(view, 0) == [[1], [1], [-1]]
        assert [frame.labels.tolist() for frame in view[0][1].frame_tracks] == [[0], [0], [0]]

    def test_every_frame_stays(self):
        view = View(_Videos(VIDEOS), Indices([SourceIndex(0, 2)], exclude=True))
        assert len(view[0][1].frame_tracks) == 3
        assert len(list(view[0][0])) == 3

    def test_other_sequences_are_left_alone(self):
        view = View(_Videos(VIDEOS), Indices([SourceIndex(0, 2, "track")], exclude=True))
        assert _track_ids(view, 1) == [[7], [7]]

    def test_a_key_the_sequence_does_not_hold_is_ignored(self):
        view = View(_Videos(VIDEOS), Indices([SourceIndex(1, 50), SourceIndex(1, 9, "track")], exclude=True))
        assert _track_ids(view, 1) == [[7], [7]]

    def test_metadata_through_the_view_renumbers_the_detections(self):
        md = Metadata(View(_Videos(VIDEOS), Indices([SourceIndex(0, 2, "track")], exclude=True)))
        rows = md.rows_at("instance").filter(pl.col("item_index") == 0)
        assert rows["target_index"].to_list() == [0, 1, 2]

    def test_the_count_runs_on_across_an_empty_frame(self):
        """target_index 0 is in frame 0, frame 1 holds none, and 1 and 2 are in frame 2."""
        videos = [[[(0, 1)], [], [(1, 2), (0, 3)]]]
        view = View(_Videos(videos), Indices([SourceIndex(0, 2)], exclude=True))
        assert _track_ids(view, 0) == [[1], [], [2]]

    def test_a_sequence_with_no_frames_stays_empty(self):
        view = View(_Videos([[[(0, 1)]], []]), Indices([SourceIndex(1, 0), SourceIndex(1, 1, "track")], exclude=True))
        assert view.selection == [0, 1]
        assert _track_ids(view, 1) == []
        assert _track_ids(view, 0) == [[1]]

    def test_the_view_pickles(self):
        view = View(_Videos(VIDEOS), Indices([SourceIndex(0, 2, "track"), SourceIndex(0, 0)], exclude=True))
        copy = pickle.loads(pickle.dumps(view))
        assert _track_ids(copy, 0) == _track_ids(view, 0) == [[], [1], [-1]]
