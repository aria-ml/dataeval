"""Tests for selecting frames from the annotations alone: track events, the best frame per track, labels held."""

import logging
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from dataeval.data import BestPerTrack, Chain, HasLabels, SequenceFrames, Stride, TrackEvents
from dataeval.data._selectors import sequence_infos
from tests.data.test_chain import _candidates_for, _KeepIf, _labeled
from tests.data.test_frames import emit, make_dataset_of


def _frame(tracks, labels=None, boxes=None, scores=None):
    """One frame's target. ``tracks=None`` omits ``track_ids``; ``scores=None`` omits ``scores``."""
    labels = [0] * len(tracks or []) if labels is None else labels
    target = SimpleNamespace(
        labels=np.asarray(labels, dtype=np.int64),
        boxes=np.asarray([[0, 0, 1, 1]] * len(labels) if boxes is None else boxes, dtype=np.float32),
    )
    if tracks is not None:
        target.track_ids = np.asarray(tracks, dtype=np.int64)
    if scores is not None:
        target.scores = np.asarray(scores, dtype=np.float32)
    return target


def _frames_of(tracks=None, labels=None, areas=None, scores=None):
    """One sequence's frame targets; ``areas`` become square boxes of that area, one list per frame."""
    n = len(next(x for x in (tracks, labels, areas, scores) if x is not None))
    out = []
    for i in range(n):
        frame_tracks = None if tracks is None else tracks[i]
        frame_labels = None if labels is None else labels[i]
        if frame_labels is None and frame_tracks is None and areas is not None:
            frame_labels = [0] * len(areas[i])
        side = None if areas is None else [float(a) ** 0.5 for a in areas[i]]
        boxes = None if side is None else [[0, 0, s, s] for s in side]
        out.append(_frame(frame_tracks, frame_labels, boxes, None if scores is None else scores[i]))
    return out


def _kept(selector, tracks=None, labels=None, areas=None, scores=None):
    dataset = make_dataset_of([_frames_of(tracks, labels, areas, scores)])
    return SequenceFrames(dataset, selector).frame_map[:, 1].tolist()


def _weights(selector, tracks=None, labels=None, areas=None, scores=None):
    dataset = make_dataset_of([_frames_of(tracks, labels, areas, scores)])
    return [meta["frames_represented"] for _, _, meta in emit(SequenceFrames(dataset, selector))]


_EVENTS = [[1], [1], [1, 2], [1, 2], [2], [2]]


def _agree(selector, dataset):
    info = next(iter(sequence_infos(dataset)))[0]
    planned = selector.plan(info)
    assert planned is not None
    assert planned.tolist() == [v.position for v in selector.select(iter(_candidates_for(dataset, 0)))]


@pytest.mark.required
class TestTrackEvents:
    def test_keeps_the_first_frame_and_each_change_of_tracks(self):
        assert _kept(TrackEvents(), _EVENTS) == [0, 2, 4]

    def test_weights_run_to_the_next_event(self):
        assert _weights(TrackEvents(), _EVENTS) == [2.0, 2.0, 2.0]

    def test_untracked_detections_are_refused_by_default(self):
        with pytest.raises(ValueError, match="from sequence 0 hold 1 detection"):
            _kept(TrackEvents(), [[1], [1, -1]])

    def test_the_refusal_counts_the_sequences_untracked_detections(self):
        with pytest.raises(ValueError, match=r"from sequence 0 hold 2 detection\(s\).*first at position 1 of them"):
            _kept(TrackEvents(), [[1], [1, -1], [-1]])

    def test_streaming_refuses_untracked_detections_too(self):
        dataset = make_dataset_of([_frames_of([[1], [1, -1]])])
        with pytest.raises(
            ValueError, match="position 1 of the frames it was offered from sequence 0 holds 1 detection"
        ):
            list(TrackEvents().select(iter(_candidates_for(dataset, 0))))

    def test_inside_a_chain_the_refusal_names_a_position_among_the_frames_offered(self):
        """Source frame 6 is the fourth frame Stride(2) passes on, so position 3, which is not frame 3."""
        tracks = [[1]] * 8
        tracks[6] = [1, -1]
        with pytest.raises(ValueError, match="first at position 3 of them"):
            SequenceFrames(make_dataset_of([_frames_of(tracks)]), Chain(Stride(2), TrackEvents()))

    def test_ignore_leaves_untracked_out(self):
        assert _kept(TrackEvents(on_missing_tracks="ignore"), [[1], [1, -1], [1]]) == [0]

    def test_label_counts_compare_untracked_detections_by_label(self):
        kept = _kept(TrackEvents(on_missing_tracks="label_counts"), [[-1], [-1], [-1]], labels=[[0], [0], [3]])
        assert kept == [0, 2]

    def test_label_counts_see_a_change_in_how_many(self):
        selector = TrackEvents(on_missing_tracks="label_counts")
        kept = _kept(selector, [[-1], [-1, -1], [-1, -1]], labels=[[0], [0, 0], [0, 0]])
        assert kept == [0, 1]

    def test_plan_and_select_agree(self):
        _agree(TrackEvents(), make_dataset_of([_frames_of(_EVENTS)]))

    def test_short_track_ids_count_as_untracked(self):
        with pytest.raises(ValueError, match="from sequence 0 hold 1 detection"):
            _kept(TrackEvents(), [[1], [1]], labels=[[0], [0, 0]])

    def test_a_missing_track_ids_counts_as_untracked(self):
        with pytest.raises(ValueError, match="from sequence 0 hold 1 detection"):
            _kept(TrackEvents(), [None], labels=[[0]])

    def test_a_target_less_info_falls_back_to_streaming(self):
        info = next(iter(sequence_infos(make_dataset_of([_frames_of(_EVENTS)]))))[0]
        assert TrackEvents().plan(replace(info, target=None)) is None

    def test_rejects_an_unknown_policy(self):
        with pytest.raises(ValueError, match="on_missing_tracks"):
            TrackEvents(on_missing_tracks="drop")  # type: ignore[arg-type]


@pytest.mark.required
class TestBestPerTrack:
    def test_keeps_each_tracks_largest_box_ties_earliest(self):
        assert _kept(BestPerTrack(), tracks=[[1], [1, 2], [1, 2]], areas=[[4], [9, 1], [9, 16]]) == [1, 2]

    def test_by_score(self):
        scores = [[0.2, 0.9], [0.8, 0.1], [0.8, 0.3]]
        assert _kept(BestPerTrack(by="score"), tracks=[[1, 2]] * 3, scores=scores) == [0, 1]

    def test_each_kept_frame_stands_for_itself(self):
        weights = _weights(BestPerTrack(), tracks=[[1], [1, 2], [1, 2]], areas=[[4], [9, 1], [9, 16]])
        assert weights == [1.0, 1.0]

    def test_scores_absent_are_refused(self):
        with pytest.raises(ValueError, match="sequence 0.*not finite"):
            _kept(BestPerTrack(by="score"), tracks=[[1]], scores=None)

    def test_a_track_whose_scores_are_all_nan_is_refused_though_another_track_scores(self):
        with pytest.raises(ValueError, match="track 2"):
            _kept(BestPerTrack(by="score"), tracks=[[1, 2], [2]], scores=[[0.9, np.nan], [np.nan]])

    def test_one_nan_candidate_in_an_otherwise_scored_track_is_refused(self):
        with pytest.raises(ValueError, match="track 1"):
            _kept(BestPerTrack(by="score"), tracks=[[1], [1]], scores=[[0.9], [np.nan]])
        with pytest.raises(ValueError, match="position 1 of the frames it was offered gives nan"):
            _kept(BestPerTrack(by="score"), tracks=[[1], [1]], scores=[[0.9], [np.nan]])

    def test_a_non_finite_area_is_refused(self):
        frames = [_frame([1], [0], boxes=[[0, 0, np.nan, 1]])]
        with pytest.raises(ValueError, match="track 1"):
            len(SequenceFrames(make_dataset_of([frames]), BestPerTrack()))

    def test_a_nan_on_an_untracked_detection_is_no_error_under_ignore(self):
        selector = BestPerTrack(by="score", on_missing_tracks="ignore")
        assert _kept(selector, tracks=[[1, -1], [1, -1]], scores=[[0.5, np.nan], [0.7, np.nan]]) == [1]

    def test_scores_in_the_class_distribution_layout_rank_by_own_class(self):
        scores = [[[0.9, 0.2]], [[0.1, 0.5]]]
        assert _kept(BestPerTrack(by="score"), tracks=[[1], [1]], labels=[[1], [1]], scores=scores) == [1]

    def test_missing_or_short_track_ids_count_as_untracked(self):
        with pytest.raises(ValueError, match="from sequence 0 hold 2 detection"):
            _kept(BestPerTrack(), tracks=[None, [1]], labels=[[0], [0, 0]])
        assert _kept(BestPerTrack(on_missing_tracks="ignore"), tracks=[None, [1]], labels=[[0], [0, 0]]) == [1]

    def test_a_track_id_repeated_across_videos_is_two_tracks(self):
        dataset = make_dataset_of([
            _frames_of([[1], [1]], areas=[[1], [2]]),
            _frames_of([[1], [1], [1]], areas=[[5], [4], [3]]),
        ])
        assert SequenceFrames(dataset, BestPerTrack()).frame_map.tolist() == [[0, 1], [1, 0]]

    def test_a_sequence_without_tracks_contributes_nothing(self, caplog):
        with caplog.at_level(logging.INFO):
            assert _kept(BestPerTrack(on_missing_tracks="ignore"), tracks=[[-1], [-1]]) == []
        assert "sequence 0" in caplog.text

    def test_plan_and_select_agree(self):
        dataset = make_dataset_of([_frames_of([[1], [1, 2], [1, 2]], areas=[[4], [9, 1], [9, 16]])])
        _agree(BestPerTrack(), dataset)

    def test_plan_refuses_a_missing_target(self):
        info = next(iter(sequence_infos(make_dataset_of([_frames_of(_EVENTS)]))))[0]
        bare = replace(info, target=None)
        with pytest.raises(ValueError, match="target"):
            BestPerTrack().plan(bare)

    def test_in_a_chain_it_ranks_the_survivors(self):
        frames = _frames_of([[1]] * 5, areas=[[1], [9], [2], [3], [4]])
        kept = SequenceFrames(make_dataset_of([frames]), Chain(Stride(2), BestPerTrack())).frame_map[:, 1].tolist()
        assert kept == [4]

    def test_streaming_refuses_untracked_detections_too(self):
        dataset = make_dataset_of([_frames_of([[1], [-1], [-1]], areas=[[1], [2], [3]])])
        with pytest.raises(ValueError, match="from sequence 0 hold 2 detection"):
            list(BestPerTrack().select(iter(_candidates_for(dataset, 0))))

    def test_rejects_unknown_choices(self):
        with pytest.raises(ValueError, match="by"):
            BestPerTrack(by="size")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="on_missing_tracks"):
            BestPerTrack(on_missing_tracks="label_counts")  # type: ignore[arg-type]


@pytest.mark.required
class TestHasLabels:
    def test_keeps_frames_with_enough_of_the_labels(self):
        assert _kept(HasLabels([7], min_count=2), labels=[[7], [7, 7], [3, 7, 7], [3]]) == [1, 2]

    def test_excludes_the_rest(self):
        assert _weights(HasLabels([7]), labels=[[7], [3], [7, 3], [3]]) == [1.0, 1.0]

    def test_a_label_that_never_occurs_is_no_error(self):
        assert _kept(HasLabels([7, 99]), labels=[[7], [3]]) == [0]

    def test_refusals(self):
        with pytest.raises(ValueError, match="labels"):
            HasLabels([])
        with pytest.raises(ValueError, match="min_count"):
            HasLabels([7], min_count=0)

    def test_in_a_chain_reads_the_survivors_labels(self):
        dataset = _labeled(12, labeled_at=(5, 7), label=7)
        assert SequenceFrames(dataset, Chain(Stride(5), HasLabels([7]))).frame_map[:, 1].tolist() == [5]

    def test_plan_and_select_agree(self):
        _agree(HasLabels([7], min_count=2), make_dataset_of([_frames_of(labels=[[7], [7, 7], [3, 7, 7], [3]])]))

    def test_after_a_streaming_stage_it_reads_the_survivors_labels(self):
        # 7 sits on frames 1 and 3; the streaming stage drops frame 1 and keeps frame 3
        dataset = _labeled(6, labeled_at=(1, 3), label=7)
        stages = Chain(_KeepIf(lambda i: i >= 2), HasLabels([7]))
        assert SequenceFrames(dataset, stages).frame_map[:, 1].tolist() == [3]

    def test_a_target_less_info_falls_back_to_streaming(self):
        info = next(iter(sequence_infos(make_dataset_of([_frames_of(labels=[[7]])]))))[0]
        assert HasLabels([7]).plan(replace(info, target=None)) is None


@pytest.mark.required
@pytest.mark.parametrize("selector", [TrackEvents(), BestPerTrack(), HasLabels([0])], ids=lambda s: type(s).__name__)
def test_sizing_a_view_decodes_nothing(selector):
    dataset = make_dataset_of([_frames_of([[1], [1, 2], [2]], areas=[[1], [2, 3], [4]])])
    assert len(SequenceFrames(dataset, selector)) > 0
    stream: Any = dataset[0][0]
    assert stream._counter["iterations"] == 0
