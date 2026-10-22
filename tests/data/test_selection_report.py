"""Tests for the account of what each stage of a frame selection did, per video and per class."""

from collections import Counter

import polars as pl
import pytest

from dataeval.data import (
    AllFrames,
    BestPerTrack,
    Chain,
    FrameInput,
    FrameRate,
    FrameSelector,
    FrameVerdict,
    HasLabels,
    Redundancy,
    SequenceFrames,
    Stride,
)
from tests.data.test_chain import _KeepIf, _KeepTopTwo, _labeled
from tests.data.test_chain import _KeepsEachTwice as _Twice
from tests.data.test_frames import _EveryOther, emit, make_dataset


class _KeepOrdinals(FrameSelector):
    """Yields a verdict for each listed ordinal of its input as it arrives, so one listed twice is kept twice."""

    needs: FrameInput = FrameInput.STRUCTURE

    def __init__(self, *ordinals: int):
        self.ordinals = ordinals

    def select(self, frames):
        for frame in frames:
            for _ in range(self.ordinals.count(frame.position)):
                yield FrameVerdict(frame.position)


class _ReturnsZero(FrameSelector):
    """Keeps every frame, then returns a value from its select generator, which only a Chain's is read for."""

    needs: FrameInput = FrameInput.STRUCTURE

    def select(self, frames):
        yield from (FrameVerdict(frame.position) for frame in frames)
        return 0  # noqa: B901


def _assert_source_frames_balance(frames: SequenceFrames) -> None:
    """Per sequence, the stages' source frames excluded sum to its frame count less what its kept frames represent."""
    stages = frames.selection_report().stages
    represented: Counter[int] = Counter()
    for _, _, metadata in emit(frames):
        represented[metadata["sequence"]] += metadata["frames_represented"]
    n_frames = dict(stages.filter(pl.col("stage") == 0).select("sequence", "offered").rows())
    excluded = dict(stages.group_by("sequence").agg(pl.col("source_frames_excluded").sum()).rows())
    for sequence, count in n_frames.items():
        assert excluded[sequence] == pytest.approx(count - represented[sequence])


class _MisreturningChain(Chain):
    """A Chain whose select returns neither a continuation nor a trace: the bug the guard is there for."""

    def select(self, frames):
        yield from super().select(frames)
        return 0  # noqa: B901


@pytest.mark.required
class TestSelectionReport:
    def test_one_thinning_selector(self):
        dataset, _ = make_dataset((6, 4))
        stages = SequenceFrames(dataset, Stride(2)).selection_report().stages
        assert stages.select(
            "sequence", "stage", "offered", "kept", "represented", "excluded", "unmeasured"
        ).rows() == [
            (0, 0, 6, 3, 3, 0, 0),
            (1, 0, 4, 2, 2, 0, 0),
        ]

    def test_the_columns_and_their_types(self):
        dataset, _ = make_dataset((6, 4))
        report = SequenceFrames(dataset, Stride(2)).selection_report()
        assert report.stages.schema == {
            "sequence": pl.Int64,
            "source_id": pl.String,
            "stage": pl.Int64,
            "selector": pl.String,
            "offered": pl.Int64,
            "kept": pl.Int64,
            "represented": pl.Int64,
            "excluded": pl.Int64,
            "unmeasured": pl.Int64,
            "source_frames_excluded": pl.Float64,
        }
        assert report.labels.schema == {
            "sequence": pl.Int64,
            "source_id": pl.String,
            "stage": pl.Int64,
            "selector": pl.String,
            "label": pl.Int64,
            "frames_offered": pl.Int64,
            "frames_kept": pl.Int64,
            "detections_offered": pl.Int64,
            "detections_kept": pl.Int64,
        }
        assert report.stages.select("source_id", "selector").row(0) == ("vid0", repr(Stride(2)))

    def test_a_filter_excludes(self):
        dataset = _labeled(6, labeled_at=(1, 4), label=7)
        stages = SequenceFrames(dataset, HasLabels([7])).selection_report().stages
        assert stages.select("offered", "kept", "represented", "excluded").row(0) == (6, 2, 0, 4)

    def test_a_chain_stage_reports_the_source_frames_it_excluded(self):
        """Its own ``excluded`` counts its input, where each frame Stride(2) kept stands for 2 source frames."""
        # 7 sits on source frames 2 and 6, both of which Stride(2) keeps: HasLabels keeps 2 of its 6 inputs
        dataset = _labeled(12, labeled_at=(2, 6), label=7)
        frames = SequenceFrames(dataset, Chain(Stride(2), HasLabels([7])))
        stages = frames.selection_report().stages
        assert stages.schema["source_frames_excluded"] == pl.Float64
        assert stages.select("stage", "excluded", "source_frames_excluded").rows() == [(0, 0, 0.0), (1, 4, 8.0)]
        assert sum(metadata["frames_represented"] for _, _, metadata in emit(frames)) == 4.0

    @pytest.mark.parametrize(
        "selector",
        [
            lambda: _EveryOther(),
            lambda: Stride(2),
            lambda: HasLabels([7]),
            lambda: Chain(_EveryOther(), _KeepIf(lambda i: i % 4 == 0, excludes=True)),
            lambda: Chain(_KeepTopTwo(), _EveryOther()),
            lambda: Chain(_EveryOther(), HasLabels([7])),
            lambda: Chain(_Twice(), _KeepOrdinals(0, 0, 2)),
        ],
        ids=[
            "a lone thinning selector",
            "a lone planning thinner",
            "a lone filter",
            "a streaming chain",
            "a two_pass chain",
            "a chain planning behind a streaming stage",
            "a chain keeping frames twice",
        ],
    )
    def test_the_source_frames_excluded_balance_what_the_kept_frames_represent(self, selector):
        _assert_source_frames_balance(SequenceFrames(_labeled(12, labeled_at=(2, 6), label=7), selector()))

    def test_a_planned_view_reports_its_source_frames_excluded_without_decoding(self):
        dataset, counters = make_dataset((6, 4))
        # Stride(2) keeps 3 and 2 frames, each standing for 2; BestPerTrack keeps only the first of them
        frames = SequenceFrames(dataset, Chain(Stride(2), BestPerTrack()))
        assert frames.selection_report().stages["source_frames_excluded"].to_list() == [0.0, 4.0, 0.0, 2.0]
        assert [c["iterations"] for c in counters] == [0, 0]
        _assert_source_frames_balance(frames)

    def test_unmeasured_is_counted(self):
        dataset, _ = make_dataset((4,), timed=False)
        assert SequenceFrames(dataset, FrameRate(10.0)).selection_report().stages["unmeasured"].to_list() == [4]

    def test_each_stage_of_a_chain_has_its_own_row(self):
        dataset, _ = make_dataset((12,))
        stages = SequenceFrames(dataset, Chain(Stride(2), Stride(3))).selection_report().stages
        assert stages.select("stage", "offered", "kept").rows() == [(0, 12, 6), (1, 6, 2)]

    def test_a_streaming_and_a_two_pass_chain_report_their_stages(self):
        dataset, _ = make_dataset((6,))
        # Each frame is flat, at its own gray level, and every flat image has the same phash, so
        # Redundancy keeps only the first it is offered. _KeepTopTwo keeps positions 3 and 4.
        for chain, expected, kept in (
            (Chain(_EveryOther(), Redundancy()), [(0, 6, 3), (1, 3, 1)], [0]),
            (Chain(_KeepTopTwo(), _EveryOther()), [(0, 6, 2), (1, 2, 1)], [3]),
        ):
            frames = SequenceFrames(dataset, chain)
            list(frames.stream())
            stages = frames.selection_report().stages
            assert stages.select("stage", "offered", "kept").rows() == expected
            assert stages["kept"].to_list()[-1] == len(frames.frame_map)
            presented = frames.frame_map[frames.frame_map[:, 0] == 0, 1].tolist()
            assert frames._record_of(0).trace[-1].positions.tolist() == presented == kept

    def test_per_class_survival(self):
        """The design's example: most frames kept, but only some of class 7's."""
        dataset = _labeled(10, labeled_at=(1, 3, 5, 7, 9), label=7)
        labels = SequenceFrames(dataset, Stride(2)).selection_report().labels
        seven = labels.filter(pl.col("label") == 7).row(0, named=True)
        assert (seven["frames_offered"], seven["frames_kept"]) == (5, 0)
        assert (seven["detections_offered"], seven["detections_kept"]) == (5, 0)
        zero = labels.filter(pl.col("label") == 0).row(0, named=True)
        assert (zero["frames_offered"], zero["frames_kept"]) == (5, 5)
        assert (zero["detections_offered"], zero["detections_kept"]) == (5, 5)

    def test_a_planned_view_reports_without_decoding(self):  # Review Focus 3
        dataset, counters = make_dataset((6, 4))
        SequenceFrames(dataset, Chain(Stride(2), Stride(2))).selection_report()
        assert [c["iterations"] for c in counters] == [0, 0]

    def test_reporting_after_streaming_walks_nothing(self):
        dataset, counters = make_dataset((6, 4))
        frames = SequenceFrames(dataset, _EveryOther())
        list(frames.stream())
        walked = [c["iterations"] for c in counters]
        frames.selection_report()
        assert [c["iterations"] for c in counters] == walked

    def test_a_label_nobody_carries_keeps_nothing_and_says_so(self):  # Review Focus 5
        dataset = _labeled(6, labeled_at=(), label=7)
        stages = SequenceFrames(dataset, HasLabels([7])).selection_report().stages
        assert stages.select("offered", "kept", "excluded").row(0) == (6, 0, 6)

    def test_a_zero_frame_sequence(self):
        dataset, _ = make_dataset((0, 4))
        report = SequenceFrames(dataset, Stride(2)).selection_report()
        assert report.stages.filter(pl.col("sequence") == 0).select("offered", "kept").row(0) == (0, 0)
        assert report.labels.filter(pl.col("sequence") == 0).is_empty()

    def test_a_zero_frame_sequence_under_a_streaming_chain_has_every_stage(self):
        dataset, _ = make_dataset((0, 4))
        stages = SequenceFrames(dataset, Chain(_EveryOther(), Stride(2))).selection_report().stages
        assert stages.filter(pl.col("sequence") == 0).select("stage", "offered", "kept").rows() == [
            (0, 0, 0),
            (1, 0, 0),
        ]

    # Duplicate keeps: `_Twice` is a streaming test selector that keeps every frame twice (declared weight 0.5 each).
    def test_a_stage_that_keeps_both_copies_drops_nothing(self):
        dataset, _ = make_dataset((3,))
        stages = SequenceFrames(dataset, Chain(_Twice(), AllFrames())).selection_report().stages
        assert stages.select("stage", "offered", "kept", "represented", "excluded").rows() == [
            (0, 3, 6, 0, 0),
            (1, 6, 6, 0, 0),
        ]

    def test_a_stage_that_keeps_one_copy_of_each_drops_the_other(self):
        dataset, _ = make_dataset((3,))
        stages = SequenceFrames(dataset, Chain(_Twice(), _EveryOther())).selection_report().stages
        assert stages.filter(pl.col("stage") == 1).select("offered", "kept", "represented").row(0) == (6, 3, 3)

    def test_a_stage_that_keeps_one_input_twice_and_drops_another(self):
        # stage 1 offered [f0, f0, f1, f1] (ordinals 0..3); keeps ordinal 0 twice and ordinal 2 once:
        # 3 verdicts, 2 distinct
        dataset, _ = make_dataset((2,))
        stages = SequenceFrames(dataset, Chain(_Twice(), _KeepOrdinals(0, 0, 2))).selection_report().stages
        assert stages.filter(pl.col("stage") == 1).select("offered", "kept", "represented").row(0) == (4, 3, 2)

    def test_a_frame_kept_twice_counts_twice_per_class(self):
        dataset = _labeled(2, labeled_at=(1,), label=7)
        labels = SequenceFrames(dataset, Chain(_Twice(), _KeepOrdinals(0, 0, 2))).selection_report().labels
        stage1 = labels.filter(pl.col("stage") == 1).sort("label")
        assert stage1.select("label", "frames_offered", "frames_kept", "detections_kept").rows() == [
            (0, 2, 2, 2),
            (7, 2, 1, 1),
        ]

    @pytest.mark.parametrize(
        ("chain", "n_frames"),
        [
            (lambda: Chain(FrameRate(10.0), AllFrames()), 4),
            (lambda: Chain(FrameRate(10.0), _EveryOther()), 4),
            (lambda: Chain(FrameRate(10.0), _KeepTopTwo()), 6),
        ],
        ids=["a planning boundary", "a streaming stage", "a two_pass stage replayed"],
    )
    def test_unmeasured_is_stage_local_in_the_report_and_cumulative_on_the_frame(self, chain, n_frames):
        dataset, _ = make_dataset((n_frames,), timed=False)
        frames = SequenceFrames(dataset, chain())
        stages = frames.selection_report().stages
        assert stages.select("stage", "unmeasured").rows() == [(0, n_frames), (1, 0)]
        assert frames.unmeasured.tolist() == [True] * len(frames.frame_map)

    def test_a_lone_selectors_return_value_is_ignored(self):
        """Only a Chain returns anything SequenceFrames reads; a lone selector's report comes from its entries."""
        dataset, _ = make_dataset((3,))
        frames = SequenceFrames(dataset, _ReturnsZero())
        assert len(list(frames.stream())) == 3
        assert frames.selection_report().stages.select("offered", "kept", "represented").row(0) == (3, 3, 0)

    def test_a_chain_returning_neither_a_continuation_nor_a_trace_is_refused(self):
        dataset, _ = make_dataset((3,))
        with pytest.raises(TypeError, match="neither a continuation nor a trace"):
            len(SequenceFrames(dataset, _MisreturningChain(_EveryOther())))

    @pytest.mark.parametrize(
        "chain",
        [
            lambda: Chain(_KeepIf(lambda i: False), _EveryOther()),
            lambda: Chain(_KeepTopTwo(), _KeepIf(lambda i: False)),
        ],
    )
    def test_an_empty_streaming_or_two_pass_chain_reports_every_stage(self, chain):
        dataset, _ = make_dataset((5,))
        stages = SequenceFrames(dataset, chain()).selection_report().stages
        assert len(stages) == 2
        assert stages["kept"].to_list()[-1] == 0
