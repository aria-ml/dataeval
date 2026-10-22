"""Tests for composing frame selectors, each stage choosing among the frames the stage before it kept."""

import gc
import weakref
from collections import Counter
from typing import Any, cast

import numpy as np
import pytest

from dataeval.config import use_seed
from dataeval.data import (
    AllFrames,
    Chain,
    EvenlySpaced,
    FrameCandidate,
    FrameIndices,
    FrameInput,
    FrameSelector,
    FrameVerdict,
    Redundancy,
    Representative,
    SequenceFrames,
    SequenceInfo,
    Stride,
)
from dataeval.data._selectors import _projected, sequence_infos
from tests.data.test_frames import (
    _Buffering,
    _Declares,
    _EveryOther,
    _FirstTwo,
    _LiesAboutPixels,
    _Overclaims,
    _PixelNovelty,
    emit,
    make_dataset,
    make_dataset_of,
    make_target,
)


class _LabelIn(FrameSelector):
    """A filter keeping the frames that show ``label``, planned from the targets -- MR 3's ``HasLabels``."""

    needs: FrameInput = FrameInput.STRUCTURE
    excludes = True

    def __init__(self, label: int):
        self.label = label

    def plan(self, info):
        assert info.target is not None
        tracks = info.target.frame_tracks
        return np.array([i for i in range(info.n_frames) if self.label in np.asarray(tracks[i].labels)], dtype=np.intp)

    def select(self, frames):
        return (FrameVerdict(frame.position) for frame in frames if self.label in np.asarray(frame.target.labels))


class _Adjacent(FrameSelector):
    """Reads the frames on either side of each one -- the shape a camera-motion measure has."""

    needs: FrameInput = FrameInput.STRUCTURE
    _needs_adjacent_frames = True

    def select(self, frames):
        return (FrameVerdict(frame.position) for frame in frames)


class _Disordered(FrameSelector):
    """Plans weighted positions out of order, which SequenceFrames cannot normalize without misaligning."""

    needs: FrameInput = FrameInput.STRUCTURE

    def _plan_weighted(self, info):
        return np.array([4, 0], dtype=np.intp), np.array([2.0, 4.0])

    def select(self, frames):
        return (FrameVerdict(frame.position) for frame in frames if frame.position in (0, 4))


class _KeepIf(FrameSelector):
    """A streaming filter on each frame's source number, excluding what it drops or, by default, representing it."""

    needs: FrameInput = FrameInput.STRUCTURE

    def __init__(self, predicate, excludes: bool = False):
        self.predicate = predicate
        self.excludes = excludes

    def select(self, frames):
        return (FrameVerdict(frame.position) for frame in frames if self.predicate(frame.frame_index))


class _Tags(FrameSelector):
    """Keeps every frame, recording its position in this stage's input as factor ``name``."""

    needs: FrameInput = FrameInput.STRUCTURE

    def __init__(self, name: str):
        self.name = name

    def select(self, frames):
        return (FrameVerdict(frame.position, factors={self.name: frame.position}) for frame in frames)


class _ReadsWeight(FrameSelector):
    """Decides from each candidate's weight as it arrives, which a streaming stage must not."""

    needs: FrameInput = FrameInput.STRUCTURE

    def select(self, frames):
        return (FrameVerdict(frame.position) for frame in frames if float(frame.weight) > 0)


class _Watch(FrameSelector):
    """Decodes every frame it is offered and keeps every 5th, noting a weak reference to each frame handle."""

    needs: FrameInput = FrameInput.PIXELS

    def __init__(self, handles: dict[int, Any]):
        self.handles = handles

    def select(self, frames):
        for frame in frames:
            assert frame.pixels.size
            self.handles[frame.frame_index] = weakref.ref(frame._frame)
            if frame.position % 5 == 0:
                yield FrameVerdict(frame.position)


class _PlansNothing(FrameSelector):
    """Overrides plan but defers every sequence to select, as the FrameSelector contract allows."""

    needs: FrameInput = FrameInput.STRUCTURE

    def plan(self, info):
        return None

    def select(self, frames):
        return (FrameVerdict(frame.position) for frame in frames)


class _KeepTopTwo(FrameSelector):
    """Keeps the two frames with the largest ``frame_index % 5``, ties to the earlier -- MR 3's quantile QualityGate."""

    needs: FrameInput = FrameInput.STRUCTURE
    two_pass = True
    excludes = True

    def select(self, frames):
        ranked = sorted((-(frame.frame_index % 5), frame.position) for frame in frames)
        return (FrameVerdict(position) for position in sorted(position for _, position in ranked[:2]))


class _Totals(FrameSelector):
    """Keeps the first frame, declaring for it ``scale`` times the summed weight of every frame it was offered."""

    needs: FrameInput = FrameInput.STRUCTURE
    two_pass = True

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def select(self, frames):
        total = sum(frame.weight for frame in frames)  # arithmetic, so a pending weight would raise
        return iter([FrameVerdict(0, weight=total * self.scale)])


class _WatchAtOnce(FrameSelector):
    """Decodes every frame it is offered, noting a weak reference to each frame handle and which outlive its input."""

    needs: FrameInput = FrameInput.PIXELS
    two_pass = True

    def __init__(self):
        self.handles: dict[int, Any] = {}
        self.alive: list[int] = []

    def select(self, frames):
        for frame in frames:
            assert frame.pixels.size
            self.handles[frame.frame_index] = weakref.ref(frame._frame)
        frame = None  # the last one offered
        gc.collect()
        self.alive = [i for i, ref in sorted(self.handles.items()) if ref() is not None]
        return iter([FrameVerdict(0)])


class _FirstTwoAtOnce(_FirstTwo):
    """_FirstTwo, declaring two_pass."""

    two_pass = True


class _KeepsEachTwice(FrameSelector):
    """Keeps every frame twice, each keep at half the frame's mass and numbered in factor ``copy``."""

    needs: FrameInput = FrameInput.STRUCTURE

    def select(self, frames):
        for frame in frames:
            yield FrameVerdict(frame.position, weight=0.5, factors={"copy": 0})
            yield FrameVerdict(frame.position, weight=0.5, factors={"copy": 1})


class _FirstAndLast(FrameSelector):
    """Keeps its first frame, and its last once its input ends -- legal alone, where the walk still holds it."""

    needs: FrameInput = FrameInput.STRUCTURE

    def select(self, frames):
        last = None
        for frame in frames:
            if last is None:
                yield FrameVerdict(frame.position)
            last = frame
        if last is not None and last.position != 0:
            yield FrameVerdict(last.position)


def _labeled(n: int, labeled_at: tuple[int, ...], label: int):
    """One sequence of ``n`` frames, each with one detection: of ``label`` at ``labeled_at``, of 0 elsewhere."""
    targets = [make_target(1) for _ in range(n)]
    for position in labeled_at:
        cast(Any, targets[position]).labels = np.array([label], dtype=np.int64)
    return make_dataset_of([targets])


def _candidates_for(dataset, seq: int) -> list[FrameCandidate]:
    """Plain candidates for every frame of sequence ``seq``, built from its stream without SequenceFrames."""
    info = list(sequence_infos(dataset))[seq][0]
    stream, target, _ = dataset[seq]
    return [
        FrameCandidate(info, position, frame.frame_index, frame.time_s, frame.pts, target.frame_tracks[position])
        for position, frame in enumerate(stream)
    ]


def _random_stages(rng: np.random.Generator) -> list[FrameSelector]:
    """One to four stages, each of a kind drawn uniformly from the pool, then its parameters."""
    pool = [
        lambda: Stride(int(rng.integers(1, 5))),
        lambda: Stride(int(rng.integers(1, 5)), jitter=True, seed=int(rng.integers(100))),
        lambda: EvenlySpaced(int(rng.integers(1, 6))),
        lambda: EvenlySpaced(int(rng.integers(1, 6)), jitter=True, seed=int(rng.integers(100))),
        _EveryOther,
        lambda: _KeepIf(lambda i: i % 3 != 0),
        lambda: _KeepIf(lambda i: i % 3 != 0, excludes=True),
        lambda: _LabelIn(7),
        _KeepTopTwo,
    ]
    return [pool[int(rng.integers(len(pool)))]() for _ in range(int(rng.integers(1, 5)))]


def _reference(stages: list[FrameSelector], n_frames: int, target: Any) -> tuple[list[int], list[float]]:
    """What a chain of ``stages`` keeps of one sequence, and the weight of each, worked out list by list.

    Each stage sees the source frames the one before kept, renumbered from 0. A stage that plans
    is applied through ``plan`` over their projected targets, any other through ``select`` over
    plain candidates. Each source frame then follows its fate through every stage -- kept, run
    into the kept frame its mass goes to, or excluded -- and a kept frame's weight is the count of
    source frames that end with it. No stage in the pool declares a weight.
    """
    survivors = list(range(n_frames))  # source positions
    owner = {frame: frame for frame in survivors}  # each source frame not excluded -> the survivor standing for it
    for stage in stages:
        # Index 0 and id "vid0" are what make_dataset_of gives its first sequence; a jittered stage seeds from them.
        positions = np.array(survivors, dtype=np.intp)
        info = SequenceInfo(0, "vid0", len(survivors), cast(Any, {}), _projected(target, positions))
        planned = stage.plan(info)
        if planned is not None:
            kept = planned.tolist()
        else:
            mass = Counter(owner.values())
            offered = [
                FrameCandidate(info, i, s, None, None, target.frame_tracks[s], weight=float(mass[s]))
                for i, s in enumerate(survivors)
            ]
            verdicts = list(stage.select(iter(offered)))
            assert all(verdict.weight is None for verdict in verdicts)
            kept = [verdict.position for verdict in verdicts]
        fate: dict[int, int] = {}
        for i, s in enumerate(survivors):
            if i in kept:
                fate[s] = s
            elif kept and not stage.excludes:
                # Runs into the last kept frame before it, or the first kept frame when none is before it.
                fate[s] = survivors[max((k for k in kept if k < i), default=kept[0])]
        owner = {frame: fate[s] for frame, s in owner.items() if s in fate}
        survivors = [survivors[k] for k in kept]
    mass = Counter(owner.values())
    return survivors, [float(mass[s]) for s in survivors]


@pytest.mark.required
class TestPlanningChains:
    def test_stages_compose_over_survivors(self):
        dataset, _ = make_dataset((12,))
        frames = SequenceFrames(dataset, Chain(Stride(2), Stride(3)))
        assert frames.frame_map[:, 1].tolist() == [0, 6]
        assert [m[2]["frames_represented"] for m in emit(frames)] == [6.0, 6.0]

    def test_a_later_stage_reads_the_targets_of_the_frames_that_survived(self):
        """Labels sit on frames 5 and 7; Stride(5) keeps 0, 5, 10, so only frame 5 carries one."""
        dataset = _labeled(12, labeled_at=(5, 7), label=7)
        frames = SequenceFrames(dataset, Chain(Stride(5), _LabelIn(7)))
        assert frames.frame_map[:, 1].tolist() == [5]
        assert [m[2]["frames_represented"] for m in emit(frames)] == [5.0]  # its own Stride run, the rest excluded

    def test_a_planning_chain_is_free_to_size(self):
        dataset, counters = make_dataset((12, 6))
        assert len(SequenceFrames(dataset, Chain(Stride(2), EvenlySpaced(2)))) == 4
        assert [c["iterations"] for c in counters] == [0, 0]

    def test_one_stage_is_that_stage(self):
        """Review Focus 5."""
        dataset, _ = make_dataset((9, 4))
        alone = SequenceFrames(dataset, Stride(3))
        chained = SequenceFrames(dataset, Chain(Stride(3)))
        assert chained.frame_map.tolist() == alone.frame_map.tolist()
        assert [m[2]["frames_represented"] for m in emit(chained)] == [m[2]["frames_represented"] for m in emit(alone)]

    def test_a_zero_frame_sequence_plans_to_nothing(self):
        """Review Focus 2."""
        dataset, _ = make_dataset((0, 4))
        chain = Chain(Stride(2), Stride(2))
        assert len(SequenceFrames(dataset, chain)) == 1
        empty = next(sequence_infos(dataset))[0]
        weighted = chain._plan_weighted(empty)
        assert weighted is not None
        assert [part.tolist() for part in weighted] == [[], []]

    def test_a_chain_never_reads_labels_it_does_not_plan_from(self):
        """Review Focus 3: the second sequence's frame targets carry no ``labels`` at all.

        The first sequence is well formed, because SequenceFrames probes ``dataset[0]`` on entry.
        """
        dataset = make_dataset_of([[make_target(1) for _ in range(3)], [object() for _ in range(12)]])
        assert len(SequenceFrames(dataset, Chain(Stride(2), Stride(3)))) == 1 + 2

    def test_an_unseeded_jittered_stage_chains_as_it_runs_alone(self):
        """Planned once per sequence, so a stage drawing fresh entropy on every call still agrees with itself."""
        dataset, _ = make_dataset((9, 7))
        with use_seed(None):
            frames = SequenceFrames(dataset, Chain(Stride(3, jitter=True)))
            assert len(frames) == 3 + 3
            rows = [m[2] for m in emit(frames)]
        totals = [sum(r["frames_represented"] for r in rows if r["sequence"] == s) for s in (0, 1)]
        assert totals == [9.0, 7.0]

    def test_a_short_sequence_is_warned_about_once(self):
        dataset, _ = make_dataset((3, 4))
        with pytest.warns(UserWarning, match="fewer than requested") as caught:
            emit(SequenceFrames(dataset, Chain(EvenlySpaced(8))))
        assert len([w for w in caught if "fewer than requested" in str(w.message)]) == 2

    def test_weights_for_positions_out_of_order_are_refused(self):
        """Normalizing such a plan would leave its weights describing other frames."""
        dataset, _ = make_dataset((6,))
        with pytest.raises(ValueError, match="_Disordered"):
            SequenceFrames(dataset, _Disordered())


@pytest.mark.required
class TestStreamingChains:
    def test_a_stage_after_a_filter_counts_survivors(self):
        """Every 2nd surviving frame, not the survivors whose source position is even."""
        dataset, _ = make_dataset((10,))
        chain = Chain(_KeepIf(lambda i: i % 3 != 0, excludes=True), _EveryOther())
        frames = SequenceFrames(dataset, chain)
        # survivors of the filter: 1, 2, 4, 5, 7, 8 -> every other: 1, 4, 7
        assert frames.frame_map[:, 1].tolist() == [1, 4, 7]

    def test_weights_compose_exactly(self):
        dataset, _ = make_dataset((10,))
        chain = Chain(_KeepIf(lambda i: i % 3 != 0, excludes=True), _EveryOther())
        rows = [m[2] for m in emit(SequenceFrames(dataset, chain))]
        # the filter excludes 0, 3, 6, 9 (mass 4); each kept frame stands for itself and the next survivor
        assert [r["frames_represented"] for r in rows] == [2.0, 2.0, 2.0]
        assert sum(r["frames_represented"] for r in rows) == 10 - 4

    def test_planning_stages_ahead_of_a_streaming_one_plan_over_the_source(self):
        """_LabelIn plans from the source target: 1, 3, 4, 8 -> every other survivor: 1, 4."""
        dataset = _labeled(10, labeled_at=(1, 3, 4, 8), label=7)
        frames = SequenceFrames(dataset, Chain(_LabelIn(7), _EveryOther()))
        assert frames.frame_map[:, 1].tolist() == [1, 4]
        assert [m[2]["frames_represented"] for m in emit(frames)] == [2.0, 2.0]

    def test_factors_from_every_stage_land_on_the_frame(self):
        dataset, _ = make_dataset((4,))
        rows = [m[2] for m in emit(SequenceFrames(dataset, Chain(_Tags("a"), _Tags("b"))))]
        assert [(r["a"], r["b"]) for r in rows] == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_the_same_factor_from_two_stages_is_refused(self):
        dataset, _ = make_dataset((4,))
        with pytest.raises(ValueError, match="'a'"):
            emit(SequenceFrames(dataset, Chain(_Tags("a"), _Tags("a"))))

    def test_a_stage_that_keeps_nothing_leaves_the_sequence_empty(self):
        """Review Focus 1."""
        dataset, _ = make_dataset((6, 4))
        frames = SequenceFrames(dataset, Chain(_KeepIf(lambda i: False), _EveryOther()))
        assert len(frames) == 0
        assert emit(frames) == []

    def test_a_zero_frame_sequence_streams_to_nothing(self):
        """Review Focus 2."""
        dataset, _ = make_dataset((0, 4))
        assert len(SequenceFrames(dataset, Chain(_EveryOther(), _EveryOther()))) == 1

    def test_one_chain_drives_two_views_independently(self):
        """Review Focus 4."""
        chain = Chain(_EveryOther(), _EveryOther())
        first, _ = make_dataset((8,))
        second, _ = make_dataset((6,))
        a, b = SequenceFrames(first, chain).stream(), SequenceFrames(second, chain).stream()
        interleaved = [next(a), next(b), next(a), next(b)]
        assert [cast(dict[str, Any], m[2])["frame"] for m in interleaved] == [0, 0, 4, 4]

    def test_only_a_stage_that_declares_pixels_reads_them(self):
        dataset, _ = make_dataset((4,))
        with pytest.raises(AttributeError, match="FrameInput.PIXELS"):
            emit(SequenceFrames(dataset, Chain(_LiesAboutPixels(), _PixelNovelty())))
        assert len(emit(SequenceFrames(dataset, Chain(_EveryOther(), _PixelNovelty())))) == 2

    def test_a_stage_claiming_more_than_its_input_is_refused_by_name(self):
        """Stage 2 holds 2 source frames, so its 1e6 is refused there, before the sequence's own check."""
        dataset, _ = make_dataset((10,))
        with pytest.raises(ValueError, match=r"stage 2, _Overclaims\(\)"):
            emit(SequenceFrames(dataset, Chain(_KeepIf(lambda i: i < 2, excludes=True), _Overclaims())))

    def test_a_stage_deciding_a_frame_it_has_released_is_refused(self):
        dataset, _ = make_dataset((6,))
        with pytest.raises(ValueError, match="_Buffering"):
            emit(SequenceFrames(dataset, Chain(_EveryOther(), _Buffering(two_pass=False))))

    def test_a_representing_stage_composes_the_masses_it_is_offered(self):
        """Stage 1 leaves 0, 2, 4, 6, 8 at mass 2 each; stage 2 keeps ordinals 0, 2, 4, each with its run."""
        dataset, _ = make_dataset((10,))
        rows = [m[2] for m in emit(SequenceFrames(dataset, Chain(_EveryOther(), _EveryOther())))]
        assert [r["frame"] for r in rows] == [0, 4, 8]
        assert [r["frames_represented"] for r in rows] == [4.0, 4.0, 2.0]

    def test_a_later_stage_whose_first_keep_is_not_its_first_frame_takes_the_frames_before_it(self):
        """Stage 2 keeps 4, 6, 8 of 0, 2, 4, 6, 8, so 4 also stands for 0 and 2: 2 + 2 + 2."""
        dataset, _ = make_dataset((10,))
        rows = [m[2] for m in emit(SequenceFrames(dataset, Chain(_EveryOther(), _KeepIf(lambda i: i >= 4))))]
        assert [r["frame"] for r in rows] == [4, 6, 8]
        assert [r["frames_represented"] for r in rows] == [6.0, 2.0, 2.0]

    def test_a_stage_that_stops_reading_early_still_accounts_for_what_it_never_saw(self):
        """_FirstTwo keeps 0 and 2 of 0, 2, 4, 6, 8 and stops; 2 also stands for 4, 6 and 8."""
        dataset, _ = make_dataset((10,))
        rows = [m[2] for m in emit(SequenceFrames(dataset, Chain(_EveryOther(), _FirstTwo())))]
        assert [r["frame"] for r in rows] == [0, 2]
        assert [r["frames_represented"] for r in rows] == [2.0, 8.0]

    def test_a_chain_lets_go_of_each_frame_once_its_stages_are_done_with_it(self):
        """No frame handle, nor the pixels decoded from it, outlives what the stream and the stages hold."""
        handles: dict[int, Any] = {}
        dataset, _ = make_dataset((40,))
        stream = SequenceFrames(dataset, Chain(_EveryOther(), _Watch(handles))).stream()
        for _ in range(3):
            next(stream)
        gc.collect()
        # 20 is the frame the stream just yielded, and 30 the one every stage holds.
        assert [i for i, ref in sorted(handles.items()) if ref() is not None] == [20, 30]

    def test_a_weight_read_before_it_settles_raises(self):
        """A streaming stage must not decide from a weight its predecessor is still settling."""
        dataset, _ = make_dataset((4,))
        with pytest.raises(RuntimeError, match="settled"):
            emit(SequenceFrames(dataset, Chain(_EveryOther(), _ReadsWeight())))

    @pytest.mark.parametrize("weight", [-1.0, float("nan")])
    def test_a_stage_declaring_a_negative_or_non_finite_weight_is_refused_by_name(self, weight):
        dataset, _ = make_dataset((6,))
        with pytest.raises(ValueError, match=r"stage 2, _Declares.*finite and not negative"):
            emit(SequenceFrames(dataset, Chain(_EveryOther(), _Declares(weight))))

    @pytest.mark.parametrize("n", [10, 11])
    def test_a_stage_after_the_first_deciding_once_its_input_ends_is_refused_on_every_video(self, n):
        """Over 11 frames the walk still holds stage 2's last frame when its input ends, and over 10 it does not."""
        dataset, _ = make_dataset((n,))
        with pytest.raises(ValueError, match=r"stage 2, _FirstAndLast\(\).*after its input ended"):
            emit(SequenceFrames(dataset, Chain(_EveryOther(), _FirstAndLast())))
        assert SequenceFrames(dataset, Chain(_FirstAndLast())).frame_map[:, 1].tolist() == [0, n - 1]


@pytest.mark.required
class TestTheBoundary:
    def test_a_whole_input_stage_first_then_a_streaming_stage(self):
        dataset, counters = make_dataset((10,))
        frames = SequenceFrames(dataset, Chain(_KeepTopTwo(), _EveryOther()))
        # frame_index % 5 is largest (4) at 4 and 9 -> survivors 4, 9 -> every other: 4
        assert frames.frame_map[:, 1].tolist() == [4]
        assert [c["iterations"] for c in counters] == [2]  # two walks, never three

    def test_a_planning_stage_after_a_streaming_one_is_the_boundary(self):
        dataset, _ = make_dataset((20,))
        frames = SequenceFrames(dataset, Chain(_EveryOther(), EvenlySpaced(3), Stride(2)))
        # survivors 0, 2, ..., 18 (10); EvenlySpaced(3) -> ordinals 0, 3, 7 -> 0, 6, 14; Stride(2) -> 0, 14
        assert frames.frame_map[:, 1].tolist() == [0, 14]

    def test_representative_after_thinning_still_accounts_for_every_frame(self):
        dataset, _ = make_dataset((24,))
        rows = [m[2] for m in emit(SequenceFrames(dataset, Chain(Stride(2), Representative(3))))]
        assert sum(r["frames_represented"] for r in rows) == pytest.approx(24.0)

    def test_two_points_needing_the_whole_input_are_refused(self):
        with pytest.raises(ValueError, match="Representative"):
            Chain(Redundancy(), Representative(20), Redundancy(), EvenlySpaced(8))

    @pytest.mark.parametrize(
        "chain",
        [
            lambda: Chain(_KeepTopTwo(), Redundancy()),
            lambda: Chain(Redundancy(), EvenlySpaced(8), Stride(2)),
            lambda: Chain(Redundancy(), Representative(20), EvenlySpaced(8)),
        ],
    )
    def test_the_accepted_shapes(self, chain):
        assert chain().two_pass

    def test_weights_compose_across_both_passes(self):
        """Every other of 20 at mass 2; the top two by frame_index % 5 are 4 and 14, which exclude the rest;
        Stride(2) resolves at once to 4, standing for both; the second pass tags it."""
        dataset, _ = make_dataset((20,))
        chain = Chain(_EveryOther(), _KeepTopTwo(), Stride(2), _Tags("b"))
        rows = [m[2] for m in emit(SequenceFrames(dataset, chain))]
        assert [(r["frame"], r["frames_represented"], r["b"]) for r in rows] == [(4, 4.0, 0)]

    def test_factors_from_either_pass_land_on_the_frame(self):
        dataset, _ = make_dataset((10,))
        rows = [m[2] for m in emit(SequenceFrames(dataset, Chain(_Tags("a"), _KeepTopTwo(), _Tags("b"))))]
        assert [(r["frame"], r["a"], r["b"]) for r in rows] == [(4, 4, 0), (9, 9, 1)]

    def test_a_whole_input_stage_is_offered_settled_weights(self):
        """Every other of 10 leaves masses still settling as each frame passes; _Totals sums them as floats."""
        dataset, _ = make_dataset((10,))
        rows = [m[2] for m in emit(SequenceFrames(dataset, Chain(_EveryOther(), _Totals())))]
        assert [(r["frame"], r["frames_represented"]) for r in rows] == [(0, 10.0)]

    def test_a_whole_input_stage_claiming_more_than_its_input_is_refused_by_name(self):
        dataset, _ = make_dataset((10,))
        with pytest.raises(ValueError, match=r"stage 2, _Totals"):
            emit(SequenceFrames(dataset, Chain(_KeepIf(lambda i: i < 4, excludes=True), _Totals(scale=2.0))))

    def test_a_whole_input_stage_that_stops_reading_early_still_accounts_for_what_it_never_saw(self):
        """Of 0, 2, 4, 6, 8 at mass 2, it reads and keeps the first two, so 2 also stands for 4, 6 and 8."""
        dataset, _ = make_dataset((10,))
        rows = [m[2] for m in emit(SequenceFrames(dataset, Chain(_EveryOther(), _FirstTwoAtOnce())))]
        assert [(r["frame"], r["frames_represented"]) for r in rows] == [(0, 2.0), (2, 8.0)]

    def test_representative_keeping_every_frame_keeps_each_ones_own_mass(self):
        """Indistinguishable frames at masses 1, 4 and 5: keeping all three splits nothing evenly."""
        dataset, _ = make_dataset((10,), shape=(3, 8, 8), still_until=10)
        rows = [m[2] for m in emit(SequenceFrames(dataset, Chain(FrameIndices({0: [0, 1, 5]}), Representative(5))))]
        assert [(r["frame"], r["frames_represented"]) for r in rows] == [(0, 1.0), (1, 4.0), (5, 5.0)]

    def test_the_first_pass_holds_no_frame_it_has_offered(self):
        """Up to a whole-input stage, a chain holds positions, weights and factors, never frames or pixels."""
        dataset, _ = make_dataset((19,))  # so the last source frame reaches it
        watch = _WatchAtOnce()
        stream = SequenceFrames(dataset, Chain(_EveryOther(), watch)).stream()
        next(stream)
        assert len(watch.handles) == 10
        assert watch.alive == []  # once its input ended
        gc.collect()
        assert [i for i, ref in watch.handles.items() if ref() is not None] == []  # once the second pass is under way

    def test_a_frame_kept_twice_before_a_planning_boundary_stays_two_frames(self):
        """Each of 4 frames is kept twice at half its mass, and EvenlySpaced(8) keeps all 8 keeps: 4 owed."""
        dataset, _ = make_dataset((4,))
        rows = [m[2] for m in emit(SequenceFrames(dataset, Chain(_KeepsEachTwice(), EvenlySpaced(8))))]
        assert [(r["frame"], r["copy"], r["frames_represented"]) for r in rows] == [
            (frame, copy, 0.5) for frame in range(4) for copy in (0, 1)
        ]

    def test_a_frame_kept_twice_before_a_whole_input_stage_stays_two_frames(self):
        """Frame 3 has the largest frame_index % 5, so _KeepTopTwo keeps both its keeps, which exclude the rest."""
        dataset, _ = make_dataset((4,))
        rows = [m[2] for m in emit(SequenceFrames(dataset, Chain(_KeepsEachTwice(), _KeepTopTwo())))]
        assert [(r["frame"], r["copy"], r["frames_represented"]) for r in rows] == [(3, 0, 0.5), (3, 1, 0.5)]


@pytest.mark.required
class TestWhatAChainRefuses:
    def test_no_stages(self):
        with pytest.raises(ValueError, match="at least one"):
            Chain()

    def test_a_stage_that_needs_adjacent_frames_after_a_thinning_stage(self):
        with pytest.raises(ValueError, match="_Adjacent"):
            Chain(_LabelIn(7), _Adjacent())

    def test_a_stage_that_needs_adjacent_frames_first_is_fine(self):
        Chain(_Adjacent(), Stride(2))

    def test_a_stage_that_needs_adjacent_frames_after_all_frames_is_fine(self):
        Chain(AllFrames(), _Adjacent())

    def test_a_planning_stage_that_returns_no_plan(self):
        """Inside a chain a stage that implements plan is applied through it, so it has to plan."""
        dataset, _ = make_dataset((4,))
        with pytest.raises(ValueError, match="returned no plan"):
            emit(SequenceFrames(dataset, Chain(_PlansNothing(), _EveryOther())))

    def test_a_stage_that_is_not_a_selector(self):
        with pytest.raises(TypeError, match=r"<class '.*Stride'>"):
            Chain(Stride(2), Stride)  # type: ignore[arg-type]

    def test_nested_chains_flatten(self):
        assert len(Chain(Chain(Stride(2), Stride(3)), Stride(2)).stages) == 3


@pytest.mark.required
class TestPlanAndSelectAgree:
    @pytest.mark.parametrize(
        "selector",
        [
            AllFrames(),
            Stride(3),
            Stride(3, jitter=True, seed=2),
            EvenlySpaced(4),
            EvenlySpaced(4, jitter=True, seed=5),
            FrameIndices({0: [1, 5, 6]}),
        ],
    )
    def test_select_keeps_exactly_what_plan_names(self, selector):
        dataset, _ = make_dataset((11,))
        info = next(iter(sequence_infos(dataset)))[0]
        planned = selector.plan(info).tolist()
        selected = [v.position for v in selector.select(iter(_candidates_for(dataset, 0)))]
        assert selected == planned

    @pytest.mark.filterwarnings("ignore:EvenlySpaced:UserWarning")
    def test_random_chains_match_the_reference(self):
        rng = np.random.default_rng(20261021)
        valid = 0
        for _ in range(200):
            stages = _random_stages(rng)
            try:
                chain = Chain(*stages)
            except ValueError:  # two stages needing their whole input; the next draw replaces it
                continue
            valid += 1
            n = int(rng.integers(0, 30))
            dataset = _labeled(n, labeled_at=tuple(rng.choice(max(n, 1), size=min(n, 3), replace=False)), label=7)
            frames = SequenceFrames(dataset, chain)
            positions, weights = _reference(stages, n, dataset[0][1])
            assert frames.frame_map[:, 1].tolist() == positions, (chain, n)
            assert [m[2]["frames_represented"] for m in emit(frames)] == pytest.approx(weights), (chain, n)
        assert valid >= 150
