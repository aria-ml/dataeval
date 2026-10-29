"""Tests for presenting a tracking dataset as an object-detection dataset of frames."""

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from itertools import islice
from typing import Any, cast

import numpy as np
import pytest

from dataeval import Metadata
from dataeval.data import (
    AllFrames,
    Crop,
    EvenlySpaced,
    FrameCandidate,
    FrameIndices,
    FrameInput,
    FrameRate,
    FrameSelector,
    FrameVerdict,
    Indices,
    Operation,
    Redundancy,
    Representative,
    Reverse,
    SequenceFrames,
    SequenceInfo,
    SourceLocator,
    Stride,
    View,
)
from dataeval.exceptions import MaiteShapeError
from dataeval.flags import ImageStats
from dataeval.protocols import (
    DatasetMetadata,
    DatumMetadata,
    MultiobjectTrackingDatum,
    SingleFrameObjectTrackingTarget,
)
from dataeval.types import SourceIndex

# --------------------------------------------------------------------------------------
# MAITE-shaped stand-ins, matching the style of tests/data/test_tracks.py
# --------------------------------------------------------------------------------------


@dataclass
class _FakeFrameTarget:
    track_ids: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


@dataclass
class _FakeVideoTarget:
    frame_tracks: Sequence[SingleFrameObjectTrackingTarget]


class _FakeFrame:
    """A decoded frame whose pixels are materialized only when read."""

    def __init__(
        self,
        index: int,
        shape: tuple[int, ...],
        time_s: float | None,
        pts: int | None,
        counter: dict,
        value: int | None = None,
    ):
        self.frame_index = index
        self._value = index if value is None else value
        self._shape = shape
        self._counter = counter
        if time_s is not None:
            self.time_s = time_s
        if pts is not None:
            self.pts = pts

    @property
    def pixels(self) -> np.ndarray:
        self._counter["pixels"] += 1
        return np.full(self._shape, self._value % 251, dtype=np.uint8)


class _CountingStream:
    """A VideoStream that records how many times it is iterated and how far."""

    def __init__(self, n_frames, shape, counter, timed=True, index=0, extra=0, still_until=None, numbering=None):
        self._still_until = still_until
        self._numbering = numbering
        self._n = n_frames
        self._shape = shape
        self._counter = counter
        self._timed = timed
        self._index = index
        self._extra = extra

    def __iter__(self) -> Iterator[_FakeFrame]:
        self._counter["iterations"] += 1
        for i in range(self._n + self._extra):
            self._counter["frames"] += 1
            yield _FakeFrame(
                i if self._numbering is None else self._numbering(i),
                self._shape,
                (i / 30.0) if self._timed else None,
                (i * 1001) if self._timed else None,
                self._counter,
                # Frames before `still_until` are pixel-identical, so a sequence can hold a static
                # stretch followed by one where every frame differs.
                value=0 if self._still_until is not None and i < self._still_until else i,
            )


class _FakeDataset:
    def __init__(self, data, metadata):
        self._data = data
        self.metadata = metadata

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]


def make_target(n_dets: int = 2) -> SingleFrameObjectTrackingTarget:
    return cast(
        SingleFrameObjectTrackingTarget,
        _FakeFrameTarget(
            track_ids=np.arange(n_dets, dtype=np.int64),
            boxes=np.tile(np.array([1.0, 2.0, 9.0, 10.0], dtype=np.float32), (n_dets, 1)),
            scores=np.ones(n_dets, dtype=np.float32),
            labels=np.zeros(n_dets, dtype=np.int64),
        ),
    )


def make_dataset(
    frame_counts=(6, 4), shape=(3, 12, 14), timed=True, extra=0, counters=None, still_until=None, numbering=None
):
    """A tracking dataset whose streams report how often they are walked.

    ``numbering`` maps a frame's position to the ``frame_index`` it declares; by default the two agree.
    """
    counters = [] if counters is None else counters
    data: list[MultiobjectTrackingDatum] = []
    for seq, n in enumerate(frame_counts):
        counter = {"iterations": 0, "frames": 0, "pixels": 0}
        counters.append(counter)
        stream = _CountingStream(
            n,
            shape,
            counter,
            timed=timed,
            index=seq,
            extra=extra if seq == 0 else 0,
            still_until=still_until,
            numbering=numbering,
        )
        target = _FakeVideoTarget(frame_tracks=[make_target() for _ in range(n)])
        data.append((
            cast(Any, stream),
            cast(Any, target),
            cast(DatumMetadata, {"id": f"vid{seq}", "height": shape[1], "width": shape[2]}),
        ))
    return _FakeDataset(data, DatasetMetadata({"id": "videos", "index2label": {0: "thing"}})), counters


class _Buffering(FrameSelector):
    """Decides only after seeing the whole sequence -- the shape a medoid selector has."""

    needs: FrameInput = FrameInput.STRUCTURE

    def __init__(self, two_pass: bool):
        self.two_pass = two_pass

    def select(self, frames):
        positions = [frame.position for frame in frames]
        return (FrameVerdict(p) for p in positions if p % 2 == 0)


class _PixelNovelty(FrameSelector):
    """An online, content-dependent selector -- the shape a key-frame extractor has."""

    needs: FrameInput = FrameInput.PIXELS

    def select(self, frames):
        previous = None
        for frame in frames:
            mean = float(frame.pixels.mean())
            if previous is None or abs(mean - previous) > 0.5:
                previous = mean
                yield FrameVerdict(frame.position, factors={"novelty": mean})


class _LiesAboutPixels(FrameSelector):
    needs: FrameInput = FrameInput.STRUCTURE

    def select(self, frames):
        return (FrameVerdict(frame.position) for frame in frames if frame.pixels.mean() >= 0)


class _ReservedName(FrameSelector):
    needs: FrameInput = FrameInput.STRUCTURE

    def select(self, frames):
        return (FrameVerdict(frame.position, factors={"sequence": 99}) for frame in frames)


class _Fickle(FrameSelector):
    """A stateful selector that changes its answer every time it is asked."""

    needs: FrameInput = FrameInput.STRUCTURE

    def __init__(self):
        self.calls = 0

    def select(self, frames):
        self.calls += 1
        return (FrameVerdict(frame.position) for frame in frames if (frame.position + self.calls) % 2 == 0)


class _FickleFactors(FrameSelector):
    """Keeps the same frames every time it is asked, but labels them differently each time."""

    needs: FrameInput = FrameInput.STRUCTURE

    def __init__(self):
        self.calls = 0

    def select(self, frames):
        self.calls += 1
        return (
            FrameVerdict(frame.position, factors={"call": self.calls}) for frame in frames if frame.position % 2 == 0
        )


class _FickleEvidence(FrameSelector):
    """Keeps the same frames every time it is asked, but says each time whether it measured them."""

    needs: FrameInput = FrameInput.STRUCTURE

    def __init__(self):
        self.calls = 0

    def select(self, frames):
        self.calls += 1
        return (
            FrameVerdict(frame.position, unmeasured=self.calls % 2 == 0) for frame in frames if frame.position % 2 == 0
        )


class _EveryOther(FrameSelector):
    """Keeps even positions, deciding frame by frame rather than planning."""

    needs: FrameInput = FrameInput.STRUCTURE

    def select(self, frames):
        return (FrameVerdict(frame.position) for frame in frames if frame.position % 2 == 0)


class _KeepsNothing(FrameSelector):
    """Reads every frame and keeps none."""

    needs: FrameInput = FrameInput.STRUCTURE

    def select(self, frames):
        return (FrameVerdict(frame.position) for frame in frames if False)


class _FirstTwo(FrameSelector):
    """Keeps the first two frames and stops reading the sequence there."""

    needs: FrameInput = FrameInput.STRUCTURE

    def select(self, frames):
        return (FrameVerdict(frame.position) for frame in islice(frames, 2))


class _ExcludeEven(FrameSelector):
    """A filter: even positions do not belong, so no kept frame stands for them, not even frame 0."""

    needs: FrameInput = FrameInput.STRUCTURE
    excludes = True

    def select(self, frames):
        return (FrameVerdict(frame.position) for frame in frames if frame.position % 2 == 1)


class _Overclaims(FrameSelector):
    """Declares far more mass for its one kept frame than the sequence holds."""

    needs: FrameInput = FrameInput.STRUCTURE

    def select(self, frames):
        return (FrameVerdict(frame.position, weight=1e6) for frame in frames if frame.position == 0)


class _Declares(FrameSelector):
    """Keeps every frame, declaring ``weight`` for each."""

    needs: FrameInput = FrameInput.STRUCTURE

    def __init__(self, weight: float):
        self.weight = weight

    def select(self, frames):
        return (FrameVerdict(frame.position, weight=self.weight) for frame in frames)


# --------------------------------------------------------------------------------------


def _info(index: int, source_id: str, n_frames: int) -> SequenceInfo:
    """A SequenceInfo for driving a selector's `select` directly, outside SequenceFrames."""
    return SequenceInfo(index, source_id, n_frames, cast(DatumMetadata, {}))


def emit(frames: SequenceFrames) -> list[tuple[Any, Any, dict[str, Any]]]:
    """Stream a view, widening each frame's metadata to a plain dict for assertion."""
    return [(pixels, target, cast(dict[str, Any], meta)) for pixels, target, meta in frames.stream()]


def meta_at(frames: SequenceFrames, index: int) -> dict[str, Any]:
    """The metadata of one frame reached through the Dataset protocol."""
    return cast(dict[str, Any], frames[index][2])


@pytest.mark.required
class TestConstruction:
    def test_rejects_a_non_tracking_dataset(self):
        images = [np.zeros((3, 4, 4)) for _ in range(3)]
        with pytest.raises(MaiteShapeError):
            SequenceFrames(cast(Any, images))

    def test_frame_counts_come_from_targets_without_decoding(self):
        dataset, counters = make_dataset((6, 4))
        frames = SequenceFrames(dataset)
        assert frames.n_source_frames == 10
        assert len(frames) == 10
        assert all(counter["iterations"] == 0 for counter in counters), "constructing decoded something"

    def test_default_selector_keeps_every_frame(self):
        dataset, _ = make_dataset((5,))
        assert len(SequenceFrames(dataset)) == len(SequenceFrames(dataset, AllFrames())) == 5

    def test_source_and_metadata(self):
        dataset, _ = make_dataset((3,))
        frames = SequenceFrames(dataset)
        assert frames.source is dataset
        assert frames.metadata["id"] == "videos-frames"
        assert cast(dict, frames.metadata)["index2label"] == {0: "thing"}

    def test_a_dataset_without_metadata_gives_its_view_a_default_id_and_no_labels(self):
        dataset, _ = make_dataset((3,))

        class Bare:
            def __len__(self) -> int:
                return len(dataset)

            def __getitem__(self, index: int) -> Any:
                return dataset[index]

        frames = SequenceFrames(cast(Any, Bare()))
        assert frames.metadata["id"] == "dataset-frames"
        assert "index2label" not in frames.metadata
        assert len(frames) == 3

    def test_repr_and_str(self):
        dataset, _ = make_dataset((3,))
        frames = SequenceFrames(dataset, Stride(2))
        assert "SequenceFrames" in repr(frames)
        assert "sequences" in str(frames)


@pytest.mark.required
class TestDecodeAccounting:
    """What this costs is a property worth asserting, not inferring from a docstring."""

    def test_stream_walks_each_sequence_exactly_once(self):
        dataset, counters = make_dataset((6, 4))
        emit(SequenceFrames(dataset, Stride(2)))
        assert [counter["iterations"] for counter in counters] == [1, 1]

    def test_two_pass_walks_each_sequence_exactly_twice(self):
        dataset, counters = make_dataset((6, 4))
        emit(SequenceFrames(dataset, _Buffering(two_pass=True)))
        assert [counter["iterations"] for counter in counters] == [2, 2]

    def test_len_is_free_for_a_planning_selector(self):
        dataset, counters = make_dataset((6, 4))
        frames = SequenceFrames(dataset, Stride(2))
        assert len(frames) == 5
        assert all(counter["iterations"] == 0 for counter in counters)

    def test_len_realizes_once_for_a_content_selector(self):
        dataset, counters = make_dataset((6, 4))
        frames = SequenceFrames(dataset, FrameRate(10.0))
        assert len(frames) > 0
        walked = [counter["iterations"] for counter in counters]
        len(frames)
        assert [counter["iterations"] for counter in counters] == walked, "realization was not cached"

    def test_a_timing_selector_never_materializes_pixels(self):
        dataset, counters = make_dataset((8,))
        frames = SequenceFrames(dataset, FrameRate(5.0))
        len(frames)
        assert counters[0]["pixels"] == 0

    def test_streaming_materializes_only_the_frames_it_yields(self):
        dataset, counters = make_dataset((9,))
        emitted = emit(SequenceFrames(dataset, Stride(3)))
        assert len(emitted) == 3
        assert counters[0]["pixels"] == 3


@pytest.mark.required
class TestSelectorContract:
    """The guards that decide whether a key-frame extractor lands cleanly or forces a redesign."""

    def test_declared_needs_is_enforced(self):
        dataset, _ = make_dataset((4,))
        with pytest.raises(AttributeError, match="FrameInput.PIXELS"):
            emit(SequenceFrames(dataset, _LiesAboutPixels()))

    def test_pixels_are_available_when_declared(self):
        dataset, _ = make_dataset((4,))
        assert len(emit(SequenceFrames(dataset, _PixelNovelty()))) >= 1

    def test_pixels_are_materialized_once_and_then_cached(self):
        """A second read of the same candidate must not pay for a second decode."""

        class _Frame:
            def __init__(self):
                self.reads = 0

            @property
            def pixels(self):
                self.reads += 1
                return np.full((3, 12, 14), 5, dtype=np.uint8)

        frame = _Frame()
        candidate = FrameCandidate(
            sequence=SequenceFrames(make_dataset((1,))[0])._sequences[0],
            position=0,
            frame_index=0,
            time_s=None,
            pts=None,
            target=make_target(),
            _frame=frame,
        )
        first = candidate.pixels
        second = candidate.pixels
        assert first is second, "the second read was not served from the cache"
        assert frame.reads == 1
        np.testing.assert_array_equal(first, np.full((3, 12, 14), 5, dtype=np.uint8))

    def test_buffering_without_declaring_two_pass_raises_and_names_the_fix(self):
        dataset, _ = make_dataset((6,))
        with pytest.raises(ValueError, match="two_pass"):
            emit(SequenceFrames(dataset, _Buffering(two_pass=False)))

    def test_the_same_selector_works_once_it_declares_two_pass(self):
        dataset, _ = make_dataset((6,))
        emitted = emit(SequenceFrames(dataset, _Buffering(two_pass=True)))
        assert [meta["frame"] for _, _, meta in emitted] == [0, 2, 4]

    @pytest.mark.parametrize("selector", [AllFrames(), Stride(2), Stride(3), FrameIndices({0: [0, 3, 5]})])
    def test_plan_and_select_agree(self, selector):
        """A shortcut that disagrees with the walk it shortcuts is invisible downstream."""
        dataset, _ = make_dataset((6,))
        info = SequenceFrames(dataset, selector)._sequences[0]
        planned = selector.plan(info).tolist()

        walked = SequenceFrames(dataset, AllFrames())
        candidates = walked._candidates(info, [], {})
        assert [verdict.position for verdict in selector.select(candidates)] == planned

    def test_reserved_factor_names_are_rejected(self):
        dataset, _ = make_dataset((3,))
        with pytest.raises(ValueError, match="reserved factor"):
            emit(SequenceFrames(dataset, _ReservedName()))

    def test_selector_factors_reach_the_frame_metadata(self):
        dataset, _ = make_dataset((5,))
        emitted = emit(SequenceFrames(dataset, _PixelNovelty()))
        assert all("novelty" in meta for _, _, meta in emitted)

    def test_frame_target_count_mismatch_raises(self):
        dataset, _ = make_dataset((6, 4), extra=2)
        with pytest.raises(ValueError, match="more frames"):
            emit(SequenceFrames(dataset))


@pytest.mark.required
class TestShippedSelectors:
    @pytest.mark.parametrize(("step", "expected"), [(1, [0, 1, 2, 3, 4, 5]), (2, [0, 2, 4]), (4, [0, 4])])
    def test_stride(self, step, expected):
        dataset, _ = make_dataset((6,))
        emitted = emit(SequenceFrames(dataset, Stride(step)))
        assert [meta["frame"] for _, _, meta in emitted] == expected

    def test_stride_rejects_zero(self):
        with pytest.raises(ValueError, match="at least 1"):
            Stride(0)

    def test_frame_indices_replays_a_selection(self):
        dataset, _ = make_dataset((6, 4))
        emitted = emit(SequenceFrames(dataset, FrameIndices({0: [0, 3], 1: [2]})))
        assert [(meta["sequence"], meta["frame"]) for _, _, meta in emitted] == [(0, 0), (0, 3), (1, 2)]

    def test_frame_indices_drops_out_of_range_positions(self):
        dataset, _ = make_dataset((3,))
        assert len(SequenceFrames(dataset, FrameIndices({0: [0, 99]}))) == 1

    def test_frame_indices_rejects_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            FrameIndices({0: [-1]})

    def test_frame_indices_round_trips_another_selector(self):
        """Reproducing a selection is what makes it reviewable."""
        dataset, _ = make_dataset((9,))
        chosen = SequenceFrames(dataset, _PixelNovelty()).frame_map
        replayed = SequenceFrames(dataset, FrameIndices({0: chosen[:, 1].tolist()})).frame_map
        np.testing.assert_array_equal(chosen, replayed)

    def test_frame_rate_thins_on_real_timestamps(self):
        # 30 fps source, 10 fps target -> every third frame
        dataset, _ = make_dataset((9,))
        emitted = emit(SequenceFrames(dataset, FrameRate(10.0)))
        assert [meta["frame"] for _, _, meta in emitted] == [0, 3, 6]

    def test_frame_rate_rejects_non_positive(self):
        with pytest.raises(ValueError, match="positive"):
            FrameRate(0)

    def test_frame_rate_without_timestamps_keeps_everything(self):
        """Thinning on a guessed frame rate would make every derived timing quietly wrong."""
        dataset, _ = make_dataset((6,), timed=False)
        assert len(emit(SequenceFrames(dataset, FrameRate(1.0)))) == 6


@pytest.mark.required
class TestBinnedSelectors:
    """Stride and EvenlySpaced: the two rules for sizing a bin, jittered and not."""

    @pytest.mark.parametrize(("count", "expected"), [(1, [0]), (2, [0, 3]), (3, [0, 2, 4]), (6, [0, 1, 2, 3, 4, 5])])
    def test_evenly_spaced_spans_the_sequence(self, count, expected):
        dataset, _ = make_dataset((6,))
        emitted = emit(SequenceFrames(dataset, EvenlySpaced(count)))
        assert [meta["frame"] for _, _, meta in emitted] == expected

    def test_evenly_spaced_is_constant_per_sequence_where_stride_is_not(self):
        """The whole point of fixing the count: a long video cannot drown out a short one."""
        dataset, _ = make_dataset((12, 4))
        even = SequenceFrames(dataset, EvenlySpaced(3)).frame_map
        strided = SequenceFrames(dataset, Stride(3)).frame_map
        assert np.bincount(even[:, 0]).tolist() == [3, 3]
        assert np.bincount(strided[:, 0]).tolist() == [4, 2]

    def test_evenly_spaced_short_sequence_warns_and_contributes_every_frame(self, caplog):
        dataset, _ = make_dataset((3,))
        with caplog.at_level(logging.WARNING), pytest.warns(UserWarning, match="fewer than requested count 10"):
            frames = SequenceFrames(dataset, EvenlySpaced(10))
        assert len(frames) == 3
        assert "EvenlySpaced: sequence 0" in caplog.text
        assert "fewer than requested count 10" in caplog.text

    def test_evenly_spaced_rejects_non_positive(self):
        with pytest.raises(ValueError, match="at least 1"):
            EvenlySpaced(0)

    @pytest.mark.parametrize(("selector", "step"), [(Stride, 3), (EvenlySpaced, 4)], ids=["stride", "evenly_spaced"])
    def test_jitter_keeps_one_frame_per_bin(self, selector, step):
        """Jitter moves the pick within its bin; it does not change how many bins there are."""
        dataset, _ = make_dataset((12,))
        plain = SequenceFrames(dataset, selector(step)).frame_map
        jittered = SequenceFrames(dataset, selector(step, jitter=True, seed=0)).frame_map
        assert len(jittered) == len(plain)
        # one pick per [start, start + 3) bin, since 12 frames / 4 bins == stride 3
        for (_, picked), (_, start) in zip(jittered, plain, strict=True):
            assert start <= picked < start + 3

    @pytest.mark.parametrize("selector", [Stride, EvenlySpaced])
    def test_jitter_is_reproducible_under_a_seed(self, selector):
        dataset, _ = make_dataset((40, 25))
        first = SequenceFrames(dataset, selector(4, jitter=True, seed=7)).frame_map
        second = SequenceFrames(dataset, selector(4, jitter=True, seed=7)).frame_map
        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize("selector", [Stride, EvenlySpaced])
    def test_different_seeds_differ(self, selector):
        dataset, _ = make_dataset((60,))
        first = SequenceFrames(dataset, selector(3, jitter=True, seed=1)).frame_map
        second = SequenceFrames(dataset, selector(3, jitter=True, seed=2)).frame_map
        assert not np.array_equal(first, second)

    @pytest.mark.parametrize("selector", [Stride, EvenlySpaced])
    def test_jitter_survives_reordering_the_dataset_around_a_sequence(self, selector):
        """Keyed on source_id, so a video's key frames do not depend on what precedes it."""
        dataset, _ = make_dataset((9, 9, 9))
        straight = SequenceFrames(dataset, selector(3, jitter=True, seed=0))
        reversed_ = SequenceFrames(View(dataset, Reverse()), selector(3, jitter=True, seed=0))

        def frames_of(view, source_id):
            return [meta["frame"] for _, _, meta in emit(view) if meta["source_id"] == source_id]

        assert frames_of(straight, "vid2") == frames_of(reversed_, "vid2")

    @pytest.mark.parametrize(("selector", "arg"), [(Stride, 2), (EvenlySpaced, 5)], ids=["stride", "evenly_spaced"])
    def test_jitter_plans_without_decoding(self, selector, arg):
        """Jitter is still a decision from structure alone, so length stays free."""
        dataset, counters = make_dataset((10,))
        frames = SequenceFrames(dataset, selector(arg, jitter=True, seed=0))
        assert len(frames) == 5
        assert counters[0]["pixels"] == 0

    @pytest.mark.parametrize("selector", [Stride, EvenlySpaced])
    def test_streaming_select_agrees_with_the_plan(self, selector):
        """`plan` is a shortcut, not an alternative -- `select` must reach the same frames."""
        chosen = selector(3, jitter=True, seed=5)
        dataset, _ = make_dataset((14, 8))
        planned = SequenceFrames(dataset, chosen).frame_map

        streamed = []
        for info in [_info(0, "vid0", 14), _info(1, "vid1", 8)]:
            candidates = iter([
                FrameCandidate(info, position, position, None, None, make_target()) for position in range(info.n_frames)
            ])
            streamed += [(info.index, v.position) for v in chosen.select(candidates)]
        np.testing.assert_array_equal(planned, np.array(streamed))


@pytest.mark.required
class TestRepresentativeSelector:
    """Kernel herding over frame descriptors: chosen by appearance, not by position."""

    def test_keeps_the_requested_count(self):
        dataset, _ = make_dataset((12,))
        assert len(emit(SequenceFrames(dataset, Representative(4, batch_size=4)))) == 4

    def test_short_sequence_contributes_every_frame(self):
        dataset, _ = make_dataset((3,))
        assert len(emit(SequenceFrames(dataset, Representative(10, batch_size=4)))) == 3

    def test_verdicts_arrive_in_position_order(self):
        dataset, _ = make_dataset((16,))
        frames = [meta["frame"] for _, _, meta in emit(SequenceFrames(dataset, Representative(5, batch_size=4)))]
        assert frames == sorted(frames)

    def test_is_deterministic(self):
        """A greedy argmax, so the same sequence and count always give the same frames."""
        dataset, _ = make_dataset((14,))
        first = SequenceFrames(dataset, Representative(5, batch_size=4)).frame_map
        second = SequenceFrames(dataset, Representative(5, batch_size=4)).frame_map
        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize("batch_size", [1, 3, 5, 100])
    def test_batching_does_not_change_the_answer(self, batch_size):
        """Batching is how frames are described, not what they are described as."""
        dataset, _ = make_dataset((15,))
        expected = SequenceFrames(dataset, Representative(4, batch_size=15)).frame_map
        np.testing.assert_array_equal(
            SequenceFrames(dataset, Representative(4, batch_size=batch_size)).frame_map, expected
        )

    def test_weights_sum_to_the_frame_count(self):
        """A kept frame stands for a scattered set, so it declares what it stands for."""
        dataset, _ = make_dataset((12,))
        emitted = emit(SequenceFrames(dataset, Representative(4, batch_size=4)))
        assert sum(meta["frames_represented"] for _, _, meta in emitted) == pytest.approx(12)

    def test_spends_more_of_its_budget_where_the_content_changes(self):
        """The reason to select on appearance rather than position."""
        # 16 pixel-identical frames, then 8 that each differ.
        dataset, _ = make_dataset((24,), shape=(3, 8, 8), still_until=16)

        def in_tail(selector):
            emitted = emit(SequenceFrames(dataset, selector))
            return sum(meta["frame"] >= 16 for _, _, meta in emitted)

        assert in_tail(Representative(4, batch_size=8)) > in_tail(EvenlySpaced(4))

    def test_weights_follow_the_mass_a_frame_stands_for(self):
        """A frame representing a long static stretch has to count for more than one showing once."""
        dataset, _ = make_dataset((24,), shape=(3, 8, 8), still_until=16)
        emitted = emit(SequenceFrames(dataset, Representative(4, batch_size=8)))
        weights = {meta["frame"]: meta["frames_represented"] for _, _, meta in emitted}
        static = sum(weight for frame, weight in weights.items() if frame < 16)
        assert static == pytest.approx(16)
        assert sum(weights.values()) == pytest.approx(24)

    def test_identical_frames_keep_the_first(self):
        """A zero-width kernel cannot choose, so the choice is made explicitly rather than by NaN."""
        dataset, _ = make_dataset((10,), shape=(3, 8, 8), still_until=10)
        emitted = emit(SequenceFrames(dataset, Representative(3, batch_size=4)))
        assert [meta["frame"] for _, _, meta in emitted] == [0, 1, 2]
        assert sum(meta["frames_represented"] for _, _, meta in emitted) == pytest.approx(10)

    def test_declares_that_it_reads_pixels_and_buffers(self):
        selector = Representative(3)
        assert selector.needs == FrameInput.PIXELS
        assert selector.two_pass is True
        assert selector.plan(_info(0, "vid0", 10)) is None

    def test_rejects_bad_arguments(self):
        with pytest.raises(ValueError, match="at least 1"):
            Representative(0)
        with pytest.raises(ValueError, match="bandwidth must be positive"):
            Representative(4, bandwidth=-1.0)

    def test_an_explicit_bandwidth_is_honored(self):
        dataset, _ = make_dataset((16,))
        wide = SequenceFrames(dataset, Representative(5, bandwidth=1e6, batch_size=8)).frame_map
        narrow = SequenceFrames(dataset, Representative(5, bandwidth=1e-2, batch_size=8)).frame_map
        assert not np.array_equal(wide, narrow)


@pytest.mark.required
class TestFrameMetadata:
    def test_carries_provenance_back_to_the_source(self):
        dataset, _ = make_dataset((4, 3))
        emitted = emit(SequenceFrames(dataset))
        first = emitted[0][2]
        assert first["sequence"] == 0
        assert first["source_id"] == "vid0"
        assert first["frame"] == 0
        assert first["sequence_n_frames"] == 4
        assert emitted[4][2]["sequence"] == 1
        assert emitted[4][2]["source_id"] == "vid1"

    def test_ids_count_across_the_whole_view(self):
        dataset, _ = make_dataset((4, 3))
        assert [meta["id"] for _, _, meta in emit(SequenceFrames(dataset))] == list(range(7))

    def test_sequence_position_spans_zero_to_one_and_is_monotone(self):
        dataset, _ = make_dataset((5,))
        positions = [meta["sequence_position"] for _, _, meta in emit(SequenceFrames(dataset))]
        assert positions[0] == 0.0
        assert positions[-1] == 1.0
        assert positions == sorted(positions)

    def test_timings_pass_through_when_declared(self):
        dataset, _ = make_dataset((4,))
        emitted = emit(SequenceFrames(dataset))
        assert [meta["pts"] for _, _, meta in emitted] == [0, 1001, 2002, 3003]
        assert emitted[1][2]["time_s"] == pytest.approx(1 / 30)

    def test_timings_absent_when_the_stream_declares_none(self):
        dataset, _ = make_dataset((4,), timed=False)
        emitted = emit(SequenceFrames(dataset))
        assert all("time_s" not in meta and "pts" not in meta for _, _, meta in emitted)


@pytest.mark.required
class TestTemporalWeights:
    """The weight invariant is invisible in every output, which is why it is asserted here."""

    @pytest.mark.parametrize("selector", [AllFrames(), Stride(2), Stride(3), FrameRate(10.0)])
    def test_frames_represented_sums_to_the_source_frame_count(self, selector):
        dataset, _ = make_dataset((9, 7))
        emitted = emit(SequenceFrames(dataset, selector))
        for sequence, expected in ((0, 9), (1, 7)):
            total = sum(meta["frames_represented"] for _, _, meta in emitted if meta["sequence"] == sequence)
            assert total == expected

    def test_stride_weights_are_the_stride_with_the_remainder_last(self):
        dataset, _ = make_dataset((7,))
        emitted = emit(SequenceFrames(dataset, Stride(3)))
        assert [meta["frames_represented"] for _, _, meta in emitted] == [3.0, 3.0, 1.0]

    @pytest.mark.parametrize("selector", [AllFrames(), Stride(2), Stride(3), FrameRate(10.0)])
    def test_seconds_represented_is_present_on_every_row_and_sums_to_the_span(self, selector):
        """A partially populated factor is dropped from analysis entirely, so the last kept frame
        of each sequence must carry a span too -- to the sequence's end, not to a successor it
        does not have."""
        dataset, _ = make_dataset((9, 7))
        emitted = emit(SequenceFrames(dataset, selector))
        assert all("seconds_represented" in meta for _, _, meta in emitted)
        for sequence, n in ((0, 9), (1, 7)):
            spans = [m["seconds_represented"] for _, _, m in emitted if m["sequence"] == sequence]
            assert sum(spans) == pytest.approx((n - 1) / 30)

    def test_seconds_represented_absent_without_timestamps(self):
        dataset, _ = make_dataset((6,), timed=False)
        emitted = emit(SequenceFrames(dataset, Stride(2)))
        assert all("seconds_represented" not in meta for _, _, meta in emitted)

    def test_unequal_frame_rates_weight_differently_by_count_and_by_time(self):
        """A frame is not a constant slice of time, which frame-count weighting cannot see."""
        dataset, _ = make_dataset((9,))
        fast = emit(SequenceFrames(dataset, Stride(1)))
        by_count = sum(meta["frames_represented"] for _, _, meta in fast)
        by_time = sum(meta.get("seconds_represented", 0.0) for _, _, meta in fast)
        assert by_count == 9
        assert by_time == pytest.approx(8 / 30)


@pytest.mark.required
class TestTheWeightContract:
    @pytest.mark.parametrize(
        "selector",
        [Stride(3, jitter=True, seed=1), EvenlySpaced(2, jitter=True, seed=4), FrameIndices({0: [2, 4], 1: [3]})],
    )
    def test_a_thinning_selector_accounts_for_every_frame(self, selector):
        dataset, _ = make_dataset((6, 4))
        frames = SequenceFrames(dataset, selector)
        per_sequence: dict[int, float] = {}
        for _, _, meta in emit(frames):
            per_sequence[meta["sequence"]] = per_sequence.get(meta["sequence"], 0.0) + meta["frames_represented"]
        assert per_sequence == {0: 6.0, 1: 4.0}

    def test_the_first_kept_frame_stands_for_the_frames_before_it(self):
        dataset, _ = make_dataset((6,))
        rows = [m[2] for m in emit(SequenceFrames(dataset, FrameIndices({0: [2, 4]})))]
        assert [r["frames_represented"] for r in rows] == [4.0, 2.0]
        assert rows[0]["seconds_represented"] == pytest.approx(4 / 30.0)  # frames are 1/30 s apart

    def test_an_excluding_selector_gives_each_kept_frame_its_own_mass(self):
        """Its first kept frame is 1, which stands for itself and not for the excluded frame 0."""
        dataset, _ = make_dataset((6,))
        rows = [m[2] for m in emit(SequenceFrames(dataset, _ExcludeEven()))]
        assert [r["frame"] for r in rows] == [1, 3, 5]
        assert [r["frames_represented"] for r in rows] == [1.0, 1.0, 1.0]

    def test_a_selector_claiming_more_mass_than_it_was_given_is_refused(self):
        dataset, _ = make_dataset((6,))
        with pytest.raises(ValueError, match="_Overclaims"):
            emit(SequenceFrames(dataset, _Overclaims()))

    @pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
    def test_a_negative_or_non_finite_weight_is_refused_by_name(self, weight):
        dataset, _ = make_dataset((6,))
        with pytest.raises(ValueError, match=r"_Declares.*finite and not negative"):
            emit(SequenceFrames(dataset, _Declares(weight)))

    def test_representative_counts_the_mass_each_candidate_carries(self):
        from dataeval.data._selectors import _kernel_herd

        points = np.array([[0.0], [0.1], [10.0]])
        _, weights = _kernel_herd(points, 2, bandwidth=1.0, masses=np.array([1.0, 1.0, 5.0]))
        assert sorted(weights.tolist()) == [2.0, 5.0]


@pytest.mark.required
class TestDatasetProtocol:
    def test_getitem_matches_stream(self):
        dataset, _ = make_dataset((5, 3))
        frames = SequenceFrames(dataset, Stride(2))
        streamed = emit(frames)
        for index, (pixels, target, meta) in enumerate(streamed):
            got_pixels, got_target, got_meta = frames[index]
            got_meta = cast(dict[str, Any], got_meta)
            np.testing.assert_array_equal(got_pixels, pixels)
            assert got_target is target
            assert got_meta == meta

    def test_iter_matches_stream(self):
        dataset, _ = make_dataset((5, 3))
        frames = SequenceFrames(dataset, Stride(2))
        assert [cast(dict, m)["id"] for _, _, m in frames] == [m["id"] for _, _, m in emit(frames)]

    def test_repeated_and_random_access(self):
        """Re-reading a position must not fall off the cursor; Metadata probes dataset[0] twice."""
        dataset, _ = make_dataset((5, 3))
        frames = SequenceFrames(dataset, Stride(2))
        first = meta_at(frames, 0)["frame"]
        assert meta_at(frames, 0)["frame"] == first
        assert meta_at(frames, 2)["frame"] == 4
        assert meta_at(frames, 2)["frame"] == 4
        assert meta_at(frames, 1)["frame"] == 2  # backward, re-decodes
        assert meta_at(frames, 4)["frame"] == 2

    def test_metadata_structures_the_frame_view(self):
        """Metadata(SequenceFrames(...)) is object detection: unit = frame, instance = detection."""
        from dataeval import Metadata

        dataset, _ = make_dataset((6, 4))
        md = Metadata(SequenceFrames(dataset, Stride(2)))
        assert md.levels == ("unit", "instance")
        assert md.level_counts == {"unit": 5, "instance": 10}
        for name in ("sequence", "frame", "frames_represented", "sequence_position", "sequence_n_frames"):
            assert any(factor.endswith(name) for factor in md.factor_names), name

    def test_sequence_is_a_unit_factor_for_grouped_splitting(self):
        """split_on="sequence" needs sequence identity on every frame row."""
        from dataeval import Metadata

        dataset, _ = make_dataset((6, 4))
        md = Metadata(SequenceFrames(dataset, Stride(2)))
        rows = md.rows_at("unit")
        assert sorted(set(rows["sequence"].to_list())) == [0, 1]

    def test_negative_and_out_of_range_indices(self):
        dataset, _ = make_dataset((4,))
        frames = SequenceFrames(dataset)
        assert meta_at(frames, -1)["frame"] == 3
        with pytest.raises(IndexError):
            frames[4]

    def test_target_is_the_frames_own_and_keeps_track_ids(self):
        dataset, _ = make_dataset((3,))
        _, target, _ = SequenceFrames(dataset)[0]
        assert hasattr(target, "track_ids")
        np.testing.assert_array_equal(np.asarray(target.boxes).shape, (2, 4))

    def test_satisfies_the_object_detection_shape(self):
        from dataeval.utils.data import validate_dataset

        dataset, _ = make_dataset((3,))
        assert validate_dataset(SequenceFrames(dataset), expected="object_detection") == "object_detection"

    def test_frame_map_and_sequence_offsets(self):
        dataset, _ = make_dataset((5, 3))
        frames = SequenceFrames(dataset, Stride(2))
        np.testing.assert_array_equal(frames.frame_map, [[0, 0], [0, 2], [0, 4], [1, 0], [1, 2]])
        np.testing.assert_array_equal(frames.sequence_offsets, [0, 3, 5])

    def test_boxes_helper_bounds_to_the_frame(self):
        dataset, _ = make_dataset((2,))
        frames = SequenceFrames(dataset)
        pixels, target, _ = frames[0]
        boxes = frames.boxes(pixels, target)
        assert len(boxes) == 2

    def test_n_dropped(self):
        dataset, _ = make_dataset((9,))
        frames = SequenceFrames(dataset, Stride(3))
        assert frames.n_dropped == 6


class _ImageStream:
    """A VideoStream replaying prepared frames."""

    def __init__(self, images):
        self._images = images

    def __iter__(self):
        for index, pixels in enumerate(self._images):
            frame = _Prepared()
            frame.frame_index, frame.pixels = index, pixels
            frame.time_s, frame.pts = index / 30.0, index * 1001
            yield frame


class _Prepared:
    """A decoded frame whose attributes are set by the stream."""

    frame_index: int
    pixels: np.ndarray
    time_s: float
    pts: int


class _UnnumberedStream(_ImageStream):
    """A VideoStream whose frames carry pixels and nothing else: no frame_index, time_s or pts."""

    def __iter__(self):
        for pixels in self._images:
            frame = _Prepared()
            frame.pixels = pixels
            yield frame


def image_dataset(images, stream=_ImageStream):
    """A single-sequence tracking dataset replaying prepared frames."""
    data = [
        (
            cast(Any, stream(images)),
            cast(Any, _FakeVideoTarget(frame_tracks=[make_target() for _ in images])),
            cast(DatumMetadata, {"id": "vid0"}),
        )
    ]
    return _FakeDataset(data, DatasetMetadata({"id": "videos"}))


def unnumbered_dataset(n_frames):
    """A single-sequence tracking dataset whose frames declare no number, so position numbers them."""
    return image_dataset(constant_frames(range(n_frames)), _UnnumberedStream)


def constant_frames(fills, shape=(3, 12, 14)):
    """Frames that are each a single constant value."""
    return [np.full(shape, value, dtype=np.uint8) for value in fills]


def drifting_frames(seed=0, size=64, step=8):
    """Frames drifting from one random image to another, a slice of columns at a time."""
    rng = np.random.RandomState(seed)
    start_image = rng.randint(0, 256, (3, size, size)).astype(np.uint8)
    end_image = rng.randint(0, 256, (3, size, size)).astype(np.uint8)
    frames = []
    for columns in range(0, size + 1, step):
        image = start_image.copy()
        image[:, :columns, :] = end_image[:, :columns, :]
        frames.append(image)
    return frames


@pytest.mark.required
class TestRedundancySelector:
    """The first shipped content-dependent selector, and the shape a key-frame extractor takes."""

    def test_identical_frames_collapse_to_one_representative(self):
        dataset = image_dataset(constant_frames([5, 5, 5, 5, 90, 90, 200]))
        emitted = emit(SequenceFrames(dataset, Redundancy(radius=0, method="xxhash")))
        assert [m["frame"] for _, _, m in emitted] == [0, 4, 6]

    def test_distinct_frames_all_survive(self):
        dataset = image_dataset(constant_frames([10, 40, 70, 100, 130, 160]))
        emitted = emit(SequenceFrames(dataset, Redundancy(radius=0, method="xxhash")))
        assert [m["frame"] for _, _, m in emitted] == list(range(6))

    def test_weights_record_what_each_representative_stands_for(self):
        dataset = image_dataset(constant_frames([5, 5, 5, 5, 90, 90, 200]))
        emitted = emit(SequenceFrames(dataset, Redundancy(radius=0, method="xxhash")))
        assert [m["frames_represented"] for _, _, m in emitted] == [4.0, 2.0, 1.0]

    def test_a_frame_that_cannot_be_hashed_is_kept_not_dropped(self):
        """No digest is no evidence the frame repeated, so the frame is kept."""
        dataset, _ = make_dataset((3,), shape=(3, 4, 4))
        emitted = emit(SequenceFrames(dataset, Redundancy(radius=0, method="phash")))
        assert [m["frame"] for _, _, m in emitted] == [0, 1, 2]

    def test_it_declares_that_it_reads_pixels_in_one_pass(self):
        assert Redundancy().needs is FrameInput.PIXELS
        assert Redundancy().two_pass is False

    def test_it_cannot_plan_so_sizing_walks_once(self):
        dataset, counters = make_dataset((6,))
        frames = SequenceFrames(dataset, Redundancy(radius=0, method="xxhash"))
        assert frames._planned is None
        assert len(frames) >= 1
        assert counters[0]["iterations"] == 1

    @pytest.mark.parametrize("method", ["phash", "dhash", "phash_d4", "dhash_d4", "xxhash"])
    def test_every_method_runs(self, method):
        dataset = image_dataset(constant_frames([5, 5, 200]))
        assert len(emit(SequenceFrames(dataset, Redundancy(radius=0, method=method)))) >= 1

    def test_rejects_a_bad_method(self):
        with pytest.raises(ValueError, match="method must be one of"):
            Redundancy(method="sha256")

    def test_rejects_a_negative_radius(self):
        with pytest.raises(ValueError, match="non-negative"):
            Redundancy(radius=-1)

    def test_round_trips_through_frame_indices(self):
        """A selection is reproducible, which is what makes it reviewable."""
        dataset = image_dataset(constant_frames([5, 5, 5, 90, 90, 200]))
        chosen = SequenceFrames(dataset, Redundancy(radius=0, method="xxhash")).frame_map
        replayed = SequenceFrames(dataset, FrameIndices({0: chosen[:, 1].tolist()})).frame_map
        np.testing.assert_array_equal(chosen, replayed)

    def test_anchors_on_the_last_kept_frame_not_the_predecessor(self):
        """The distinction that separates selecting key frames from measuring redundancy.

        Anchoring on the predecessor might keep a frame that should be dropped,
        because nothing new has accumulated since the last thing recorded.
        """
        from dataeval.core import hamming_distance, phash

        images = drifting_frames()
        digests = [phash(image) for image in images]

        # Dynamically find a triplet of frames (anchor, dropped1, dropped2)
        # where dropped2 is close enough to anchor to be dropped, but far enough
        # from dropped1 that if the selector anchored on predecessors, it would be kept.
        triplet = None
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                for k in range(j + 1, len(images)):
                    d_ij = hamming_distance(digests[i], digests[j])
                    d_ik = hamming_distance(digests[i], digests[k])
                    d_jk = hamming_distance(digests[j], digests[k])

                    radius = max(d_ij, d_ik)
                    if d_jk > radius:
                        triplet = (i, j, k, radius)
                        break
                if triplet:
                    break
            if triplet:
                break

        assert triplet is not None, "Could not find a suitable triplet in the fixture"
        i, j, k, radius = triplet

        test_images = [images[i], images[j], images[k]]
        kept = [m["frame"] for _, _, m in emit(SequenceFrames(image_dataset(test_images), Redundancy(radius, "phash")))]

        # Frame 0 is kept, Frame 1 is dropped (dist <= radius), Frame 2 is dropped (dist <= radius to 0)
        assert kept == [0]


@pytest.mark.required
class TestAFrameKeptWithoutEvidenceIsMarked:
    def test_frame_rate_marks_untimed_frames(self):
        dataset, _ = make_dataset((4,), timed=False)
        frames = SequenceFrames(dataset, FrameRate(10.0))
        entries = frames._record_of(0).entries
        assert [e["unmeasured"] for e in entries] == [True, True, True, True]

    def test_frame_rate_marks_nothing_on_timed_frames(self):
        dataset, _ = make_dataset((4,))
        assert not any(e["unmeasured"] for e in SequenceFrames(dataset, FrameRate(10.0))._record_of(0).entries)

    def test_redundancy_marks_a_frame_it_has_no_digest_for(self):
        """A frame too small to hash has no digest, and the next one, which hashes, is kept as measured."""
        dataset = image_dataset([np.zeros((3, 4, 4), dtype=np.uint8), *constant_frames([90, 90])])
        frames = SequenceFrames(dataset, Redundancy(radius=0, method="phash"))
        entries = frames._record_of(0).entries
        assert [e["position"] for e in entries] == [0, 1]
        assert [e["unmeasured"] for e in entries] == [True, False]

    def test_the_view_lists_which_frames_were_kept_unmeasured(self):
        untimed, _ = make_dataset((4, 3), timed=False)
        frames = SequenceFrames(untimed, FrameRate(10.0))
        assert frames.unmeasured.tolist() == [True] * len(frames.frame_map)
        timed, _ = make_dataset((4, 3))
        frames = SequenceFrames(timed, FrameRate(10.0))
        assert frames.unmeasured.tolist() == [False] * len(frames.frame_map)

    def test_a_planned_view_marks_nothing_without_decoding(self):
        dataset, counters = make_dataset((6, 4))
        assert SequenceFrames(dataset, Stride(2)).unmeasured.tolist() == [False] * 5
        assert [c["iterations"] for c in counters] == [0, 0]

    def test_the_flag_is_not_a_datum_factor(self):
        dataset, _ = make_dataset((4,), timed=False)
        assert "unmeasured" not in emit(SequenceFrames(dataset, FrameRate(10.0)))[0][2]

    def test_a_replay_of_a_recorded_sequence_keeps_the_flags(self):
        dataset, counters = make_dataset((4,), timed=False)
        frames = SequenceFrames(dataset, FrameRate(10.0))
        list(frames.stream())  # records the sequence
        replayed = [entry["unmeasured"] for entry, _, _ in frames._walk(frames._sequences[0])]
        assert counters[0]["iterations"] == 2  # the second walk replayed the record
        assert replayed == [True, True, True, True]

    def test_a_selector_that_marks_the_same_frames_differently_on_two_walks_is_refused(self):
        dataset, _ = make_dataset((6, 4))
        frames = SequenceFrames(dataset, _FickleEvidence())
        stream = frames.stream()
        next(stream)  # paused inside sequence 0
        _ = frames.frame_map  # records every sequence with the second call's flags
        with pytest.raises(ValueError, match="chose different frames on two walks of sequence 0"):
            list(stream)


class _Medoid(FrameSelector):
    """One representative per sequence, standing for frames it is not adjacent to."""

    two_pass = True

    def select(self, frames):
        positions = [frame.position for frame in frames]
        return iter([FrameVerdict(positions[0], weight=float(len(positions)))])


class _Rewrites(FrameSelector):
    """A selector that does more than choose, and says so."""

    invalidates = ImageStats.DIMENSION_WIDTH

    def select(self, frames):
        return (FrameVerdict(frame.position) for frame in frames)


class _OverrunningPlan(FrameSelector):
    """A plan naming positions the sequence does not have, out of order and repeated."""

    def plan(self, info):
        return np.array([2, 0, 2, 99], dtype=np.intp)

    def select(self, frames):
        return (FrameVerdict(frame.position) for frame in frames if frame.position in (0, 2))


@pytest.mark.required
class TestSelectorDeclarations:
    """What a selector declares has to reach the view, or the declaration is decoration."""

    def test_a_declared_weight_overrides_the_gap_to_the_next_kept_frame(self):
        dataset, _ = make_dataset((6,))
        emitted = emit(SequenceFrames(dataset, _Medoid()))
        assert [meta["frames_represented"] for _, _, meta in emitted] == [6.0]

    def test_a_selector_that_rewrites_content_invalidates_through_the_view(self):
        from dataeval.data._invalidates import invalidated_stats

        dataset, _ = make_dataset((4,))
        assert invalidated_stats(SequenceFrames(dataset, Stride(2))) == ImageStats.NONE
        assert invalidated_stats(SequenceFrames(dataset, _Rewrites())) == ImageStats.DIMENSION_WIDTH

    def test_a_plan_is_reduced_to_what_the_walk_will_actually_emit(self):
        """Otherwise __len__ and frame_map promise frames the replay never yields."""
        dataset, _ = make_dataset((4,))
        frames = SequenceFrames(dataset, _OverrunningPlan())
        assert len(frames) == 2
        assert frames.frame_map.tolist() == [[0, 0], [0, 2]]
        assert [meta["frame"] for _, _, meta in emit(frames)] == [0, 2]


@pytest.mark.required
class TestDecodeCost:
    def test_locating_frames_costs_no_decode_when_the_selector_plans(self):
        dataset, counters = make_dataset((6, 4))
        frames = SequenceFrames(dataset, Stride(2))
        assert frames.frame_map.tolist() == [[0, 0], [0, 2], [0, 4], [1, 0], [1, 2]]
        assert frames.sequence_offsets.tolist() == [0, 3, 5]
        assert [counter["iterations"] for counter in counters] == [0, 0]

    def test_getitem_reads_the_target_off_the_cursor(self):
        """The walk already pairs frame with target; re-reading it indexes the source per frame."""
        dataset, _ = make_dataset((5,))
        frames = SequenceFrames(dataset, Stride(2))
        reads = []
        original = type(dataset).__getitem__
        type(dataset).__getitem__ = lambda self, index: (reads.append(index), original(self, index))[1]
        try:
            frames[0]
            after_first = len(reads)
            frames[1]
            frames[2]
        finally:
            type(dataset).__getitem__ = original
        # Reads are per sequence walked, not per frame returned.
        assert len(reads) == after_first


@pytest.mark.required
class TestTrackMap:
    """Which track each presented detection belongs to, read from the targets."""

    def test_one_row_per_detection_in_measurement_order(self):
        frames = SequenceFrames(make_dataset((3, 2))[0])
        assert frames.track_map.tolist() == [
            [0, 0], [0, 1],
            [1, 0], [1, 1],
            [2, 0], [2, 1],
            [3, 0], [3, 1],
            [4, 0], [4, 1],
        ]  # fmt: skip

    def test_it_follows_the_selection_rather_than_the_source(self):
        frames = SequenceFrames(make_dataset((6,))[0], Stride(2))
        assert frames.track_map[:, 0].tolist() == [0, 0, 1, 1, 2, 2]

    def test_reading_it_decodes_nothing(self):
        """Track ids live in the targets, which cost no decode -- the same source as frame counts."""
        dataset, counters = make_dataset((6, 4))
        assert len(SequenceFrames(dataset, Stride(2)).track_map) == 10
        assert [counter["frames"] for counter in counters] == [0, 0]

    def test_an_unlinked_detection_carries_its_own_marker_through(self):
        dataset, _ = make_dataset((2,))
        for target in dataset[0][1].frame_tracks:
            target.track_ids = np.array([-1, 3], dtype=np.int64)
        assert SequenceFrames(dataset).track_map[:, 1].tolist() == [-1, 3, -1, 3]

    def test_a_frame_with_no_detections_takes_no_rows(self):
        dataset, _ = make_dataset((3,))
        dataset[0][1].frame_tracks[1].track_ids = np.empty(0, dtype=np.int64)
        assert SequenceFrames(dataset).track_map[:, 0].tolist() == [0, 0, 2, 2]

    def test_an_empty_view_has_an_empty_map(self):
        frames = SequenceFrames(make_dataset((4,))[0], FrameIndices({0: []}))
        assert frames.frame_map.shape == (0, 2)
        assert frames.track_map.shape == (0, 2)


@pytest.mark.required
class TestTheSelectionIsFixedByTheFirstWalk:
    def test_a_stateful_selector_is_consulted_once_per_sequence(self):
        dataset, _ = make_dataset((6, 4))
        selector = _Fickle()
        frames = SequenceFrames(dataset, selector)
        first = [m["frame"] for _, _, m in emit(frames)]
        second = [m["frame"] for _, _, m in emit(frames)]
        assert first == second
        assert frames.frame_map[:, 1].tolist() == first
        assert selector.calls == 2

    def test_a_two_pass_selector_pays_its_second_pass_once(self):
        dataset, counters = make_dataset((6, 4))
        frames = SequenceFrames(dataset, _Buffering(two_pass=True))
        list(frames.stream())
        assert [c["iterations"] for c in counters] == [2, 2]
        list(frames.stream())
        assert [c["iterations"] for c in counters] == [3, 3]

    def test_reading_the_map_after_streaming_walks_nothing(self):
        dataset, counters = make_dataset((6, 4))
        frames = SequenceFrames(dataset, _EveryOther())
        list(frames.stream())
        walked = [c["iterations"] for c in counters]
        _ = frames.frame_map, len(frames), frames.n_dropped
        assert [c["iterations"] for c in counters] == walked
        assert frames.n_dropped == 5

    def test_stopping_right_after_the_last_frame_still_records(self):
        dataset, counters = make_dataset((6, 4))
        frames = SequenceFrames(dataset, _EveryOther())
        stream = frames.stream()
        _ = [next(stream) for _ in range(5)]  # 3 + 2 kept, never pulled past the last
        walked = [c["iterations"] for c in counters]
        _ = frames.frame_map
        assert [c["iterations"] for c in counters] == walked

    def test_an_abandoned_walk_records_only_finished_sequences(self):
        dataset, counters = make_dataset((6, 4))
        frames = SequenceFrames(dataset, _EveryOther())
        stream = frames.stream()
        next(stream)
        _ = frames.frame_map
        assert counters[0]["iterations"] == 2
        assert counters[1]["iterations"] == 1

    def test_random_access_after_a_partial_stream_matches_the_stream(self):
        dataset, _ = make_dataset((6, 4))
        frames = SequenceFrames(dataset, _EveryOther())
        streamed = list(frames.stream())
        partial = SequenceFrames(dataset, _EveryOther())
        next(partial.stream())
        assert {k: v for k, v in partial[3][2].items() if k != "id"} == {
            k: v for k, v in streamed[3][2].items() if k != "id"
        }

    def test_frames_without_a_frame_index_replay(self):
        """Frames carrying pixels but no frame_index are numbered by position on both walks."""
        frames = SequenceFrames(unnumbered_dataset(5), _EveryOther())
        first = [m for _, _, m in emit(frames)]
        assert [m["frame"] for m in first] == [0, 2, 4]
        assert [m for _, _, m in frames.stream()] == first

    def test_a_selector_that_stops_reading_early_still_replays(self):
        """The record numbers every frame, so the first walk reads past where the selector stopped."""
        dataset, _ = make_dataset((6,))
        frames = SequenceFrames(dataset, _FirstTwo())
        first = [m for _, _, m in emit(frames)]
        assert [m["frame"] for m in first] == [0, 1]
        assert [m for _, _, m in frames.stream()] == first
        assert frames._record_of(0).frame_numbers.tolist() == list(range(6))


@pytest.mark.required
class TestTheReplayContract:
    def test_a_source_that_changes_between_walks_is_refused(self):
        dataset, _ = make_dataset((4,), numbering=lambda i: i)
        frames = SequenceFrames(dataset, _EveryOther())
        list(frames.stream())
        dataset[0][0]._numbering = lambda i: i + 50  # the source now numbers its frames differently
        with pytest.raises(ValueError, match="changed between walks"):
            list(frames.stream())

    def test_a_selector_that_answers_differently_on_two_walks_of_a_sequence_is_refused(self):
        dataset, _ = make_dataset((6, 4))
        frames = SequenceFrames(dataset, _Fickle())
        stream = frames.stream()
        next(stream)  # paused inside sequence 0
        _ = frames.frame_map  # walks and records every sequence while the stream is paused
        with pytest.raises(ValueError, match="chose different frames on two walks of sequence 0"):
            list(stream)

    def test_a_selector_that_labels_the_same_frames_differently_on_two_walks_is_refused(self):
        dataset, _ = make_dataset((6, 4))
        frames = SequenceFrames(dataset, _FickleFactors())
        stream = frames.stream()
        next(stream)  # paused inside sequence 0
        _ = frames.frame_map  # records every sequence with the second call's factors
        with pytest.raises(ValueError, match="chose different frames on two walks of sequence 0"):
            list(stream)

    def test_a_source_that_renumbers_between_two_passes_is_refused(self):
        counters: list[dict] = []
        # Each iteration of the stream numbers its frames differently.
        dataset, _ = make_dataset((4,), counters=counters, numbering=lambda i: i + 50 * counters[0]["iterations"])
        with pytest.raises(ValueError, match="changed between walks"):
            list(SequenceFrames(dataset, _Buffering(two_pass=True)).stream())

    def test_a_one_shot_stream_names_the_contract(self):
        dataset, _ = make_dataset((4,))
        stream, target, meta = dataset[0]
        one_shot = iter(list(stream))
        dataset._data[0] = (one_shot, target, meta)  # every dataset[0] now hands back the same iterator
        with pytest.raises(ValueError, match="already consumed"):
            list(SequenceFrames(dataset, _Buffering(two_pass=True)).stream())

    def test_a_stream_spent_by_another_reader_names_the_contract(self):
        dataset, _ = make_dataset((4,))
        stream, target, meta = dataset[0]
        one_shot = iter(list(stream))
        dataset._data[0] = (one_shot, target, meta)
        list(one_shot)  # another reader, such as Metadata, got there first
        with pytest.raises(ValueError, match="already consumed"):
            list(SequenceFrames(dataset, _EveryOther()).stream())


# --------------------------------------------------------------------------------------
# View transform provenance: SequenceFrames sits between a tracking dataset and a View,
# so a frame's address has to survive being renumbered above it and below it.
# --------------------------------------------------------------------------------------


def make_frame_target(boxes: Sequence[Sequence[float]], track_ids: Sequence[int]):
    """One frame's detections, with the boxes and track ids given."""
    return cast(
        SingleFrameObjectTrackingTarget,
        _FakeFrameTarget(
            track_ids=np.asarray(track_ids, dtype=np.int64),
            boxes=np.asarray(boxes, dtype=np.float32),
            scores=np.ones(len(boxes), dtype=np.float32),
            labels=np.zeros(len(boxes), dtype=np.int64),
        ),
    )


def make_dataset_of(frame_targets: Sequence[Sequence[Any]], shape=(3, 12, 14)):
    """A tracking dataset whose frames carry the per-frame targets given, sequence by sequence."""
    data: list[MultiobjectTrackingDatum] = []
    for seq, targets in enumerate(frame_targets):
        counter = {"iterations": 0, "frames": 0, "pixels": 0}
        stream = _CountingStream(len(targets), shape, counter, timed=True, index=seq)
        data.append((
            cast(Any, stream),
            cast(Any, _FakeVideoTarget(frame_tracks=list(targets))),
            cast(DatumMetadata, {"id": f"vid{seq}", "height": shape[1], "width": shape[2]}),
        ))
    return _FakeDataset(data, DatasetMetadata({"id": "videos", "index2label": {0: "thing"}}))


class _Blurs(Operation):
    """An operation that declares an invalidation without touching the datum.

    A tracking datum's image is a stream rather than a raster, so the shipped geometric
    operations cannot run below the frame view. What is under test here is whether the
    invalidation walk crosses the frame view at all, which needs only the declaration.
    """

    @property
    def invalidates(self) -> ImageStats:
        return ImageStats.VISUAL_SHARPNESS

    def apply(self, view: View) -> None:  # pragma: no cover - selection is unchanged
        pass


@pytest.mark.required
class TestProvenanceThroughAViewAboveTheFrameView:
    """``View(SequenceFrames(...))`` -- the frame view is the object-detection dataset a View reorders."""

    def test_a_reordering_view_keeps_every_frame_with_its_own_video(self):
        dataset, _ = make_dataset((4, 3))
        view = View(SequenceFrames(dataset), Reverse())
        got = [cast(dict[str, Any], view[index][2]) for index in range(len(view))]
        assert [meta["source_id"] for meta in got] == ["vid1"] * 3 + ["vid0"] * 4
        assert [meta["sequence"] for meta in got] == [1, 1, 1, 0, 0, 0, 0]
        assert [meta["frame"] for meta in got] == [2, 1, 0, 3, 2, 1, 0]

    def test_a_filtering_view_keeps_every_frame_with_its_own_video(self):
        dataset, _ = make_dataset((4, 3))
        view = View(SequenceFrames(dataset), Indices([5, 1]))
        got = [cast(dict[str, Any], view[index][2]) for index in range(len(view))]
        assert [(meta["source_id"], meta["frame"]) for meta in got] == [("vid1", 1), ("vid0", 1)]

    def test_the_datum_id_stays_the_frame_views_position_where_the_address_is_the_views(self):
        """Two numberings meet here, exactly as they do for `DetectionCrops`.

        A statistic measured over the view is addressed by *view* position; the datum's own
        ``id`` is its position in the frame view underneath. `View.resolve_indices` is the
        documented way between them, and `frame_map` turns the result back into a
        ``(sequence, frame)`` pair.
        """
        dataset, _ = make_dataset((4, 3))
        frames = SequenceFrames(dataset)
        view = View(frames, Reverse())
        assert cast(dict[str, Any], view[0][2])["id"] == 6
        assert view.resolve_indices(0) == [6]
        assert frames.frame_map[6].tolist() == [1, 2]

    def test_resolving_every_view_position_reproduces_the_frames_it_presents(self):
        dataset, _ = make_dataset((5, 4))
        frames = SequenceFrames(dataset, Stride(2))
        view = View(frames, Indices([3, 0, 2]))
        resolved = view.resolve_indices()
        assert frames.frame_map[resolved].tolist() == [[1, 0], [0, 0], [0, 4]]
        assert [cast(dict[str, Any], view[i][2])["frame"] for i in range(len(view))] == [0, 0, 4]

    def test_a_geometric_transform_carries_track_ids_along_with_the_boxes(self):
        """A crop that drops an out-of-frame detection must drop its track id in step.

        The frame's target is passed through whole so ``track_ids`` stays reachable; a
        transform that masks ``boxes`` without masking ``track_ids`` would leave the two
        arrays different lengths and silently mislabel which object is which.
        """
        target = make_frame_target([[0.0, 0.0, 4.0, 4.0], [10.0, 8.0, 14.0, 12.0]], [7, 9])
        dataset = make_dataset_of([[target, target]])
        view = View(SequenceFrames(dataset), Crop((0, 0, 6, 6)))
        _, cropped, _ = view[0]
        assert np.asarray(cropped.boxes).shape == (1, 4)
        assert np.asarray(cast(Any, cropped).track_ids).tolist() == [7]

    def test_a_transform_that_drops_no_detection_leaves_the_track_ids_whole(self):
        target = make_frame_target([[0.0, 0.0, 4.0, 4.0], [1.0, 1.0, 5.0, 5.0]], [7, 9])
        dataset = make_dataset_of([[target]])
        view = View(SequenceFrames(dataset), Crop((0, 0, 6, 6)))
        _, cropped, _ = view[0]
        assert np.asarray(cast(Any, cropped).track_ids).tolist() == [7, 9]

    def test_track_map_reports_the_source_detections_not_the_ones_a_transform_left(self):
        """`track_map` is read off the source targets, so a transform above it is invisible.

        This pins a real limit rather than a desired behavior: a caller that crops at the
        view level and then links per-detection statistics through `frames.track_map` gets
        a map with more rows than the view has detections. Linking has to be done against
        the targets the view actually yields.
        """
        target = make_frame_target([[0.0, 0.0, 4.0, 4.0], [10.0, 8.0, 14.0, 12.0]], [7, 9])
        dataset = make_dataset_of([[target, target]])
        frames = SequenceFrames(dataset)
        view = View(frames, Crop((0, 0, 6, 6)))
        presented = sum(len(np.asarray(view[index][1].boxes)) for index in range(len(view)))
        assert presented == 2
        assert len(frames.track_map) == 4

    def test_an_invalidating_operation_above_the_frame_view_is_seen(self):
        from dataeval.data._invalidates import invalidated_stats

        dataset, _ = make_dataset((4,))
        view = View(SequenceFrames(dataset), Crop((0, 0, 6, 6)))
        assert invalidated_stats(view) & ImageStats.DIMENSION_WIDTH


@pytest.mark.required
class TestProvenanceThroughAViewBelowTheFrameView:
    """``SequenceFrames(View(...))`` -- the tracking dataset has already been renumbered."""

    def test_a_filtered_source_is_named_by_id_where_its_position_has_moved(self):
        """`sequence` is the position in what was handed in; `source_id` is which video it is."""
        dataset, _ = make_dataset((4, 3))
        frames = SequenceFrames(View(dataset, Indices([1])))
        emitted = emit(frames)
        assert len(emitted) == 3
        assert {meta["sequence"] for _, _, meta in emitted} == {0}
        assert {meta["source_id"] for _, _, meta in emitted} == {"vid1"}
        assert [meta["frame"] for _, _, meta in emitted] == [0, 1, 2]

    def test_a_reordered_source_keeps_every_frame_with_its_own_video(self):
        dataset, _ = make_dataset((4, 3))
        emitted = emit(SequenceFrames(View(dataset, Reverse())))
        assert [meta["source_id"] for _, _, meta in emitted] == ["vid1"] * 3 + ["vid0"] * 4
        assert [meta["sequence"] for _, _, meta in emitted] == [0] * 3 + [1] * 4

    def test_frame_map_addresses_the_view_it_was_built_over_not_the_dataset_under_it(self):
        """`frame_map`'s first column indexes what SequenceFrames was handed, and says so."""
        dataset, _ = make_dataset((4, 3))
        below = View(dataset, Indices([1]))
        frames = SequenceFrames(below)
        np.testing.assert_array_equal(frames.frame_map, [[0, 0], [0, 1], [0, 2]])
        assert below.resolve_indices(0) == [1]

    def test_an_invalidating_operation_below_the_frame_view_stays_visible(self):
        """The walk crosses the frame view through `source`, as it does for `DetectionCrops`."""
        from dataeval.data._invalidates import invalidated_stats

        dataset, _ = make_dataset((4,))
        frames = SequenceFrames(View(dataset, _Blurs()))
        assert invalidated_stats(View(frames, Reverse())) == ImageStats.VISUAL_SHARPNESS

    def test_the_chain_is_walkable_through_one_public_attribute(self):
        dataset, _ = make_dataset((4,))
        below = View(dataset, Reverse())
        view = View(SequenceFrames(below), Reverse())
        chain = []
        node: Any = view
        while node is not None:
            chain.append(type(node).__name__)
            node = getattr(node, "source", None)
        assert chain == ["View", "SequenceFrames", "View", "_FakeDataset"]


@pytest.mark.required
class TestAddressingTheFrameView:
    """`SourceLocator` over the frame view -- what an evaluator's address resolves to."""

    def test_the_frame_view_addresses_as_an_image_task(self):
        dataset, _ = make_dataset((4, 3))
        locator = SourceLocator(SequenceFrames(dataset))
        assert locator.item_level == "unit"
        assert locator.levels == ("unit", "instance")

    def test_an_unkeyed_address_names_one_frame(self):
        dataset, _ = make_dataset((4, 3))
        frames = SequenceFrames(dataset)
        found = SourceLocator(frames)[SourceIndex(4)]
        assert found.level == "unit"
        np.testing.assert_array_equal(found.pixels, frames[4][0])

    def test_a_keyed_address_names_one_detection_in_one_frame(self):
        dataset, _ = make_dataset((4, 3))
        frames = SequenceFrames(dataset)
        found = SourceLocator(frames)[SourceIndex(4, 1)]
        assert found.level == "instance"
        np.testing.assert_array_equal(found.box, np.asarray(frames[4][1].boxes)[1])

    def test_the_reframing_costs_the_sequence_and_track_levels(self):
        """The source has four levels and the frame view presents two.

        A track is still reachable through the target's ``track_ids``, but it is no longer
        addressable: an address measured over the frame view cannot name one.
        """
        dataset, _ = make_dataset((4, 3))
        locator = SourceLocator(SequenceFrames(dataset))
        with pytest.raises(ValueError, match="'track'-level data, but this dataset's levels"):
            locator[SourceIndex(0, 0, "track")]
        with pytest.raises(ValueError, match="'unit', 'instance'"):
            locator[SourceIndex(0, None, "sequence")]

    def test_the_source_position_undoes_a_view_above_the_frame_view(self):
        """The round trip the frame view has to support: view position to (sequence, frame)."""
        dataset, _ = make_dataset((4, 3))
        frames = SequenceFrames(dataset)
        found = SourceLocator(View(frames, Reverse()))[SourceIndex(0)]
        assert found.source_item_index == 6
        assert frames.frame_map[found.source_item_index].tolist() == [1, 2]

    def test_the_untransformed_source_behind_a_frame_view_is_the_frame_view(self):
        """`View.root` stops at the frame view, which is the dataset the view renumbers."""
        dataset, _ = make_dataset((4, 3))
        frames = SequenceFrames(dataset)
        assert SourceLocator(View(frames, Reverse())).source is frames

    def test_a_transform_above_the_frame_view_shows_up_between_the_two_reads(self):
        dataset, _ = make_dataset((4,))
        frames = SequenceFrames(dataset)
        found = SourceLocator(View(frames, Crop((0, 0, 6, 6))))[SourceIndex(1)]
        assert found.pixels.shape == (3, 6, 6)
        assert found.source_pixels.shape == (3, 12, 14)

    def test_an_address_gathered_in_batch_stays_with_its_own_frame(self):
        dataset, _ = make_dataset((4, 3))
        frames = SequenceFrames(dataset)
        locator = SourceLocator(frames)
        found = locator.gather([SourceIndex(0), SourceIndex(4), SourceIndex(6)])
        assert [item.item_index for item in found] == [0, 4, 6]
        for item in found:
            np.testing.assert_array_equal(item.pixels, frames[item.item_index][0])

    def test_a_frame_is_addressed_flat_here_and_keyed_against_the_source(self):
        """The reframing moves a frame between two address spaces, and both are in use.

        Over the frame view a frame is an *item*: `SourceIndex(4)`, flat across every
        sequence. Over the tracking dataset underneath it is a keyed `unit` row of one
        sequence: `SourceIndex(1, 0, "unit")`. `Duplicates` reports frame duplicates in the
        second space (it reads `frame_map`) while `Outliers` and `image_stats` address the
        first, so which locator resolves a finding depends on which produced it.
        """
        dataset, _ = make_dataset((4, 3))
        frames = SequenceFrames(dataset)
        with pytest.raises(ValueError, match="carries a key at 'unit' level"):
            SourceLocator(frames)[SourceIndex(1, 0, "unit")]
        np.testing.assert_array_equal(
            SourceLocator(frames)[SourceIndex(4)].pixels,
            SourceLocator(dataset)[SourceIndex(1, 0, "unit")].pixels,
        )
        assert frames.frame_map[4].tolist() == [1, 0]

    def test_frame_refuses_a_level_that_spans_frames(self):
        dataset, _ = make_dataset((4, 3))
        locator = SourceLocator(dataset)
        with pytest.raises(TypeError, match="spans frames"):
            locator.frame(locator[SourceIndex(0, None, "sequence")])
        with pytest.raises(TypeError, match="spans frames"):
            locator.frame(locator[SourceIndex(0, 0, "track")])


@pytest.mark.required
def test_representative_on_an_empty_sequence_yields_nothing():
    """No frames, no sequence: the walk finds nothing to be representative of."""
    assert list(Representative(3, batch_size=2).select(iter([]))) == []


@pytest.mark.required
class TestTranslatingToTheSource:
    def test_source_indices_name_kept_frames_by_their_own_number(self):
        dataset, _ = make_dataset((6, 4), numbering=lambda i: 100 + 3 * i)
        frames = SequenceFrames(dataset, Stride(2))
        assert frames.source_indices[:4] == (
            SourceIndex(0, 100, "unit"),
            SourceIndex(0, 106, "unit"),
            SourceIndex(0, 112, "unit"),
            SourceIndex(1, 100, "unit"),
        )

    def test_frames_and_detections(self):
        dataset, _ = make_dataset((6, 4), numbering=lambda i: 100 + 3 * i)  # 2 detections per frame
        frames = SequenceFrames(dataset, Stride(2))
        assert frames.to_source([SourceIndex(1), SourceIndex(1, 1)]) == [
            SourceIndex(0, 106, "unit"),
            SourceIndex(0, 5, "instance"),
        ]

    def test_a_level_stated_address_is_refused(self):
        dataset, _ = make_dataset((6, 4))
        with pytest.raises(ValueError, match="already names a level"):
            SequenceFrames(dataset).to_source([SourceIndex(0, 1, "unit")])

    def test_a_detection_the_frame_lacks_is_refused(self):
        dataset, _ = make_dataset((6, 4))
        with pytest.raises(ValueError, match=r"has 2 detection\(s\)"):
            SequenceFrames(dataset).to_source([SourceIndex(0, 2)])

    def test_an_ambiguous_frame_number_is_refused(self):
        dataset, _ = make_dataset((6,), numbering=lambda i: i // 2)
        with pytest.raises(ValueError, match="ambiguous"):
            SequenceFrames(dataset).to_source([SourceIndex(0)])

    def test_a_negative_address_is_refused(self):
        dataset, _ = make_dataset((6, 4))
        with pytest.raises(IndexError, match="-1"):
            SequenceFrames(dataset).to_source([SourceIndex(-1)])

    def test_translating_after_streaming_walks_nothing(self):
        dataset, counters = make_dataset((6, 4))
        frames = SequenceFrames(dataset, _EveryOther())
        list(frames.stream())
        walked = [c["iterations"] for c in counters]
        frames.to_source([SourceIndex(0), SourceIndex(4, 1)])
        assert [c["iterations"] for c in counters] == walked

    def test_an_unwalked_view_walks_once_and_says_so(self, caplog):
        dataset, counters = make_dataset((6, 4))
        frames = SequenceFrames(dataset, Stride(2))
        with caplog.at_level(logging.INFO):
            frames.to_source([SourceIndex(0)])
        assert "walking" in caplog.text
        assert [c["iterations"] for c in counters] == [1, 0]  # only the sequence the address names

    def test_a_stateful_selector_translates_the_frames_it_streamed(self):
        dataset, _ = make_dataset((6, 4), numbering=lambda i: 100 + i)
        frames = SequenceFrames(dataset, _Fickle())
        streamed = [cast(dict[str, Any], m) for _, _, m in frames.stream()]
        got = frames.to_source([SourceIndex(i) for i in range(len(streamed))])
        assert got == [SourceIndex(m["sequence"], m["frame"], "unit") for m in streamed]

    def test_a_selector_that_keeps_nothing(self):
        from dataeval.quality import Duplicates

        dataset, _ = make_dataset((6, 4))
        frames = SequenceFrames(dataset, _KeepsNothing())
        assert frames.source_indices == ()
        assert frames.to_source([]) == []
        result = Duplicates(flags=ImageStats.HASH_XXHASH).evaluate(frames, levels="unit")
        assert result.exact == []

    def test_empty_and_zero_frame(self):
        dataset, _ = make_dataset((0, 3))
        frames = SequenceFrames(dataset, _EveryOther())
        assert frames.to_source([]) == []
        assert frames.source_indices == (SourceIndex(1, 0, "unit"), SourceIndex(1, 2, "unit"))


@pytest.mark.required
class TestFindingsLandOnTheSourceRows:
    """The design's headline: a thinned, non-planning view, frame numbers that are not positions."""

    def _setup(self):
        dataset, _ = make_dataset((6, 4), numbering=lambda i: 100 + 3 * i)
        return dataset, SequenceFrames(dataset, _EveryOther()), Metadata(dataset)

    def test_frame_values(self):
        _, frames, md = self._setup()
        n = len(frames)
        md.add_factors(
            {"rating": np.arange(n, dtype=float)}, source_index=frames.to_source([SourceIndex(i) for i in range(n)])
        )
        rows = md.rows_at("unit")
        got = dict(zip(zip(rows["item_index"], rows["unit_index"], strict=True), rows["rating"], strict=True))
        assert got[(0, 100)] == 0.0
        assert got[(0, 106)] == 1.0
        assert got[(1, 100)] == 3.0
        assert got[(0, 103)] is None

    def test_detection_values(self):
        _, frames, md = self._setup()
        addresses = [SourceIndex(1, 0), SourceIndex(1, 1)]  # view frame 1 is source position 2
        md.add_factors({"iou": [0.3, 0.4]}, source_index=frames.to_source(addresses))
        rows = md.rows_at("instance")
        got = dict(zip(zip(rows["item_index"], rows["target_index"], strict=True), rows["iou"], strict=True))
        assert got[(0, 4)] == 0.3
        assert got[(0, 5)] == 0.4
        assert sum(v is not None for v in got.values()) == 2

    def test_detection_values_with_uneven_counts_per_frame(self):
        """Detections are numbered by a running count across a sequence's frames, not by a stride."""
        counts = [[3, 0, 1, 4, 2], [1, 2]]
        dataset = make_dataset_of([[make_target(n) for n in sequence] for sequence in counts])
        frames = SequenceFrames(dataset, _EveryOther())  # keeps positions 0, 2 and 4, then 0
        md = Metadata(dataset)
        # View frame 1 is position 2 (one detection), 2 is position 4 (two), and 3 is sequence 1's first.
        addresses = [SourceIndex(1, 0), SourceIndex(2, 0), SourceIndex(2, 1), SourceIndex(3, 0)]
        md.add_factors({"iou": [0.1, 0.2, 0.3, 0.4]}, source_index=frames.to_source(addresses))
        rows = md.rows_at("instance")
        got = dict(zip(zip(rows["item_index"], rows["target_index"], strict=True), rows["iou"], strict=True))
        landed = {key: v for key, v in got.items() if v is not None}
        assert landed == {(0, 3): 0.1, (0, 8): 0.2, (0, 9): 0.3, (1, 0): 0.4}

    def test_the_documented_image_stats_recipe(self):
        from dataeval.core import image_stats

        _, frames, md = self._setup()
        stats = image_stats(frames, stats=ImageStats.VISUAL_BRIGHTNESS)
        md.add_factors(stats, source_index=frames.to_source(stats["source_index"]))
        assert md.rows_at("unit")["unit_brightness"].is_not_null().sum() == len(frames)

    def test_selector_values_ride_back_the_same_way(self):
        _, frames, md = self._setup()
        represented = [cast(dict[str, Any], m)["frames_represented"] for _, _, m in frames.stream()]
        md.add_factors({"frames_represented": represented}, source_index=frames.source_indices)
        assert md.rows_at("unit")["frames_represented"].drop_nulls().sum() == 10


@pytest.mark.required
class TestASequenceCarriesItsTarget:
    def test_sequence_infos_fill_the_target(self):
        from dataeval.data._selectors import sequence_infos

        dataset, _ = make_dataset((3, 2))
        lengths = []
        for info, _ in sequence_infos(dataset):
            assert info.target is not None
            lengths.append(len(info.target.frame_tracks))
        assert lengths == [3, 2]

    def test_the_target_stays_out_of_repr_and_equality(self):
        """A real video's target is megabytes of annotations, and every candidate's repr embeds its sequence's."""
        from dataeval.data._selectors import sequence_infos

        dataset, _ = make_dataset((300,))
        info = next(sequence_infos(dataset))[0]
        assert repr(info) == repr(replace(info, target=None))
        assert info == replace(info, target=None)

    def test_a_projection_reads_the_survivors_own_targets(self):
        from dataeval.data._selectors import _projected

        dataset = make_dataset_of([[make_target(n) for n in (1, 2, 3, 4, 5)]])
        target = dataset[0][1]
        view = _projected(target, np.array([1, 4]))
        assert len(view.frame_tracks) == 2
        assert view.frame_tracks[0] is target.frame_tracks[1]
        assert view.frame_tracks[1] is target.frame_tracks[4]
        assert view.frame_tracks[-1] is target.frame_tracks[4]
        assert list(view.frame_tracks[1:]) == [target.frame_tracks[4]]
        with pytest.raises(IndexError):
            view.frame_tracks[2]
