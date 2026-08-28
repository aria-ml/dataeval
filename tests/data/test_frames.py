"""Tests for presenting a tracking dataset as an object-detection dataset of frames."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

from dataeval.data import (
    AllFrames,
    Crop,
    FrameIndices,
    FrameInput,
    FrameRate,
    FrameSelector,
    FrameVerdict,
    Indices,
    Operation,
    Redundancy,
    Reverse,
    SequenceFrames,
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

    def __init__(self, index: int, shape: tuple[int, ...], time_s: float | None, pts: int | None, counter: dict):
        self.frame_index = index
        self._shape = shape
        self._counter = counter
        if time_s is not None:
            self.time_s = time_s
        if pts is not None:
            self.pts = pts

    @property
    def pixels(self) -> np.ndarray:
        self._counter["pixels"] += 1
        return np.full(self._shape, self.frame_index % 251, dtype=np.uint8)


class _CountingStream:
    """A VideoStream that records how many times it is iterated and how far."""

    def __init__(self, n_frames, shape, counter, timed=True, index=0, extra=0):
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
                i,
                self._shape,
                (i / 30.0) if self._timed else None,
                (i * 1001) if self._timed else None,
                self._counter,
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


def make_dataset(frame_counts=(6, 4), shape=(3, 12, 14), timed=True, extra=0, counters=None):
    """A tracking dataset whose streams report how often they are walked."""
    counters = [] if counters is None else counters
    data: list[MultiobjectTrackingDatum] = []
    for seq, n in enumerate(frame_counts):
        counter = {"iterations": 0, "frames": 0, "pixels": 0}
        counters.append(counter)
        stream = _CountingStream(n, shape, counter, timed=timed, index=seq, extra=extra if seq == 0 else 0)
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


# --------------------------------------------------------------------------------------


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
        from dataeval.utils._validate import validate_dataset

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


def image_dataset(images):
    """A single-sequence tracking dataset replaying prepared frames."""
    data = [
        (
            cast(Any, _ImageStream(images)),
            cast(Any, _FakeVideoTarget(frame_tracks=[make_target() for _ in images])),
            cast(DatumMetadata, {"id": "vid0"}),
        )
    ]
    return _FakeDataset(data, DatasetMetadata({"id": "videos"}))


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
        assert sum(m["frames_represented"] for _, _, m in emitted) == 7

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
        assert frames.track_map.shape == (0, 2)


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

        This pins a real limit rather than a desired behaviour: a caller that crops at the
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
        second space (it reads `frame_map`) while `Outliers` and `compute_stats` address the
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
