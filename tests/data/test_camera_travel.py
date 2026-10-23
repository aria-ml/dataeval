"""Tests for CameraTravel, which keeps a frame each time the estimated camera travel reaches a set fraction of it."""

import logging
import sys
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pytest

from dataeval.core import ego_stats
from dataeval.core._ego_stats import _PairReading
from dataeval.data import AllFrames, CameraTravel, Chain, HasLabels, SequenceFrames, Stride
from dataeval.protocols import DatasetMetadata, DatumMetadata, MultiobjectTrackingDatum, VideoStream
from tests.core.test_ego_stats import (
    FakeFrame,
    _blown_out,
    _pan,
    _unreadable,
    ambiguous_stream,
    panning_stream,
    texture,
)
from tests.data.test_frames import _FakeDataset, _FakeVideoTarget, emit, make_dataset, make_target

cv2 = pytest.importorskip("cv2", reason="CameraTravel requires opencv")

pytestmark = pytest.mark.optional

_SHAPE = (1, 36, 64)  # the stubbed tests' frames: W is 64 and H is 36


def _reading(tx: float = 0.0, ty: float = 0.0, **flags: bool) -> _PairReading:
    return _PairReading(
        div=0.0,
        curl=0.0,
        shear=0.0,
        t0x=tx,
        t0y=ty,
        t0_mag=float(np.hypot(tx, ty)),
        foe_x=np.nan,
        foe_y=np.nan,
        motion="pan",
        degenerate=flags.get("degenerate", False),
        border_capture=flags.get("border_capture", False),
        ambiguous=flags.get("ambiguous", False),
        mover_frac=0.0,
        mover_speed=np.nan,
    )


def _stub(monkeypatch: pytest.MonkeyPatch, readings: Sequence[_PairReading | None]) -> None:
    """Return the next reading on each pair measured; a test that builds two views stubs again."""
    iterator = iter(readings)
    monkeypatch.setattr("dataeval.core._ego_stats._measure_pair", lambda *args, **kwargs: next(iterator))


def _dataset_of(streams: Sequence[Sequence[FakeFrame]]) -> _FakeDataset:
    data: list[MultiobjectTrackingDatum] = []
    for seq, stream in enumerate(streams):
        height, width = stream[0].pixels.shape[-2:]
        target = _FakeVideoTarget(frame_tracks=[make_target() for _ in stream])
        data.append((
            cast(Any, stream),
            cast(Any, target),
            cast(DatumMetadata, {"id": f"vid{seq}", "height": int(height), "width": int(width)}),
        ))
    return _FakeDataset(data, DatasetMetadata({"id": "videos", "index2label": {0: "thing"}}))


def _kept(frames: SequenceFrames) -> tuple[list[int], list[float], list[bool]]:
    """Sequence 0's kept source positions, the frames each stands for, and whether it was kept unmeasured."""
    rows = frames.frame_map[:, 0] == 0
    represented = [m["frames_represented"] for _, _, m in emit(frames) if m["sequence"] == 0]
    return frames.frame_map[rows, 1].tolist(), represented, frames.unmeasured[rows].tolist()


def _nine() -> SequenceFrames:
    return SequenceFrames(make_dataset((9,), shape=_SHAPE)[0], CameraTravel(0.25))


class TestTravel:  # stubbed readings
    def test_keeps_the_first_frame_then_each_frame_the_travel_reaches_the_distance(self, monkeypatch):
        _stub(monkeypatch, [_reading(4.0)] * 8)  # 4 px of 64 per pair; 0.25 is reached after 4 pairs
        positions, represented, unmeasured = _kept(_nine())
        assert positions == [0, 4, 8]
        assert represented == [4.0, 4.0, 1.0]
        assert unmeasured == [False] * 3

    def test_vertical_travel_is_a_fraction_of_the_height(self, monkeypatch):
        _stub(monkeypatch, [_reading(ty=3.0)] * 6)  # 3 px of 36; 0.25 is 9 px
        assert _kept(SequenceFrames(make_dataset((7,), shape=_SHAPE)[0], CameraTravel(0.25)))[0] == [0, 3, 6]

    def test_reversals_short_of_the_distance_cancel(self, monkeypatch):
        _stub(monkeypatch, [_reading(t) for t in [4.0, 4.0, 4.0, -4.0, -4.0, -4.0] * 2])  # peaks at 12 px, short of 16
        assert _kept(SequenceFrames(make_dataset((13,), shape=_SHAPE)[0], CameraTravel(0.25)))[0] == [0]

    def test_a_return_after_a_keep_is_kept_again(self, monkeypatch):
        # out 16 px (kept at 4), then back 16 px to the first view (kept at 8): travel restarts at each keep
        _stub(monkeypatch, [_reading(4.0)] * 4 + [_reading(-4.0)] * 4)
        assert _kept(_nine())[0] == [0, 4, 8]

    def test_the_pixels_are_summed_so_an_exact_threshold_is_reached_on_the_pair_that_reaches_it(self, monkeypatch):
        # 0.5 px per pair on a 100 px frame: 10 pairs are exactly 5 px, which is 0.05. Summing the per-pair
        # fractions instead rounds just short of it and keeps one pair late.
        _stub(monkeypatch, [_reading(0.5)] * 11)
        frames = SequenceFrames(make_dataset((12,), shape=(1, 100, 100))[0], CameraTravel(0.05))
        assert _kept(frames)[0] == [0, 10]

    def test_a_refused_pair_keeps_its_frame_unmeasured_and_restarts_the_travel(self, monkeypatch):
        # without the restart, the travel would reach 16 px at frame 5
        _stub(monkeypatch, [_reading(4.0), None, _reading(4.0), _reading(4.0), _reading(4.0), _reading(4.0)])
        positions, _, unmeasured = _kept(SequenceFrames(make_dataset((7,), shape=_SHAPE)[0], CameraTravel(0.25)))
        assert positions == [0, 2, 6]
        assert unmeasured == [False, True, False]

    @pytest.mark.parametrize(
        "refused",
        [
            None,
            _reading(4.0, ambiguous=True),
            _reading(4.0, border_capture=True),
            _reading(4.0, degenerate=True),
            _reading(np.nan),
            _reading(ty=np.inf),
        ],
    )
    def test_an_untrusted_or_non_finite_reading_is_no_reading(self, monkeypatch, refused):  # Review Focus 2
        _stub(monkeypatch, [refused])
        positions, _, unmeasured = _kept(SequenceFrames(make_dataset((2,), shape=_SHAPE)[0], CameraTravel(0.25)))
        assert positions == [0, 1]
        assert unmeasured == [False, True]

    def test_a_non_finite_reading_never_enters_the_running_sum(self, monkeypatch):
        _stub(monkeypatch, [_reading(4.0), _reading(np.nan)] + [_reading(4.0)] * 4)
        # the NaN pair restarts the travel at frame 2; 16 px then accrue by frame 6
        positions, _, unmeasured = _kept(SequenceFrames(make_dataset((7,), shape=_SHAPE)[0], CameraTravel(0.25)))
        assert positions == [0, 2, 6]
        assert unmeasured == [False, True, False]

    def test_refused_pairs_are_logged_once_per_sequence(self, monkeypatch, caplog):
        _stub(monkeypatch, [None] * 4)
        frames = SequenceFrames(make_dataset((3, 3), shape=_SHAPE)[0], CameraTravel(0.25))
        with caplog.at_level(logging.INFO, logger="dataeval"):
            len(frames)  # walks every sequence
        records = [r for r in caplog.records if r.getMessage().startswith("CameraTravel:")]
        assert [r.levelno for r in records] == [logging.INFO] * 2
        assert [r.getMessage() for r in records] == [
            f"CameraTravel: 2 of 2 frame pair(s) in sequence {k} had no trusted reading; "
            "their frames were kept unmeasured."
            for k in (0, 1)
        ]

    def test_zero_and_one_frame_videos(self, monkeypatch):  # Review Focus 5
        _stub(monkeypatch, [])
        frames = SequenceFrames(make_dataset((0, 1), shape=_SHAPE)[0], CameraTravel(0.25))
        assert frames.frame_map.tolist() == [[1, 0]]  # (sequence, position)

    def test_weights_sum_to_the_frame_count(self, monkeypatch):
        _stub(monkeypatch, [_reading(4.0)] * 8)  # the first test's video: 4 + 4 + 1
        assert sum(_kept(_nine())[1]) == 9.0


class TestMeasured:  # real optical flow, unstubbed
    @pytest.mark.parametrize("shift", [8, -8])
    def test_a_pan_is_measured_through_the_ego_engine(self, shift):
        frames = SequenceFrames(_dataset_of([panning_stream(shift=shift, n_frames=13)]), CameraTravel(0.045))
        assert _kept(frames)[0] == [0, 4, 8, 12]  # 8 of 640 px per pair; 0.045 is 28.8 px

    @pytest.mark.parametrize(
        "stream",
        [
            lambda: [FakeFrame(np.roll(texture(), 8 * k, axis=1), k, channels=3) for k in range(13)],
            lambda: _pan(8, 13, dtype=np.float32),
        ],
        ids=["rgb", "float in [0, 1]"],
    )
    def test_color_and_float_pans_keep_the_frames_a_gray_one_does(self, stream):  # Review Focus 3
        assert _kept(SequenceFrames(_dataset_of([stream()]), CameraTravel(0.045)))[0] == [0, 4, 8, 12]

    def test_a_pair_ego_stats_refuses_is_kept_unmeasured(self):
        stream = ambiguous_stream(n_frames=8)
        positions, _, unmeasured = _kept(SequenceFrames(_dataset_of([stream]), CameraTravel(0.045)))
        assert positions == list(range(8))
        assert unmeasured == [False] + [True] * 7
        trusted = ego_stats(cast("VideoStream", stream))["stats"]["ego_trusted"]
        untrusted = [k for k, ok in enumerate(trusted) if k and not ok]
        assert [p for p, u in zip(positions, unmeasured, strict=True) if u] == untrusted


class TestCuts:  # real optical flow
    def test_the_first_frame_of_a_new_shot_is_kept(self):
        # still on texture(0) for 8 frames, then a cut to a still texture(1): before the guard, only [0] was kept
        stream = [FakeFrame(texture(0), k) for k in range(8)] + [FakeFrame(texture(1), k) for k in range(8, 16)]
        positions, _, unmeasured = _kept(SequenceFrames(_dataset_of([stream]), CameraTravel(0.1)))
        assert positions == [0, 8]
        assert unmeasured == [False, True]

    def test_a_cut_at_the_first_pair_is_caught(self):  # Review Focus 4
        stream = [FakeFrame(texture(0), 0)] + [FakeFrame(texture(1), k) for k in range(1, 6)]
        positions, _, unmeasured = _kept(SequenceFrames(_dataset_of([stream]), CameraTravel(0.1)))
        assert positions == [0, 1]
        assert unmeasured == [False, True]

    def test_a_field_of_view_switch_starts_a_new_anchor(self):
        # still on texture(0), then its center at 2x: the same scene, but no motion explains the second frame
        zoomed = cv2.resize(texture(0)[90:270, 160:480], (640, 360))
        stream = [FakeFrame(texture(0), k) for k in range(6)] + [FakeFrame(zoomed, k) for k in range(6, 12)]
        positions, _, unmeasured = _kept(SequenceFrames(_dataset_of([stream]), CameraTravel(0.1)))
        assert positions == [0, 6]
        assert unmeasured == [False, True]


class TestUnreadable:  # Review Focus 1, real and unstubbed
    def test_a_change_of_frame_size_refuses_the_pair(self):
        stream = _pan(8, 3) + _pan(8, 3, size=(180, 320), start=3)
        positions, _, unmeasured = _kept(SequenceFrames(_dataset_of([stream]), CameraTravel(0.04)))
        # 16/640 < 0.04 <= 16/320
        assert positions == [0, 3, 5]
        assert unmeasured == [False, True, False]

    def test_a_change_of_size_that_processes_alike_refuses_the_pair(self):
        # frame 3 is frame 2 upscaled 2x: identical once processed, so only its original size tells them apart
        stream = _pan(8, 3)
        stream.append(FakeFrame(cv2.resize(stream[2].pixels[0], (1280, 720), interpolation=cv2.INTER_NEAREST), 3))
        positions, _, unmeasured = _kept(SequenceFrames(_dataset_of([stream]), CameraTravel(0.045)))
        assert positions == [0, 3]
        assert unmeasured == [False, True]

    def test_frames_too_small_for_optical_flow_are_kept_unmeasured(self):
        # 10x64 crashes DIS natively; Task 1's guard refuses it before DIS runs
        positions, _, unmeasured = _kept(SequenceFrames(_dataset_of([_pan(1, 4, size=(10, 64))]), CameraTravel(0.25)))
        assert positions == [0, 1, 2, 3]
        assert unmeasured == [False, True, True, True]

    def test_a_sequence_without_finite_pixels_is_kept_unmeasured(self):
        stream = [_unreadable(k) for k in range(4)]
        positions, _, unmeasured = _kept(SequenceFrames(_dataset_of([stream]), CameraTravel(0.045)))
        assert positions == [0, 1, 2, 3]
        assert unmeasured == [False, True, True, True]

    def test_an_unreadable_frame_between_readable_ones(self):
        # pairs (2,3) and (3,4) are refused, so 3 and 4 are kept unmeasured and the travel restarts at 4: 32 px at 8
        stream = _pan(8, 3, dtype=np.float32) + [_unreadable(3)] + _pan(8, 5, dtype=np.float32, start=4)
        positions, _, unmeasured = _kept(SequenceFrames(_dataset_of([stream]), CameraTravel(0.045)))
        assert positions == [0, 3, 4, 8]
        assert unmeasured == [False, True, True, False]

    def test_a_blown_out_stretch_is_kept_unmeasured(self):
        positions, _, unmeasured = _kept(SequenceFrames(_dataset_of([_blown_out(5)]), CameraTravel(0.045)))
        assert positions == [0, 1, 2, 3, 4]
        assert unmeasured == [False] + [True] * 4

    def test_a_dropout_between_two_shots_is_kept_unmeasured(self):
        # still on one shot, black at 3-6, then a pan of 8 px a frame: every pair touching black is refused, so the
        # travel restarts at 7, the pan's first frame, and reaches 28.8 px at 11
        black = np.zeros((360, 640), np.uint8)
        stream = [FakeFrame(texture(7), k) for k in range(3)] + [FakeFrame(black, k) for k in range(3, 7)]
        stream += _pan(8, 5, start=7)
        positions, _, unmeasured = _kept(SequenceFrames(_dataset_of([stream]), CameraTravel(0.045)))
        assert positions == [0, 3, 4, 5, 6, 7, 11]
        assert unmeasured == [False] + [True] * 5 + [False]


class TestComposition:
    def test_first_in_a_chain_or_behind_all_frames(self, monkeypatch):
        readings = [_reading(4.0)] * 8
        dataset = make_dataset((9,), shape=_SHAPE)[0]
        _stub(monkeypatch, readings)
        assert _kept(SequenceFrames(dataset, Chain(AllFrames(), CameraTravel(0.25))))[0] == [0, 4, 8]
        _stub(monkeypatch, readings)
        positions, represented, _ = _kept(SequenceFrames(dataset, Chain(CameraTravel(0.25), Stride(2))))
        assert positions == [0, 8]
        assert represented == [8.0, 1.0]

    def test_behind_a_stage_that_drops_frames_is_refused(self):
        with pytest.raises(ValueError, match="either side"):
            Chain(HasLabels([7]), CameraTravel(0.5))

    def test_a_nested_chain_cannot_hide_it(self):  # Review Focus 4
        with pytest.raises(ValueError, match="either side"):
            Chain(Stride(2), Chain(CameraTravel(0.5)))


class TestConstruction:
    @pytest.mark.parametrize("distance", [0.0, -0.1, np.nan, np.inf])
    def test_refuses_a_distance_that_is_not_a_positive_finite_fraction(self, distance):
        with pytest.raises(ValueError, match="distance"):
            CameraTravel(distance)

    def test_without_opencv_it_raises_what_ego_stats_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cv2", None)
        with pytest.raises(ImportError, match="opencv-python"):
            CameraTravel(0.25)

    def test_distance_is_stored_as_a_float_and_shown_in_the_repr(self):
        assert repr(CameraTravel(1)) == "CameraTravel(distance=1.0)"
