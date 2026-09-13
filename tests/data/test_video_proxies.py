"""Tests for VideoSegments and VideoStitch proxy classes."""

from typing import Any, cast

import numpy as np
import pytest

from dataeval.data._video_proxies import _FrameTracksTarget, _LazyStream, _OffsetFrameTarget, _ShiftedFrame
from dataeval.protocols import (
    MultiobjectTrackingTarget,
    SingleFrameObjectTrackingTarget,
    VideoFrame,
    _is_protocol_instance,
)
from tests.data.test_frames import _FakeFrame, make_target


def fake_frame(index: int = 3, timed: bool = True) -> _FakeFrame:
    counter = {"pixels": 0}
    return _FakeFrame(index, (3, 4, 5), index / 30.0 if timed else None, index * 1001 if timed else None, counter)


class _Held:
    """Frame stub with a fixed pixel array for memory sharing checks."""

    frame_index = 0

    def __init__(self) -> None:
        self.pixels = np.zeros((3, 4, 5))


class TestShiftedFrame:
    def test_satisfies_the_frame_protocol(self):
        assert _is_protocol_instance(_ShiftedFrame(fake_frame(), 0), VideoFrame)

    def test_index_and_shifts(self):
        proxy = _ShiftedFrame(fake_frame(index=3), 7, time_shift=-0.1, pts_shift=-3003)
        assert proxy.frame_index == 7
        assert proxy.time_s == pytest.approx(3 / 30 - 0.1)
        assert proxy.pts == 0

    def test_defaults_pass_timings_through(self):
        proxy = _ShiftedFrame(fake_frame(index=3), 0)
        assert proxy.time_s == pytest.approx(3 / 30)
        assert proxy.pts == 3003

    def test_pixels_are_read_on_demand_and_not_copied(self):
        held = _Held()
        assert np.shares_memory(_ShiftedFrame(held, 0).pixels, held.pixels)

    def test_missing_timings_read_as_absent_but_are_still_declared(self):
        proxy = _ShiftedFrame(fake_frame(timed=False), 0)
        assert getattr(proxy, "time_s", None) is None
        assert getattr(proxy, "pts", None) is None
        assert _is_protocol_instance(proxy, VideoFrame)

    def test_none_timings_read_as_absent_rather_than_raising(self):
        class _Untimed:
            frame_index = 0
            pixels = np.zeros((3, 4, 5))
            time_s = None
            pts = None

        proxy = _ShiftedFrame(_Untimed(), 0, time_shift=1.0, pts_shift=1)
        assert getattr(proxy, "time_s", None) is None
        assert getattr(proxy, "pts", None) is None

    def test_forwards_public_attributes_but_never_private_ones(self):
        frame = fake_frame()
        frame.camera = "left"  # type: ignore[attr-defined]
        proxy = _ShiftedFrame(frame, 0)
        assert proxy.camera == "left"
        with pytest.raises(AttributeError):
            _ = proxy._nothing

    def test_repr(self):
        assert "frame_index=7" in repr(_ShiftedFrame(fake_frame(), 7))


class TestOffsetFrameTarget:
    def test_satisfies_the_frame_target_protocol(self):
        assert _is_protocol_instance(_OffsetFrameTarget(make_target(2), 10), SingleFrameObjectTrackingTarget)

    def test_offsets_tracked_ids_and_keeps_untracked(self):
        target = make_target(3)
        target.track_ids = np.array([0, -1, 5])  # type: ignore[attr-defined]
        proxy = _OffsetFrameTarget(target, 10)
        np.testing.assert_array_equal(proxy.track_ids, [10, -1, 15])

    def test_other_arrays_are_the_sources_own(self):
        target = cast(Any, make_target(2))
        proxy = _OffsetFrameTarget(target, 10)
        assert proxy.boxes is target.boxes
        assert proxy.labels is target.labels
        assert proxy.scores is target.scores

    def test_missing_scores_read_as_absent(self):
        class _Bare:
            labels = np.zeros(1)
            boxes = np.zeros((1, 4))
            track_ids = np.zeros(1)

        assert getattr(_OffsetFrameTarget(_Bare(), 1), "scores", None) is None

    def test_forwards_public_attributes(self):
        target = cast(Any, make_target(1))
        target.confidence_source = "model-a"
        assert _OffsetFrameTarget(target, 1).confidence_source == "model-a"


class TestFrameTracksTarget:
    def test_satisfies_the_tracking_target_protocol(self):
        target = _FrameTracksTarget([make_target(), make_target()])
        assert _is_protocol_instance(target, MultiobjectTrackingTarget)
        assert len(target.frame_tracks) == 2

    def test_holds_the_given_objects_in_order(self):
        frames = [make_target(), make_target(3)]
        held = _FrameTracksTarget(iter(frames))
        assert all(a is b for a, b in zip(held.frame_tracks, frames, strict=True))

    def test_repr(self):
        assert "n_frames=2" in repr(_FrameTracksTarget([make_target(), make_target()]))


class TestLazyStream:
    def test_each_iteration_opens_afresh(self):
        opened: list[int] = []

        def walk():
            opened.append(1)
            return iter([1, 2])

        stream = _LazyStream(walk, "test")
        assert list(stream) == [1, 2]
        assert list(stream) == [1, 2]
        assert len(opened) == 2
        assert "test" in repr(stream)
