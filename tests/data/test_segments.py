"""Tests for VideoSegments."""

import logging
from typing import Any, cast

import numpy as np
import pytest

from dataeval.data import Cuts, VideoSegments, Window
from dataeval.exceptions import MaiteShapeError
from dataeval.flags import ImageStats
from dataeval.protocols import (
    DatasetMetadata,
    DatumMetadata,
    MultiobjectTrackingTarget,
    VideoFrame,
    _is_protocol_instance,
)
from tests.data.test_frames import _CountingStream, _FakeDataset, _FakeVideoTarget, make_dataset, make_target


def meta_at(view: VideoSegments, index: int) -> dict[str, Any]:
    return cast(dict[str, Any], view[index][2])


def with_metadata(frame_counts, extra: list[dict[str, Any]], *, timed: bool = True):
    """Create a tracking dataset with specified metadata entries."""
    data = []
    counters = []
    for seq, n in enumerate(frame_counts):
        counter = {"iterations": 0, "frames": 0, "pixels": 0}
        counters.append(counter)
        stream = _CountingStream(n, (3, 12, 14), counter, timed=timed)
        target = _FakeVideoTarget(frame_tracks=[make_target() for _ in range(n)])
        metadata = {"id": f"vid{seq}", "height": 12, "width": 14, **extra[seq]}
        data.append((cast(Any, stream), cast(Any, target), cast(DatumMetadata, metadata)))
    return _FakeDataset(data, DatasetMetadata({"id": "videos", "index2label": {0: "thing"}})), counters


class TestConstruction:
    def test_rejects_a_non_tracking_dataset(self):
        images = [np.zeros((3, 4, 4)) for _ in range(3)]
        with pytest.raises(MaiteShapeError):
            VideoSegments(cast(Any, images), Window(2))

    def test_requires_a_planner(self):
        dataset, _ = make_dataset((3,))
        with pytest.raises(TypeError, match="SegmentPlanner"):
            VideoSegments(dataset, cast(Any, 3))

    def test_rejects_an_unknown_timestamp_policy(self):
        dataset, _ = make_dataset((3,))
        with pytest.raises(ValueError, match="timestamps"):
            VideoSegments(dataset, Window(2), timestamps=cast(Any, "utc"))

    def test_sizes_without_decoding(self):
        dataset, counters = make_dataset((6, 4))
        view = VideoSegments(dataset, Window(4))
        assert len(view) == 3
        np.testing.assert_array_equal(view.segment_map, [[0, 0, 4], [0, 4, 6], [1, 0, 4]])
        assert view.n_source_frames == 10
        assert view.n_dropped_frames == 0
        assert all(counter["iterations"] == 0 for counter in counters), "constructing decoded something"

    def test_dropped_frames_and_empty_plans_are_counted_and_logged(self, caplog):
        dataset, _ = make_dataset((6, 3))
        with caplog.at_level("INFO", logger="dataeval"):
            view = VideoSegments(dataset, Window(4, drop_remainder=True))
        assert len(view) == 1
        assert view.n_dropped_frames == 5
        assert any("plans no segment" in record.message for record in caplog.records)

    def test_overlap_warns_with_the_redecode_count(self, caplog):
        dataset, _ = make_dataset((6,))
        with caplog.at_level("WARNING", logger="dataeval"):
            view = VideoSegments(dataset, Window(4, stride=2))
        np.testing.assert_array_equal(view.segment_map, [[0, 0, 4], [0, 2, 6], [0, 4, 6]])
        warnings = [record.message for record in caplog.records if record.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "4 frame(s)" in warnings[0]
        assert "repeated unit rows" in warnings[0]

    def test_invalid_plan_is_reported(self):
        dataset, _ = make_dataset((6,))
        with pytest.raises(ValueError, match="Cuts"):
            VideoSegments(dataset, Cuts({"vid0": [6]}))

    def test_source_metadata_repr_and_str(self):
        dataset, _ = make_dataset((3,))
        view = VideoSegments(dataset, Window(2))
        assert view.source is dataset
        assert view.planner.size == 2  # type: ignore[attr-defined]
        assert view.timestamps == "source"
        assert view.metadata["id"] == "videos-segments"
        assert cast(dict, view.metadata)["index2label"] == {0: "thing"}
        assert view.invalidates == ImageStats.NONE
        assert "VideoSegments" in repr(view)
        assert "segments" in str(view)

    def test_segment_map_is_read_only(self):
        dataset, _ = make_dataset((3,))
        view = VideoSegments(dataset, Window(2))
        with pytest.raises(ValueError, match="read-only"):
            view.segment_map[0, 0] = 1


class TestGetItem:
    def test_negative_and_out_of_range_indices(self):
        dataset, _ = make_dataset((6,))
        view = VideoSegments(dataset, Window(4))
        assert meta_at(view, -1)["start_frame"] == 4
        with pytest.raises(IndexError):
            view[2]

    def test_target_is_the_sources_own_slice(self):
        dataset, _ = make_dataset((6,))
        view = VideoSegments(dataset, Window(4))
        _, target, _ = view[1]
        source_tracks = dataset[0][1].frame_tracks
        assert _is_protocol_instance(target, MultiobjectTrackingTarget)
        assert all(a is b for a, b in zip(target.frame_tracks, source_tracks[4:6], strict=True))

    def test_metadata_carries_lineage_and_a_positional_id(self):
        dataset, _ = make_dataset((6, 4))
        view = VideoSegments(dataset, Window(4))
        assert meta_at(view, 1) == {
            "id": 1,
            "source_id": "vid0",
            "segment_index": 1,
            "start_frame": 4,
            "end_frame": 6,
            "height": 12,
            "width": 14,
        }
        assert meta_at(view, 2)["segment_index"] == 0
        assert meta_at(view, 2)["source_id"] == "vid1"

    def test_source_id_falls_back_to_the_position(self):
        dataset, _ = with_metadata((3,), [{}])
        del cast(dict, dataset[0][2])["id"]
        assert meta_at(VideoSegments(dataset, Window(2)), 0)["source_id"] == 0

    def test_per_frame_lists_are_sliced_and_scalars_kept(self):
        dataset, _ = with_metadata(
            (6,),
            [
                {
                    "size": 999,
                    "weather": "rain",
                    "frame_quality": [10, 11, 12, 13, 14, 15],
                    "sensor": {"gain": (0, 1, 2, 3, 4, 5), "name": "left"},
                    "corners": [1, 2, 3, 4],
                    "blur": np.arange(6) * 0.5,
                }
            ],
        )
        meta = meta_at(VideoSegments(dataset, Window(4)), 1)
        assert meta["frame_quality"] == [14, 15]
        assert meta["sensor"] == {"gain": (4, 5), "name": "left"}
        assert meta["corners"] == [1, 2, 3, 4]
        assert meta["weather"] == "rain"
        np.testing.assert_array_equal(meta["blur"], [2.0, 2.5])
        assert "size" not in meta

    def test_injected_keys_displace_source_keys_with_one_warning_each(self, caplog):
        dataset, _ = with_metadata((6,), [{"source_id": "mine", "start_frame": 99}])
        view = VideoSegments(dataset, Window(2))
        with caplog.at_level("WARNING", logger="dataeval"):
            first = meta_at(view, 0)
            second = meta_at(view, 1)
        assert first["source_id"] == "vid0"
        assert first["start_frame"] == 0
        assert second["start_frame"] == 2
        displaced = [r.message for r in caplog.records if "displaced" in r.message]
        assert len(displaced) == 2

    def test_satisfies_the_tracking_dataset_shape(self):
        from dataeval.utils.data import validate_dataset

        dataset, _ = make_dataset((3,))
        view = VideoSegments(dataset, Window(2))
        assert validate_dataset(view, expected="multiobject_tracking") == "multiobject_tracking"

    def test_iter_matches_getitem(self):
        dataset, _ = make_dataset((5, 3))
        view = VideoSegments(dataset, Window(2))
        assert [cast(dict, m)["id"] for _, _, m in view] == [meta_at(view, i)["id"] for i in range(len(view))]


class TestFrames:
    def test_frame_index_is_local_and_timings_follow_the_policy(self):
        dataset, _ = make_dataset((6,))
        frames = list(VideoSegments(dataset, Window(4))[1][0])
        assert [f.frame_index for f in frames] == [0, 1]
        assert [f.time_s for f in frames] == pytest.approx([4 / 30, 5 / 30])
        assert [f.pts for f in frames] == [4004, 5005]
        local = list(VideoSegments(dataset, Window(4), timestamps="local")[1][0])
        assert [f.frame_index for f in local] == [0, 1]
        assert [f.time_s for f in local] == pytest.approx([0.0, 1 / 30])
        assert [f.pts for f in local] == [0, 1001]

    def test_frames_satisfy_the_protocol(self):
        dataset, _ = make_dataset((3,))
        frame = next(iter(VideoSegments(dataset, Window(2))[0][0]))
        assert _is_protocol_instance(frame, VideoFrame)

    def test_untimed_frames_stay_untimed(self):
        dataset, _ = make_dataset((4,), timed=False)
        for policy in ("source", "local"):
            frames = list(VideoSegments(dataset, Window(2), timestamps=cast(Any, policy))[1][0])
            assert all(getattr(f, "time_s", None) is None for f in frames)
            assert all(getattr(f, "pts", None) is None for f in frames)

    def test_pixels_are_the_source_frames(self):
        dataset, _ = make_dataset((6,), shape=(3, 2, 2))
        frames = list(VideoSegments(dataset, Window(4))[1][0])
        np.testing.assert_array_equal(frames[0].pixels, np.full((3, 2, 2), 4, dtype=np.uint8))

    def test_stream_is_reiterable(self):
        dataset, _ = make_dataset((6,))
        stream = VideoSegments(dataset, Window(4))[1][0]
        assert [f.frame_index for f in stream] == [0, 1]
        assert [f.frame_index for f in stream] == [0, 1]

    def test_short_stream_raises_naming_the_video(self):
        counter = {"iterations": 0, "frames": 0, "pixels": 0}
        stream = _CountingStream(6, (3, 4, 4), counter)
        target = _FakeVideoTarget(frame_tracks=[make_target() for _ in range(8)])
        dataset = _FakeDataset([(stream, target, {"id": "short"})], DatasetMetadata({"id": "videos"}))
        view = VideoSegments(cast(Any, dataset), Window(4))
        with pytest.raises(ValueError, match="'short'.*yielded only 6"):
            list(view[1][0])

    def test_long_stream_raises_on_the_last_segment(self):
        dataset, _ = make_dataset((6,), extra=1)
        view = VideoSegments(dataset, Window(4))
        assert len(list(view[0][0])) == 4
        with pytest.raises(ValueError, match="more frames"):
            list(view[1][0])


class TestDecodeAccounting:
    def test_in_order_access_decodes_each_frame_once(self):
        dataset, counters = make_dataset((6, 4))
        for stream, _, _ in VideoSegments(dataset, Window(2)):
            list(stream)
        assert [counter["frames"] for counter in counters] == [6, 4]
        assert [counter["iterations"] for counter in counters] == [1, 1]

    def test_metadata_walk_is_single_pass(self):
        from dataeval import Metadata

        dataset, counters = make_dataset((6, 4))
        md = Metadata(VideoSegments(dataset, Window(2)))
        assert md.level_counts["unit"] == 10
        assert [counter["iterations"] for counter in counters] == [1, 1]
        assert [counter["frames"] for counter in counters] == [6, 4]

    def test_out_of_order_access_decodes_the_skipped_frames_and_warns_once(self, caplog):
        dataset, counters = make_dataset((6,))
        view = VideoSegments(dataset, Window(2))
        with caplog.at_level("WARNING", logger="dataeval"):
            assert [f.time_s for f in view[2][0]] == pytest.approx([4 / 30, 5 / 30])
            assert [f.time_s for f in view[0][0]] == pytest.approx([0.0, 1 / 30])
            assert [f.time_s for f in view[1][0]] == pytest.approx([2 / 30, 3 / 30])
        assert counters[0]["frames"] == 6 + 2 + 2
        assert counters[0]["iterations"] == 2
        skips = [r.message for r in caplog.records if "decodes and discards" in r.message]
        assert len(skips) == 1

    def test_two_live_handles_do_not_interfere(self):
        dataset, counters = make_dataset((6,))
        view = VideoSegments(dataset, Window(2))
        first = iter(view[0][0])
        second = iter(view[1][0])
        assert next(first).time_s == pytest.approx(0.0)
        assert next(second).time_s == pytest.approx(2 / 30)
        assert [f.time_s for f in first] == pytest.approx([1 / 30])
        assert [f.time_s for f in second] == pytest.approx([3 / 30])
        assert counters[0]["iterations"] == 2

    def test_abandoned_handle_does_not_corrupt_a_later_one(self):
        dataset, _ = make_dataset((6,))
        view = VideoSegments(dataset, Window(2))
        held = iter(view[0][0])
        next(held)
        assert [f.time_s for f in view[1][0]] == pytest.approx([2 / 30, 3 / 30])
        assert [f.time_s for f in held] == pytest.approx([1 / 30])

    def test_released_handle_returns_the_loan(self):
        dataset, counters = make_dataset((6,))
        view = VideoSegments(dataset, Window(2))
        held = iter(view[0][0])
        next(held)
        held.close()  # type: ignore[attr-defined]
        list(view[1][0])
        assert counters[0]["iterations"] == 2

    def test_overlapping_windows_redecode_only_the_overlap_without_skip_warnings(self, caplog):
        dataset, counters = make_dataset((6,))
        view = VideoSegments(dataset, Window(4, stride=2))
        with caplog.at_level("WARNING", logger="dataeval"):
            for stream, _, _ in view:
                list(stream)
        assert counters[0]["frames"] == 4 + (2 + 4) + (4 + 2)
        assert counters[0]["iterations"] == 3
        assert not any("decodes and discards" in r.message for r in caplog.records)

    def test_repeated_read_of_one_segment_reopens_without_a_warning(self, caplog):
        dataset, counters = make_dataset((6,))
        view = VideoSegments(dataset, Window(2))
        with caplog.at_level("WARNING", logger="dataeval"):
            list(view[0][0])
            list(view[0][0])
        assert counters[0]["iterations"] == 2
        assert counters[0]["frames"] == 4
        assert [r.message for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_one_source_read_per_video_for_an_in_order_pass(self):
        """Verify sequential access reads each source datum once."""

        class _Counting:
            def __init__(self, inner):
                self._inner, self.metadata, self.reads = inner, inner.metadata, 0

            def __len__(self):
                return len(self._inner)

            def __getitem__(self, index):
                self.reads += 1
                return self._inner[index]

        inner, counters = make_dataset((6, 4))
        dataset = _Counting(inner)
        view = VideoSegments(dataset, Window(2))
        dataset.reads = 0
        for _, (stream, _, _) in enumerate(view):
            list(stream)
        assert dataset.reads == 2
        assert [counter["frames"] for counter in counters] == [6, 4]

    def test_last_segment_releases_the_decoder(self):
        dataset, counters = make_dataset((4,))
        view = VideoSegments(dataset, Window(2))
        list(view[0][0])
        list(view[1][0])
        assert view._cursor is None
        assert counters[0]["frames"] == 4
