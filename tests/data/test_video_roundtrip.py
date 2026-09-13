"""Integration and round-trip tests for VideoSegments and VideoStitch."""

from typing import Any, cast

import numpy as np
import pytest

from dataeval import Metadata
from dataeval.data import Cuts, SegmentPlanner, SequenceFrames, VideoSegments, VideoStitch, Window, build_tracks
from dataeval.data._invalidates import invalidating_sources
from dataeval.flags import ImageStats
from tests.data.test_frames import _FakeDataset, make_dataset
from tests.data.test_stitch import make_mot


def test_public_names_are_exported():
    import dataeval.data as data

    for name in ("Cuts", "SegmentPlanner", "VideoSegments", "VideoStitch", "Window"):
        assert name in data.__all__
    assert data.__all__ == sorted(data.__all__, key=lambda n: (n[0].islower(), n))


class TestRoundTrip:
    def test_stitching_segments_reproduces_each_source_video(self):
        dataset, _ = make_mot(
            [[[0, 1], [0, 1], [1], [2, 3], [2]], [[7], [7], [7]]],
            metadata=[
                {"weather": "rain", "size": 100, "frame_quality": [1, 2, 3, 4, 5], "sensor": {"gain": [5, 4, 3, 2, 1]}},
                {"weather": "sun", "size": 60, "frame_quality": [9, 8, 7], "sensor": {"gain": [1, 2, 3]}},
            ],
        )
        view = VideoStitch(VideoSegments(dataset, Window(2)), group_by="source_id", track_ids="preserve")
        assert len(view) == 2
        for index in range(2):
            stream, target, meta = view[index]
            source_stream, source_target, source_meta = dataset[index]
            frames, source_frames = list(stream), list(source_stream)
            assert [f.frame_index for f in frames] == [f.frame_index for f in source_frames]
            assert [f.time_s for f in frames] == pytest.approx([f.time_s for f in source_frames])
            assert [f.pts for f in frames] == [f.pts for f in source_frames]
            assert all(np.array_equal(a.pixels, b.pixels) for a, b in zip(frames, source_frames, strict=True))
            assert all(a is b for a, b in zip(target.frame_tracks, source_target.frame_tracks, strict=True))
            got = cast(dict[str, Any], meta)
            expected = cast(dict[str, Any], source_meta)
            assert got["id"] == index
            assert got["source_id"] == expected["id"]
            for key in ("height", "width", "weather", "frame_quality", "sensor"):
                assert got[key] == expected[key], key
            assert "size" not in got
            assert "segment_index" not in got
            assert "start_frame" not in got
            assert "end_frame" not in got

    def test_local_timestamps_do_not_round_trip_but_offset_repairs_them(self):
        dataset, _ = make_mot([[[0], [0], [0], [0]]])
        segments = VideoSegments(dataset, Window(2), timestamps="local")
        restarting = list(VideoStitch(segments, group_by="source_id", track_ids="preserve")[0][0])
        assert [f.time_s for f in restarting] == pytest.approx([0, 1 / 30, 0, 1 / 30])
        repaired = VideoStitch(segments, group_by="source_id", track_ids="preserve", timestamps="offset")
        assert [f.time_s for f in repaired[0][0]] == pytest.approx([0, 1 / 30, 2 / 30, 3 / 30])

    def test_sliding_windows_are_refused(self):
        dataset, _ = make_mot([[[0], [0], [0], [0]]])
        with pytest.raises(ValueError, match="overlap"):
            VideoStitch(VideoSegments(dataset, Window(2, stride=1)), group_by="source_id", track_ids="preserve")

    def test_offset_policy_splits_a_track_that_crossed_a_cut(self):
        dataset, _ = make_mot([[[0], [0], [0], [0]]])
        _, target, _ = VideoStitch(VideoSegments(dataset, Window(2)), group_by="source_id")[0]
        ids = [int(np.asarray(t.track_ids)[0]) for t in target.frame_tracks]
        assert ids == [0, 0, 1, 1]


class TestMetadataIntegration:
    def test_segments_structure_as_tracking_with_lineage_at_the_sequence_level(self):
        dataset, _ = make_dataset((6, 4))
        md = Metadata(VideoSegments(dataset, Window(3)))
        assert md.levels == ("sequence", "unit", "track", "instance")
        assert dict(md.level_counts) == {"sequence": 4, "unit": 10, "track": 8, "instance": 20}
        for name in ("source_id", "segment_index", "start_frame", "end_frame"):
            assert any(factor.endswith(name) for factor in md.factor_names), name
        rows = md.rows_at("sequence")
        assert rows["segment_index"].to_list() == [0, 1, 0, 1]
        assert rows["start_frame"].to_list() == [0, 3, 0, 3]
        assert md.rows_at("unit")["unit_index"].to_list() == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]

    def test_stitch_structures_as_one_sequence_with_offset_tracks(self):
        dataset, _ = make_dataset((6, 4))
        md = Metadata(VideoStitch(dataset, group_by=None))
        assert dict(md.level_counts) == {"sequence": 1, "unit": 10, "track": 4, "instance": 20}
        assert sorted(md.rows_at("track")["track_id"].to_list()) == [0, 1, 2, 3]
        assert any(factor.endswith("n_sources") for factor in md.factor_names)

    def test_per_frame_lists_reach_the_unit_level(self):
        dataset, _ = make_mot([[[0], [0], [0], [0]]], metadata=[{"quality": [1, 2, 3, 4]}])
        md = Metadata(VideoSegments(dataset, Window(2)))
        assert md.rows_at("unit")["quality"].to_list() == [1, 2, 3, 4]


class TestSiblingViews:
    def test_sequence_frames_over_segments_splits_by_segment_at_no_extra_decode(self):
        dataset, counters = make_dataset((6, 4))
        md = Metadata(SequenceFrames(VideoSegments(dataset, Window(3))))
        assert sorted(set(md.rows_at("unit")["sequence"].to_list())) == [0, 1, 2, 3]
        # Verify segmenting introduces no additional stream iterations.
        plain, baseline = make_dataset((6, 4))
        Metadata(SequenceFrames(plain)).rows_at("unit")
        assert [c["iterations"] for c in counters] == [c["iterations"] for c in baseline]
        assert [c["frames"] for c in counters] == [c["frames"] for c in baseline]

    def test_build_tracks_is_per_segment_and_per_stitch(self):
        dataset, _ = make_dataset((6, 4))
        by_segment = build_tracks(VideoSegments(dataset, Window(3)))
        assert list(by_segment) == ["0", "1", "2", "3"]
        assert all(sorted(tracks) == [0, 1] for tracks in by_segment.values())
        by_stitch = build_tracks(VideoStitch(dataset, group_by=None))
        assert sorted(by_stitch["0"]) == [0, 1, 2, 3]

    def test_invalidating_sources_walks_through_both_views(self):
        class _Invalidating(_FakeDataset):
            invalidates = ImageStats.HASH

        base, _ = make_dataset((4,))
        inner = _Invalidating(base._data, base.metadata)
        for view in (VideoSegments(inner, Window(2)), VideoStitch(inner, group_by=None)):
            assert invalidating_sources(view) == [("_Invalidating", ImageStats.HASH)]

    def test_a_custom_planner_drives_the_view(self):
        class Halves(SegmentPlanner):
            def plan(self, info):
                mid = info.n_frames // 2
                return np.array([[0, mid], [mid, info.n_frames]], dtype=np.intp)

        dataset, _ = make_dataset((6, 4))
        view = VideoSegments(dataset, Halves())
        np.testing.assert_array_equal(view.segment_map, [[0, 0, 3], [0, 3, 6], [1, 0, 2], [1, 2, 4]])

    def test_cuts_from_metadata_drive_the_view(self):
        dataset, _ = make_mot([[[0]] * 6], metadata=[{"shots": [2, 4]}])
        view = VideoSegments(dataset, Cuts("shots"))
        np.testing.assert_array_equal(view.segment_map, [[0, 0, 2], [0, 2, 4], [0, 4, 6]])
