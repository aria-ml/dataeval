"""Tests for VideoStitch."""

import logging
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pytest

from dataeval.data import VideoStitch
from dataeval.exceptions import MaiteShapeError
from dataeval.flags import ImageStats
from dataeval.protocols import (
    DatasetMetadata,
    DatumMetadata,
    MultiobjectTrackingTarget,
    VideoFrame,
    _is_protocol_instance,
)
from tests.data.test_frames import _CountingStream, _FakeDataset, _FakeFrameTarget, _FakeVideoTarget


def tracked(track_ids: Sequence[int]) -> _FakeFrameTarget:
    n = len(track_ids)
    return _FakeFrameTarget(
        track_ids=np.asarray(track_ids, dtype=np.int64),
        boxes=np.tile(np.array([1.0, 2.0, 9.0, 10.0], dtype=np.float32), (n, 1)),
        scores=np.ones(n, dtype=np.float32),
        labels=np.zeros(n, dtype=np.int64),
    )


def make_mot(
    videos: Sequence[Sequence[Sequence[int]]],
    *,
    metadata: Sequence[dict[str, Any]] | None = None,
    timed: bool = True,
    shapes: Sequence[tuple[int, int, int]] | None = None,
):
    """Create a tracking dataset from per-frame track IDs."""
    data = []
    counters = []
    for v, frames in enumerate(videos):
        counter = {"iterations": 0, "frames": 0, "pixels": 0}
        counters.append(counter)
        shape = shapes[v] if shapes else (3, 12, 14)
        stream = _CountingStream(len(frames), shape, counter, timed=timed)
        target = _FakeVideoTarget(frame_tracks=[tracked(ids) for ids in frames])
        meta: dict[str, Any] = {"id": f"vid{v}", "height": shape[1], "width": shape[2]}
        if metadata:
            meta.update(metadata[v])
        data.append((cast(Any, stream), cast(Any, target), cast(DatumMetadata, meta)))
    return _FakeDataset(data, DatasetMetadata({"id": "videos", "index2label": {0: "thing"}})), counters


class _CountingDataset:
    """Dataset wrapper tracking indexing access counts."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.metadata = inner.metadata
        self.reads = 0

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int):
        self.reads += 1
        return self._inner[index]


def meta_at(view: VideoStitch, index: int) -> dict[str, Any]:
    return cast(dict[str, Any], view[index][2])


THREE = [[[0], [0], [0]], [[1], [1]], [[2], [2], [2], [2]]]


class TestConstruction:
    def test_rejects_a_non_tracking_dataset(self):
        with pytest.raises(MaiteShapeError):
            VideoStitch(cast(Any, [np.zeros((3, 4, 4))]), group_by=None)

    def test_group_by_is_required(self):
        dataset, _ = make_mot(THREE)
        with pytest.raises(TypeError):
            VideoStitch(dataset)  # type: ignore[call-arg]

    @pytest.mark.parametrize("kwargs", [{"track_ids": "merge"}, {"timestamps": "utc"}])
    def test_rejects_unknown_policies(self, kwargs):
        dataset, _ = make_mot(THREE)
        with pytest.raises(ValueError, match="must be"):
            VideoStitch(dataset, group_by=None, **kwargs)

    def test_group_by_none_is_one_video_sized_without_decoding(self):
        dataset, counters = make_mot(THREE)
        view = VideoStitch(dataset, group_by=None)
        assert len(view) == 1
        assert view.groups == ((0, 1, 2),)
        assert view.group_keys == (None,)
        np.testing.assert_array_equal(view.frame_offsets[0], [0, 3, 5, 9])
        assert all(counter["iterations"] == 0 for counter in counters)

    def test_group_by_key_in_first_appearance_order(self):
        dataset, _ = make_mot(THREE, metadata=[{"scene": "a"}, {"scene": "b"}, {"scene": "a"}])
        view = VideoStitch(dataset, group_by="scene")
        assert view.groups == ((0, 2), (1,))
        assert view.group_keys == ("a", "b")

    def test_group_by_callable(self):
        dataset, _ = make_mot(THREE)
        view = VideoStitch(dataset, group_by=lambda m: str(m["id"])[-1] == "1")
        assert view.groups == ((0, 2), (1,))

    def test_order_by_key_and_callable_are_stable(self):
        dataset, _ = make_mot(THREE, metadata=[{"order": 2}, {"order": 1}, {"order": 1}])
        assert VideoStitch(dataset, group_by=None, order_by="order").groups == ((1, 2, 0),)
        by_callable = VideoStitch(dataset, group_by=None, order_by=lambda m: -cast(Any, m)["order"])
        assert by_callable.groups == ((0, 1, 2),)

    def test_missing_key_raises_naming_the_video(self):
        dataset, _ = make_mot(THREE, metadata=[{"scene": "a"}, {}, {"scene": "a"}])
        with pytest.raises(KeyError, match="vid1"):
            VideoStitch(dataset, group_by="scene")
        with pytest.raises(KeyError, match="vid1"):
            VideoStitch(dataset, group_by=None, order_by="scene")

    def test_geometry_disagreement_raises(self):
        dataset, _ = make_mot(THREE, shapes=[(3, 12, 14), (3, 12, 14), (3, 8, 8)])
        with pytest.raises(ValueError, match="height"):
            VideoStitch(dataset, group_by=None)
        assert len(VideoStitch(dataset, group_by="height")) == 2

    def test_absent_geometry_is_logged_not_raised(self, caplog):
        dataset, _ = make_mot(THREE, metadata=[{"time_base": 1}, {}, {"time_base": 1}])
        with caplog.at_level("INFO", logger="dataeval"):
            VideoStitch(dataset, group_by=None)
        assert any("time_base" in r.message and "not every video" in r.message for r in caplog.records)

    def test_overlapping_segments_of_one_source_raise(self):
        lineage = [
            {"source_id": "s", "start_frame": 0, "end_frame": 3},
            {"source_id": "s", "start_frame": 2, "end_frame": 4},
            {"source_id": "s", "start_frame": 4, "end_frame": 8},
        ]
        dataset, _ = make_mot(THREE, metadata=lineage)
        with pytest.raises(ValueError, match="overlap"):
            VideoStitch(dataset, group_by="source_id")

    def test_segments_out_of_source_order_are_named_as_such(self):
        lineage = [
            {"source_id": "s", "start_frame": 3, "end_frame": 6},
            {"source_id": "s", "start_frame": 0, "end_frame": 3},
        ]
        dataset, _ = make_mot([THREE[0], THREE[0]], metadata=lineage)
        with pytest.raises(ValueError, match="out of source order.*order_by"):
            VideoStitch(dataset, group_by="source_id")
        assert len(VideoStitch(dataset, group_by="source_id", order_by="start_frame")) == 1

    def test_array_valued_geometry_is_compared_elementwise(self):
        def bases(*pairs):
            return [{"time_base": np.array(pair)} for pair in pairs]

        dataset, _ = make_mot(THREE, metadata=bases([1, 30], [1, 30], [1, 30]))
        assert len(VideoStitch(dataset, group_by=None)) == 1
        dataset, _ = make_mot(THREE, metadata=bases([1, 30], [1, 25], [1, 30]))
        with pytest.raises(ValueError, match="time_base"):
            VideoStitch(dataset, group_by=None)

    def test_one_source_read_per_constituent_per_pass(self):
        """Verify each constituent is indexed only once per pass."""
        inner, _ = make_mot(THREE)
        dataset = _CountingDataset(inner)
        view = VideoStitch(dataset, group_by=None)
        dataset.reads = 0
        stream, _, _ = view[0]
        assert [f.frame_index for f in stream] == list(range(9))
        assert dataset.reads == 3

    def test_a_second_walk_reads_its_own_streams(self):
        """Verify re-iterating stream reads fresh source streams."""
        inner, counters = make_mot(THREE)
        dataset = _CountingDataset(inner)
        view = VideoStitch(dataset, group_by=None)
        stream, _, _ = view[0]
        assert len(list(stream)) == 9
        assert len(list(stream)) == 9
        assert [counter["iterations"] for counter in counters] == [2, 2, 2]

    def test_gap_between_segments_is_logged(self, caplog):
        lineage = [
            {"source_id": "s", "start_frame": 0, "end_frame": 3},
            {"source_id": "s", "start_frame": 3, "end_frame": 5},
            {"source_id": "s", "start_frame": 9, "end_frame": 13},
        ]
        dataset, _ = make_mot(THREE, metadata=lineage)
        with caplog.at_level("INFO", logger="dataeval"):
            VideoStitch(dataset, group_by="source_id")
        assert any("gap of 4 frame(s)" in r.message for r in caplog.records)

    def test_shared_ids_raise_under_error_policy(self):
        dataset, _ = make_mot([[[0, 1]], [[1, 2]]])
        with pytest.raises(ValueError, match=r"vid0.*vid1.*\[1\]"):
            VideoStitch(dataset, group_by=None, track_ids="error")
        disjoint, _ = make_mot([[[0, 1]], [[2, 3]]])
        assert len(VideoStitch(disjoint, group_by=None, track_ids="error")) == 1

    def test_repeated_source_ids_are_logged(self, caplog):
        dataset, _ = make_mot(THREE)
        cast(dict, dataset[1][2])["id"] = "vid0"
        with caplog.at_level("INFO", logger="dataeval"):
            VideoStitch(dataset, group_by=None)
        assert any("share a datum id" in r.message for r in caplog.records)

    def test_source_metadata_repr_and_str(self):
        dataset, _ = make_mot(THREE)
        view = VideoStitch(dataset, group_by=None)
        assert view.source is dataset
        assert view.metadata["id"] == "videos-stitched"
        assert cast(dict, view.metadata)["index2label"] == {0: "thing"}
        assert view.invalidates == ImageStats.NONE
        assert "VideoStitch" in repr(view)
        assert "stitched" in str(view)


class TestTargets:
    def test_offset_policy(self):
        dataset, _ = make_mot([[[0, 1], [1]], [[0, -1]], [[]], [[2]]])
        view = VideoStitch(dataset, group_by=None)
        np.testing.assert_array_equal(view.track_id_offsets[0], [0, 2, 3, 3])
        _, target, _ = view[0]
        assert _is_protocol_instance(target, MultiobjectTrackingTarget)
        tracks = target.frame_tracks
        assert len(tracks) == 5
        source = dataset[0][1].frame_tracks
        assert tracks[0] is source[0]
        assert tracks[1] is source[1]
        np.testing.assert_array_equal(tracks[2].track_ids, [2, -1])
        np.testing.assert_array_equal(tracks[3].track_ids, [])
        np.testing.assert_array_equal(tracks[4].track_ids, [5])

    def test_preserve_policy_passes_targets_through(self):
        dataset, _ = make_mot([[[0]], [[0]]])
        view = VideoStitch(dataset, group_by=None, track_ids="preserve")
        np.testing.assert_array_equal(view.track_id_offsets[0], [0, 0])
        _, target, _ = view[0]
        assert target.frame_tracks[1] is dataset[1][1].frame_tracks[0]

    def test_negative_and_out_of_range_indices(self):
        dataset, _ = make_mot(THREE, metadata=[{"scene": "a"}, {"scene": "b"}, {"scene": "a"}])
        view = VideoStitch(dataset, group_by="scene")
        assert meta_at(view, -1)["scene"] == "b"
        with pytest.raises(IndexError):
            view[2]

    def test_satisfies_the_tracking_dataset_shape(self):
        from dataeval.utils.data import validate_dataset

        dataset, _ = make_mot(THREE)
        view = VideoStitch(dataset, group_by=None)
        assert validate_dataset(view, expected="multiobject_tracking") == "multiobject_tracking"

    def test_iter_matches_getitem(self):
        dataset, _ = make_mot(THREE, metadata=[{"scene": "a"}, {"scene": "b"}, {"scene": "a"}])
        view = VideoStitch(dataset, group_by="scene")
        assert [cast(dict, m)["id"] for _, _, m in view] == [0, 1]


class TestMetadata:
    def test_scalar_rules(self, caplog):
        per_video = [
            {"weather": "rain", "size": 10, "cam": 1},
            {"weather": "rain", "size": 20, "cam": 2},
            {"weather": "rain"},
        ]
        dataset, _ = make_mot(THREE, metadata=per_video)
        view = VideoStitch(dataset, group_by=None)
        with caplog.at_level("INFO", logger="dataeval"):
            meta = meta_at(view, 0)
            meta_at(view, 0)
        assert meta == {"id": 0, "n_sources": 3, "height": 12, "width": 14, "weather": "rain"}
        omitted = [r.message for r in caplog.records if "omitted" in r.message]
        assert len(omitted) == 1
        assert "cam" in omitted[0]

    def test_size_is_summed_when_every_video_declares_it(self):
        dataset, _ = make_mot(THREE, metadata=[{"size": 10}, {"size": 20}, {"size": 5}])
        assert meta_at(VideoStitch(dataset, group_by=None), 0)["size"] == 35

    def test_per_frame_lists_are_concatenated_and_wrong_lengths_dropped(self, caplog):
        per_video = [
            {"q": [1, 2, 3], "corners": [1, 2, 3, 4], "sensor": {"gain": np.arange(3), "name": "x"}},
            {"q": (4, 5), "corners": [1, 2, 3, 4], "sensor": {"gain": np.arange(2), "name": "x"}},
            {"q": [6, 7, 8, 9], "corners": [1, 2, 3], "sensor": {"gain": np.arange(4), "name": "x"}},
        ]
        dataset, _ = make_mot(THREE, metadata=per_video)
        with caplog.at_level("INFO", logger="dataeval"):
            meta = meta_at(VideoStitch(dataset, group_by=None), 0)
        assert meta["q"] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        np.testing.assert_array_equal(meta["sensor"]["gain"], [0, 1, 2, 0, 1, 0, 1, 2, 3])
        assert meta["sensor"]["name"] == "x"
        assert "corners" not in meta
        assert any("corners" in r.message for r in caplog.records)

    def test_keys_absent_from_some_videos_are_omitted(self):
        dataset, _ = make_mot(THREE, metadata=[{"a": 1}, {}, {"b": 2}])
        meta = meta_at(VideoStitch(dataset, group_by=None), 0)
        assert "a" not in meta
        assert "b" not in meta


class TestStream:
    def test_chains_with_a_running_frame_index_and_source_timings(self):
        dataset, counters = make_mot([[[0], [0], [0]], [[1], [1]]])
        frames = list(VideoStitch(dataset, group_by=None)[0][0])
        assert [f.frame_index for f in frames] == [0, 1, 2, 3, 4]
        assert [f.time_s for f in frames] == pytest.approx([0, 1 / 30, 2 / 30, 0, 1 / 30])
        assert [f.pts for f in frames] == [0, 1001, 2002, 0, 1001]
        assert _is_protocol_instance(frames[0], VideoFrame)
        assert [c["iterations"] for c in counters] == [1, 1]

    def test_untimed_frames_stay_untimed(self):
        dataset, _ = make_mot(THREE, timed=False)
        frames = list(VideoStitch(dataset, group_by=None)[0][0])
        assert all(getattr(f, "time_s", None) is None for f in frames)

    def test_pixels_are_the_source_frames(self):
        dataset, _ = make_mot([[[0]], [[1], [1]]], shapes=[(3, 2, 2), (3, 2, 2)])
        frames = list(VideoStitch(dataset, group_by=None)[0][0])
        np.testing.assert_array_equal(frames[2].pixels, np.full((3, 2, 2), 1, dtype=np.uint8))

    def test_stream_is_reiterable(self):
        dataset, _ = make_mot(THREE)
        stream = VideoStitch(dataset, group_by=None)[0][0]
        assert len(list(stream)) == 9
        assert len(list(stream)) == 9

    def test_short_and_long_streams_raise_naming_the_video(self):
        dataset, _ = make_mot([[[0]], [[1], [1]]])
        cast(Any, dataset[1][1]).frame_tracks = [tracked([1])] * 3
        with pytest.raises(ValueError, match="'vid1'.*yielded only 2"):
            list(VideoStitch(dataset, group_by=None)[0][0])
        cast(Any, dataset[1][1]).frame_tracks = [tracked([1])]
        with pytest.raises(ValueError, match="'vid1'.*more frames"):
            list(VideoStitch(dataset, group_by=None)[0][0])

    def test_frame_size_mismatch_raises_lazily(self):
        dataset, _ = make_mot([[[0]], [[1]]], shapes=[(3, 4, 4), (3, 8, 8)])
        cast(dict, dataset[1][2]).update(height=4, width=4)  # metadata lies
        view = VideoStitch(dataset, group_by=None)
        with pytest.raises(ValueError, match="frame size"):
            list(view[0][0])


class TestOffsetTimeline:
    def test_source_policy_warns_once_when_time_steps_backwards(self, caplog):
        dataset, _ = make_mot([[[0], [0], [0]], [[1], [1]], [[2]]])
        view = VideoStitch(dataset, group_by=None)
        with caplog.at_level("WARNING", logger="dataeval"):
            list(view[0][0])
            list(view[0][0])
        backwards = [r.message for r in caplog.records if "steps backwards" in r.message]
        assert len(backwards) == 1
        assert "vid1" in backwards[0]

    def test_source_policy_warns_when_only_pts_steps_backwards(self, caplog):
        from tests.data.test_frames import _FakeFrame

        class _PtsOnly:
            def __init__(self, n: int) -> None:
                self._n = n

            def __iter__(self):
                counter = {"pixels": 0}
                return (_FakeFrame(i, (3, 12, 14), None, i * 1001, counter) for i in range(self._n))

        dataset, _ = make_mot([[[0], [0]], [[1], [1]]])
        data = [(cast(Any, _PtsOnly(2)), target, meta) for _, target, meta in dataset._data]
        view = VideoStitch(_FakeDataset(data, dataset.metadata), group_by=None)
        with caplog.at_level("WARNING", logger="dataeval"):
            frames = list(view[0][0])
        assert [f.pts for f in frames] == [0, 1001, 0, 1001]
        backwards = [r.message for r in caplog.records if "steps backwards" in r.message]
        assert len(backwards) == 1
        assert "pts" in backwards[0]

    def test_offset_policy_makes_a_continuous_timeline(self, caplog):
        dataset, _ = make_mot([[[0], [0], [0]], [[1], [1]]])
        view = VideoStitch(dataset, group_by=None, timestamps="offset")
        with caplog.at_level("WARNING", logger="dataeval"):
            frames = list(view[0][0])
        assert [f.time_s for f in frames] == pytest.approx([i / 30 for i in range(5)])
        assert [f.pts for f in frames] == [i * 1001 for i in range(5)]
        assert [f.frame_index for f in frames] == [0, 1, 2, 3, 4]
        assert [r.message for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_single_frame_constituent_reuses_the_last_known_interval(self):
        dataset, _ = make_mot([[[0], [0], [0]], [[1]], [[2], [2]]])
        frames = list(VideoStitch(dataset, group_by=None, timestamps="offset")[0][0])
        assert [f.time_s for f in frames] == pytest.approx([i / 30 for i in range(6)])
        assert [f.pts for f in frames] == [i * 1001 for i in range(6)]

    def test_no_known_interval_shares_the_boundary_timestamp_and_logs(self, caplog):
        dataset, _ = make_mot([[[0]], [[1], [1]]])
        with caplog.at_level("INFO", logger="dataeval"):
            frames = list(VideoStitch(dataset, group_by=None, timestamps="offset")[0][0])
        assert [f.time_s for f in frames] == pytest.approx([0.0, 0.0, 1 / 30])
        assert any("no frame interval" in r.message for r in caplog.records)

    def test_untimed_constituents_pass_through_under_offset(self):
        dataset, _ = make_mot([[[0], [0]], [[1], [1]]], timed=False)
        frames = list(VideoStitch(dataset, group_by=None, timestamps="offset")[0][0])
        assert all(getattr(f, "time_s", None) is None for f in frames)
        assert [f.frame_index for f in frames] == [0, 1, 2, 3]
