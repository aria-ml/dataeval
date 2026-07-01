from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest

from dataeval.data._tracks import build_tracks
from dataeval.protocols import (
    DatasetMetadata,
    DatumMetadata,
    MultiobjectTrackingDataset,
    MultiobjectTrackingDatum,
    MultiobjectTrackingTarget,
    SingleFrameObjectTrackingTarget,
    VideoStream,
)
from dataeval.types import Track


@dataclass
class _FakeFrame:
    """``SingleFrameObjectTrackingTarget``-shaped stand-in.

    A real class (rather than a ``SimpleNamespace``) so the type checker can
    see the declared members and verify it structurally matches the protocol.
    Fields are typed ``np.ndarray`` because an ndarray satisfies the protocol's
    array-typed members (``ArrayLike``/``Array``) while remaining concrete.
    """

    track_ids: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


@dataclass
class _FakeVideoTarget:
    """``MultiobjectTrackingTarget``-shaped stand-in (one video's frame tracks)."""

    frame_tracks: Sequence[SingleFrameObjectTrackingTarget]


def make_frame(track_ids, boxes, scores, labels) -> SingleFrameObjectTrackingTarget:
    """Build a bare ``SingleFrameObjectTrackingTarget``-shaped target."""
    return _FakeFrame(
        track_ids=np.array(track_ids, dtype=np.int64),
        boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4),
        scores=np.array(scores, dtype=np.float32),
        labels=np.array(labels, dtype=np.int64),
    )


def make_empty_frame() -> SingleFrameObjectTrackingTarget:
    """Build a bare empty ``SingleFrameObjectTrackingTarget``-shaped target."""
    return make_frame([], np.empty((0, 4)), [], [])


def make_video_target(frames: Sequence[SingleFrameObjectTrackingTarget]) -> MultiobjectTrackingTarget:
    """Build a bare ``MultiobjectTrackingTarget``-shaped target."""
    return _FakeVideoTarget(frame_tracks=list(frames))


class _FakeDataset:
    """``MultiobjectTrackingDataset``-shaped stand-in.

    Structurally matches ``AnnotatedDataset[MultiobjectTrackingDatum]``: it
    provides ``__getitem__`` returning a ``MultiobjectTrackingDatum``,
    ``__len__``, and a ``metadata`` property typed as ``DatasetMetadata``.

    ``build_tracks`` consumes the dataset with ``source[i]`` and
    ``len(source)``, so the indexing and sizing protocols are implemented on
    the *type* (a plain class). Python resolves ``__len__``/``__getitem__`` on
    the type for these implicit protocols, so attaching them to a
    ``SimpleNamespace`` instance (as this helper used to) is silently ignored
    and ``len()``/indexing would raise ``TypeError``.
    """

    def __init__(self, data: Sequence[MultiobjectTrackingDatum], metadata: DatasetMetadata) -> None:
        self._data = list(data)
        self._metadata = metadata

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[MultiobjectTrackingDatum]:
        return iter(self._data)

    def __getitem__(self, index: int) -> MultiobjectTrackingDatum:
        return self._data[index]

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata


class _ExplodingList(list):
    """Video-stream stand-in that raises if its contents are read.

    Subclasses ``list`` so it still reads as list-like at runtime, while
    failing loudly if ``build_tracks`` ever iterates, indexes, or measures
    the video-stream slot it is expected to ignore.
    """

    def __iter__(self):
        raise AssertionError("video-stream slot was iterated but must be unused")

    def __getitem__(self, index):
        raise AssertionError("video-stream slot was indexed but must be unused")

    def __len__(self):
        raise AssertionError("video-stream slot length was read but must be unused")


class _ExplodingDict(dict):
    """Dataset-metadata stand-in that raises if read.

    Subclasses ``dict`` so it reads as a mapping at runtime, while proving
    ``build_tracks`` never reads the dataset-level metadata.
    """

    def __getitem__(self, key):
        raise AssertionError("dataset metadata was read but must be unused")

    def __iter__(self):
        raise AssertionError("dataset metadata was iterated but must be unused")

    def get(self, *args, **kwargs):
        raise AssertionError("dataset metadata was read but must be unused")

    def keys(self):
        raise AssertionError("dataset metadata keys were read but must be unused")


def make_dataset(items, metadata: DatasetMetadata | None = None) -> MultiobjectTrackingDataset:
    """Build a ``MultiobjectTrackingDataset``-shaped stand-in.

    ``items`` is a list of ``(video_target, metadata_id)`` pairs. Each datum is
    a ``(video_stream, video_target, datum_metadata)`` triple. The video-stream
    slot and the dataset-level ``metadata`` are present only to satisfy the
    datum/dataset types; ``build_tracks`` reads neither (see the dedicated
    "is_never_accessed" tripwire tests). The per-datum ``DatumMetadata`` *is*
    read, to key the result.

    The empty video-stream slot is ``cast`` to ``VideoStream`` because it is a
    never-read placeholder and an empty literal carries no element type for the
    checker to match against the protocol.
    """
    data: list[MultiobjectTrackingDatum] = [
        (cast(VideoStream, []), video_target, DatumMetadata(id=meta_id)) for video_target, meta_id in items
    ]
    return _FakeDataset(data, DatasetMetadata(id="") if metadata is None else metadata)


@pytest.mark.required
class TestBuildTracks:
    """Tests for ``build_tracks``."""

    def test_returns_track_objects(self):
        vt = make_video_target([make_frame([1], [[0, 0, 10, 10]], [0.9], [0])])
        tracks = build_tracks(vt)
        assert isinstance(tracks[1], Track)

    def test_basic_inversion_single_track(self):
        # One track present in three consecutive frames.
        vt = make_video_target([
            make_frame([1], [[0, 0, 10, 10]], [0.9], [2]),
            make_frame([1], [[10, 0, 20, 10]], [0.8], [2]),
            make_frame([1], [[20, 0, 30, 10]], [0.7], [2]),
        ])
        tracks = build_tracks(vt)
        assert list(tracks.keys()) == [1]
        track = tracks[1]
        assert track.track_id == 1
        np.testing.assert_array_equal(track.frame_indices, [0, 1, 2])
        assert track.boxes.shape == (3, 4)
        np.testing.assert_array_equal(track.boxes[0], [0, 0, 10, 10])
        np.testing.assert_array_equal(track.boxes[-1], [20, 0, 30, 10])
        np.testing.assert_array_equal(track.labels, [2, 2, 2])
        np.testing.assert_allclose(track.scores, [0.9, 0.8, 0.7], rtol=1e-6)

    def test_multiple_tracks_keys_sorted(self):
        vt = make_video_target([
            make_frame([7, 3], [[0, 0, 10, 10], [5, 5, 15, 15]], [0.9, 0.8], [1, 2]),
            make_frame([3], [[6, 6, 16, 16]], [0.7], [2]),
            make_frame([7], [[20, 0, 30, 10]], [0.95], [1]),
        ])
        tracks = build_tracks(vt)
        # Keys come out in ascending track-ID order.
        assert list(tracks.keys()) == [3, 7]
        np.testing.assert_array_equal(tracks[3].frame_indices, [0, 1])
        np.testing.assert_array_equal(tracks[7].frame_indices, [0, 2])

    def test_track_with_gap(self):
        # Track 7 appears in frame 0 and frame 3 but is absent from 1 and 2.
        vt = make_video_target([
            make_frame([7], [[0, 0, 10, 10]], [0.9], [1]),
            make_frame([3], [[5, 5, 15, 15]], [0.8], [2]),
            make_frame([3], [[6, 6, 16, 16]], [0.8], [2]),
            make_frame([7], [[20, 0, 30, 10]], [0.95], [1]),
        ])
        tracks = build_tracks(vt)
        # The gap is preserved as non-consecutive frame indices.
        np.testing.assert_array_equal(tracks[7].frame_indices, [0, 3])
        assert tracks[7].boxes.shape == (2, 4)

    def test_single_appearance_track(self):
        vt = make_video_target([
            make_frame([5], [[1, 1, 2, 2]], [0.5], [0]),
            make_frame([9], [[3, 3, 4, 4]], [0.6], [1]),
        ])
        tracks = build_tracks(vt)
        np.testing.assert_array_equal(tracks[5].frame_indices, [0])
        np.testing.assert_array_equal(tracks[9].frame_indices, [1])
        assert tracks[5].boxes.shape == (1, 4)

    def test_empty_frames_are_skipped(self):
        # Frames with no detections must not advance any track's observations,
        # but the frame index counter still advances (enumerate over all frames).
        vt = make_video_target([
            make_frame([1], [[0, 0, 10, 10]], [0.9], [0]),
            make_empty_frame(),
            make_frame([1], [[10, 0, 20, 10]], [0.8], [0]),
        ])
        tracks = build_tracks(vt)
        # Frame 1 was empty -> track 1 observed at frames 0 and 2.
        np.testing.assert_array_equal(tracks[1].frame_indices, [0, 2])

    def test_all_empty_video_returns_empty_mapping(self):
        vt = make_video_target([make_empty_frame(), make_empty_frame()])
        assert build_tracks(vt) == {}

    def test_no_frames_returns_empty_mapping(self):
        vt = make_video_target([])
        assert build_tracks(vt) == {}

    def test_observations_in_ascending_frame_order(self):
        vt = make_video_target([make_frame([1], [[i, 0, i + 1, 1]], [1.0], [0]) for i in range(5)])
        track = build_tracks(vt)[1]
        # Frame indices are strictly increasing.
        assert list(track.frame_indices) == sorted(track.frame_indices)
        np.testing.assert_array_equal(track.frame_indices, [0, 1, 2, 3, 4])

    def test_multiple_detections_same_frame(self):
        # Two different track IDs detected within the same frame.
        vt = make_video_target([make_frame([1, 2], [[0, 0, 5, 5], [10, 10, 20, 20]], [0.9, 0.4], [0, 1])])
        tracks = build_tracks(vt)
        assert set(tracks.keys()) == {1, 2}
        np.testing.assert_array_equal(tracks[1].boxes[0], [0, 0, 5, 5])
        np.testing.assert_array_equal(tracks[2].boxes[0], [10, 10, 20, 20])
        assert tracks[2].labels[0] == 1


@pytest.mark.required
class TestBuildAllTracks:
    """Tests for ``build_tracks``."""

    def test_basic_multi_sequence(self):
        seq_a = make_video_target([make_frame([1], [[0, 0, 10, 10]], [0.9], [0])])
        seq_b = make_video_target([
            make_frame([2], [[1, 1, 2, 2]], [0.5], [3]),
            make_frame([2], [[2, 2, 3, 3]], [0.5], [3]),
        ])
        dataset = make_dataset([(seq_a, "vid_a"), (seq_b, "vid_b")])
        result = build_tracks(dataset)

        assert set(result.keys()) == {"vid_a", "vid_b"}
        assert list(result["vid_a"].keys()) == [1]
        np.testing.assert_array_equal(result["vid_b"][2].frame_indices, [0, 1])

    def test_sequence_id_coerced_to_string(self):
        # Metadata ids that are not strings are stringified.
        seq = make_video_target([make_frame([1], [[0, 0, 10, 10]], [0.9], [0])])
        dataset = make_dataset([(seq, 0), (seq, 1)])
        result = build_tracks(dataset)
        assert set(result.keys()) == {"0", "1"}

    def test_empty_dataset(self):
        assert build_tracks(make_dataset([])) == {}

    def test_sequence_with_only_empty_frames(self):
        empty_seq = make_video_target([make_empty_frame()])
        normal_seq = make_video_target([make_frame([1], [[0, 0, 10, 10]], [0.9], [0])])
        dataset = make_dataset([(empty_seq, "empty"), (normal_seq, "normal")])
        result = build_tracks(dataset)
        assert result["empty"] == {}
        assert list(result["normal"].keys()) == [1]

    def test_each_value_is_a_build_tracks_mapping(self):
        seq = make_video_target([
            make_frame([1], [[0, 0, 10, 10]], [0.9], [0]),
            make_frame([1], [[10, 0, 20, 10]], [0.8], [0]),
        ])
        dataset = make_dataset([(seq, "only")])
        result = build_tracks(dataset)
        track = result["only"][1]
        assert isinstance(track, Track)
        np.testing.assert_array_equal(track.frame_indices, [0, 1])

    def test_video_stream_slot_is_never_accessed(self):
        # The video-stream slot is only present to satisfy the datum type. If
        # build_tracks ever reads it, the tripwire raises and this fails.
        seq = make_video_target([make_frame([1], [[0, 0, 10, 10]], [0.9], [0])])
        datum: MultiobjectTrackingDatum = (cast(VideoStream, _ExplodingList()), seq, DatumMetadata(id="vid"))
        dataset = _FakeDataset([datum], DatasetMetadata(id=""))
        result = build_tracks(dataset)
        assert list(result["vid"].keys()) == [1]

    def test_dataset_level_metadata_is_never_accessed(self):
        # The dataset-level metadata is only present to satisfy the dataset
        # type; result keys come from the per-datum id, not from here. The cast
        # smuggles in a deliberately wrong concrete type to prove non-use.
        seq = make_video_target([make_frame([1], [[0, 0, 10, 10]], [0.9], [0])])
        datum: MultiobjectTrackingDatum = (cast(VideoStream, []), seq, DatumMetadata(id="vid"))
        dataset = _FakeDataset([datum], cast(DatasetMetadata, _ExplodingDict(id="")))
        result = build_tracks(dataset)
        assert list(result["vid"].keys()) == [1]
