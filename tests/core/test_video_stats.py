import time
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pytest

from dataeval import Metadata
from dataeval.config import use_max_processes
from dataeval.core import combine_stats_results, video_stats
from dataeval.flags import VideoStats
from dataeval.protocols import DatasetMetadata
from dataeval.quality import Duplicates
from dataeval.types import SourceIndex

pytestmark = pytest.mark.required


class _Frame:
    """A decoded frame. Built with ``pixels=None``, any read of its pixels fails the test."""

    def __init__(self, pixels: Any, time_s: float, frame_index: int = 0) -> None:
        self._pixels = pixels
        self.time_s = time_s
        self.frame_index = frame_index
        self.pts = frame_index

    @property
    def pixels(self) -> Any:
        assert self._pixels is not None, "pixels were read"
        return self._pixels


def _stream(n: int, fps: float = 30, shape: tuple[int, ...] = (3, 8, 8)) -> list[_Frame]:
    return [_Frame(np.zeros(shape, np.uint8), k / fps, k) for k in range(n)]


class _OneShot:
    """A stream that can be iterated only once, as a live decoder can."""

    def __init__(self, frames: Sequence[_Frame]) -> None:
        self._frames = frames
        self._used = False

    def __iter__(self):
        assert not self._used, "stream iterated twice"
        self._used = True
        return iter(self._frames)


class _CountingItems:
    """An indexable source without ``metadata`` that counts how often an item is fetched.

    Items are ``(stream, None, datum_metadata)`` triples, or the bare streams when `bare`.
    """

    def __init__(self, streams: Sequence[Any], *, bare: bool = False) -> None:
        self.streams = streams
        self.bare = bare
        self.calls = 0

    def __len__(self) -> int:
        return len(self.streams)

    def __getitem__(self, i: int) -> Any:
        self.calls += 1
        return self.streams[i] if self.bare else (self.streams[i], None, {"id": i})


class _CountingDataset(_CountingItems):
    """A tracking dataset that counts how often an item is fetched."""

    metadata = DatasetMetadata(id="counting")


def _textured(n_distinct, repeats, dtype=np.uint8, channels=3, near_black=False):
    """Seeded random textures, each repeated `repeats` times, as consecutive frames."""
    rng = np.random.default_rng(0)
    high = 2 if near_black else 256
    frames = []
    for _ in range(n_distinct):
        texture = rng.integers(0, high, (channels, 32, 32)).astype(np.uint8)
        pixels = texture.astype(np.float32) / 255 if dtype == np.float32 else texture
        frames += [pixels] * repeats
    return [_Frame(p, k / 30, k) for k, p in enumerate(frames)]


def _tracking(streams):
    from tests.embeddings.test_embeddings import MockDataset
    from tests.metadata.test_structurers import _FrameTracks, _MOTTarget

    targets = [_MOTTarget([_FrameTracks([0]) for _ in stream]) for stream in streams]
    return MockDataset(streams, targets, None)


def _mot_dataset(frames_per_sequence: Sequence[int]):
    from tests.embeddings.test_embeddings import MockDataset
    from tests.metadata.test_structurers import _FrameTracks, _MOTTarget

    streams = [_stream(n) for n in frames_per_sequence]
    targets = [_MOTTarget([_FrameTracks([0]) for _ in stream]) for stream in streams]
    return MockDataset(streams, targets, None)


def test_descriptive_on_constant_rate():
    s = video_stats(_stream(10))["stats"]
    assert s["n_frames"].tolist() == [10]
    assert s["video_duration"][0] == pytest.approx(10 / 30)
    assert s["frame_rate"][0] == pytest.approx(30.0)
    assert s["video_duration"].dtype == np.float64


def test_one_addressed_row_per_video():
    r = video_stats(_CountingDataset([_stream(4), [], _stream(6)]))
    assert r["source_index"] == [SourceIndex(0), SourceIndex(1), SourceIndex(2)]
    assert r["stats"]["n_frames"].tolist() == [4, 0, 6]
    assert np.isnan(r["stats"]["video_duration"][1])


def test_descriptive_never_reads_pixels():
    video_stats([_Frame(None, k / 30) for k in range(5)], stats=VideoStats.DESCRIPTIVE)


def test_a_one_shot_stream_is_read_once():
    assert video_stats(_OneShot(_stream(3)), stats=VideoStats.NO_EGO)["stats"]["n_frames"].tolist() == [3]


def test_each_item_is_fetched_once():
    ds = _CountingDataset([_stream(2), _stream(3)])
    video_stats(ds)
    assert ds.calls == 2


@pytest.mark.parametrize("bare", [False, True], ids=["triples", "streams"])
def test_each_item_is_fetched_once_without_metadata(bare: bool):
    items = _CountingItems([_stream(2), _stream(3)], bare=bare)
    assert video_stats(cast("Any", items))["stats"]["n_frames"].tolist() == [2, 3]
    assert items.calls == 2


def test_a_list_of_streams_is_not_a_dataset():
    assert video_stats([_stream(2), _stream(3)])["stats"]["n_frames"].tolist() == [2, 3]


def test_only_requested_names_come_back():
    assert set(video_stats(_stream(3), stats=VideoStats.DESCRIPTIVE_FRAME_COUNT)["stats"]) == {"n_frames"}


class _Unreadable:
    def __iter__(self):
        raise AssertionError("the stream was read before the parameters were checked")


def test_a_negative_redundancy_radius_is_refused_before_reading():
    with pytest.raises(ValueError, match="redundancy_radius"):
        video_stats(cast(Any, _Unreadable()), redundancy_radius=-1)


def test_nothing_selected_warns():
    with pytest.warns(UserWarning, match="no statistics"):
        r = video_stats(_stream(3), stats=VideoStats.NONE)
    assert r["source_index"] == [SourceIndex(0)]


def test_an_empty_dataset_returns_empty_columns():
    r = video_stats(_CountingDataset([]), stats=VideoStats.DESCRIPTIVE)
    assert r["source_index"] == []
    assert all(len(v) == 0 for v in r["stats"].values())
    assert set(r["stats"]) == {"n_frames", "video_duration", "frame_rate"}


def test_counts_stay_int64_with_no_videos():
    assert video_stats(_CountingDataset([]))["stats"]["n_frames"].dtype == np.int64


def test_parallel_matches_serial():
    ds = _CountingDataset([_stream(n) for n in (3, 5, 2, 7)])
    serial = video_stats(ds)
    with use_max_processes(2):
        parallel = video_stats(_CountingDataset([_stream(n) for n in (3, 5, 2, 7)]))
    for name in serial["stats"]:
        np.testing.assert_array_equal(serial["stats"][name], parallel["stats"][name])


def test_a_failing_video_stops_the_rest_under_max_processes():
    measured: list[int] = []

    class _Broken:
        def __iter__(self):
            raise RuntimeError("decoder failed")

    class _Slow:
        def __iter__(self):
            time.sleep(0.02)
            measured.append(1)
            return iter(_stream(2))

    videos = [_Broken(), *(_Slow() for _ in range(40))]
    with use_max_processes(2), pytest.raises(RuntimeError, match="decoder failed"):
        video_stats(cast(Any, videos), stats=VideoStats.DESCRIPTIVE)
    assert len(measured) < 10  # the videos already in flight, not all 40


def test_attaches_to_metadata_by_address():
    ds = _mot_dataset(frames_per_sequence=[3, 5])
    md = Metadata(ds)
    md.add_factors(video_stats(ds))
    assert md.rows_at("sequence")["n_frames"].to_list() == [3, 5]


def test_two_results_combine():
    stats, index, _ = combine_stats_results([video_stats(_stream(2)), video_stats(_stream(4))])
    assert stats["n_frames"].tolist() == [2, 4]
    assert [si.item for si in index] == [0, 1]


def test_identical_frames_are_one_long_run():
    s = video_stats(_textured(1, 6), stats=VideoStats.REDUNDANCY)["stats"]
    assert s["redundant_fraction"][0] == pytest.approx(5 / 6)
    assert s["longest_run"].tolist() == [6]


def test_changing_frames_have_no_run():
    s = video_stats(_textured(6, 1), stats=VideoStats.REDUNDANCY)["stats"]
    assert s["redundant_fraction"].tolist() == [0.0]
    assert s["longest_run"].tolist() == [0]


@pytest.mark.parametrize("n", [0, 1])
def test_too_short_for_a_run(n):
    s = video_stats(_textured(1, n) if n else [], stats=VideoStats.REDUNDANCY)["stats"]
    assert s["redundant_fraction"].tolist() == [0.0]
    assert s["longest_run"].tolist() == [0]


@pytest.mark.parametrize(
    ("dtype", "channels", "near_black"),
    [(np.uint8, 3, False), (np.uint8, 3, True), (np.float32, 3, False), (np.uint8, 1, False)],
)
def test_agrees_with_duplicates_on_float_and_gray_frames(dtype, channels, near_black):
    ds = _tracking([_textured(2, 3, dtype, channels, near_black), _textured(3, 2, dtype, channels, near_black)])
    ours = video_stats(ds, stats=VideoStats.REDUNDANCY | VideoStats.DESCRIPTIVE_FRAME_COUNT)["stats"]
    theirs = Duplicates().evaluate(ds).aggregate_by_sequence()
    assert ours["redundant_fraction"].tolist() == pytest.approx(theirs["redundant_fraction"].to_list())
    assert ours["longest_run"].tolist() == theirs["longest_run"].to_list()
    assert ours["n_frames"].tolist() == theirs["n_frames"].to_list()


def test_the_radius_is_shared_with_duplicates():
    from dataeval.core._temporal import DEFAULT_REDUNDANCY_RADIUS

    assert Duplicates().redundancy_radius == DEFAULT_REDUNDANCY_RADIUS == 4
