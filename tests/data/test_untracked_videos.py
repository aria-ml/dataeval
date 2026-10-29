"""Videos with no tracking targets are measured as video with no detections."""

from typing import Any

import numpy as np
import pytest

from dataeval.data import SequenceFrames, SourceLocator, Stride
from dataeval.data._video_proxies import as_tracking, is_pure_video
from dataeval.quality import Duplicates, Outliers
from dataeval.types import SourceIndex
from tests.quality.test_duplicates import make_tracking_dataset

_SHAPE = (3, 24, 32)
FLASH = (2, 8)
ORIGINAL, COPY = (0, 3), (3, 11)


def _sequences() -> list[list[np.ndarray]]:
    """Four videos of noise, with a flat white frame in video 2 and a copy of video 0's frame 3 in video 3."""
    rng = np.random.default_rng(0)
    sequences = [[rng.integers(100, 140, _SHAPE, dtype=np.uint8) for _ in range(20)] for _ in range(4)]
    sequences[FLASH[0]][FLASH[1]] = np.full(_SHAPE, 255, dtype=np.uint8)
    sequences[COPY[0]][COPY[1]] = sequences[ORIGINAL[0]][ORIGINAL[1]].copy()
    return sequences


class Counted:
    """A stream that counts its walks and has no length."""

    def __init__(self, stream: Any) -> None:
        self._stream, self.walks = stream, 0

    def __iter__(self):
        self.walks += 1
        return iter(self._stream)


class CountedSized(Counted):
    """The same, with a length."""

    def __len__(self) -> int:
        return len(list(self._stream))


class Bare:
    """A dataset whose datum is the stream alone, with no metadata attribute."""

    def __init__(self, dataset: Any, sized: bool = True) -> None:
        self._dataset, self.streams = dataset, {}
        self._sized = sized

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Any:
        stream = (CountedSized if self._sized else Counted)(self._dataset[index][0])
        self.streams[index] = stream
        return stream


class NoTargets(Bare):
    """A dataset whose datum is (stream, None, metadata)."""

    def __getitem__(self, index: int) -> Any:
        stream, _, metadata = self._dataset[index]
        return (CountedSized if self._sized else Counted)(stream), None, metadata


@pytest.fixture(scope="module")
def tracking() -> Any:
    return make_tracking_dataset(_sequences(), shape=_SHAPE)


@pytest.fixture(params=[Bare, NoTargets], ids=["bare", "no-targets"])
def videos(request, tracking) -> Any:
    return request.param(tracking)


@pytest.mark.required
class TestRecognition:
    def test_streams_and_target_less_triples_are_pure_video(self, videos):
        assert is_pure_video(videos)

    def test_a_tracking_dataset_is_not(self, tracking):
        assert not is_pure_video(tracking)

    def test_image_datasets_are_not(self):
        images = [np.zeros((3, 8, 8), dtype=np.uint8) for _ in range(3)]
        assert not is_pure_video(images)
        assert not is_pure_video(np.zeros((3, 3, 8, 8)))
        assert not is_pure_video([(image, None, {"id": 0}) for image in images])
        assert not is_pure_video([])

    def test_adapting_a_tracking_dataset_returns_it_unchanged(self, tracking):
        assert as_tracking(tracking) is tracking


@pytest.mark.required
class TestAdapter:
    def test_every_frame_has_no_detections(self, videos):
        _, target, metadata = as_tracking(videos)[1]
        assert len(target.frame_tracks) == 20
        assert all(np.size(frame.boxes) == 0 and np.size(frame.track_ids) == 0 for frame in target.frame_tracks)
        assert "id" in metadata

    def test_a_sized_stream_is_counted_without_a_walk(self, tracking):
        videos = Bare(tracking, sized=True)
        adapted = as_tracking(videos)
        assert len(adapted[0][1].frame_tracks) == 20
        assert videos.streams[0].walks == 0

    def test_an_unsized_stream_is_counted_by_one_walk_that_is_kept(self, tracking):
        videos = Bare(tracking, sized=False)
        adapted = as_tracking(videos)
        assert len(adapted[0][1].frame_tracks) == 20
        assert videos.streams[0].walks == 1
        assert len(adapted[0][1].frame_tracks) == 20
        assert videos.streams[0].walks == 0, "the count was not kept: the second stream was walked"

    def test_a_datasets_own_metadata_names_the_adapter(self, tracking):
        videos = Bare(tracking)
        videos.metadata = {"id": "clips"}  # type: ignore[attr-defined]
        assert as_tracking(videos).metadata["id"] == "clips"  # type: ignore[attr-defined]


@pytest.mark.required
class TestAsVideo:
    def test_a_frame_view_matches_the_one_over_the_tracking_dataset(self, videos, tracking):
        bare, reference = SequenceFrames(videos, Stride(2)), SequenceFrames(tracking, Stride(2))
        assert bare.source_indices == reference.source_indices
        assert bare.n_source_frames == reference.n_source_frames == 80
        assert all(np.array_equal(a[0], b[0]) for a, b in zip(bare.stream(), reference.stream(), strict=True)), (
            "the frames differ"
        )
        assert all(np.size(target.boxes) == 0 for _, target, _ in bare.stream())

    def test_outliers_flags_the_planted_frame(self, videos):
        result = Outliers().evaluate(videos)
        assert sorted(result.frames.outliers) == [SourceIndex(*FLASH, "unit")]

    def test_outliers_matches_the_tracking_dataset(self, videos, tracking):
        flagged = lambda result: sorted(result.frames.outliers)  # noqa: E731
        assert flagged(Outliers().evaluate(videos)) == flagged(Outliers().evaluate(tracking))

    def test_duplicates_finds_the_planted_copy(self, videos):
        exact: Any = Duplicates().evaluate(videos).exact
        pairs = {frozenset(map(tuple, group)) for group in exact}
        assert pairs == {frozenset({ORIGINAL, COPY})}

    def test_a_finding_resolves_to_its_frame(self, videos):
        found = SourceLocator(videos)[SourceIndex(*FLASH, "unit")]
        assert np.asarray(found.pixels).mean() == 255

    def test_there_are_no_tracks_to_judge(self, videos):
        with pytest.raises(ValueError, match="no ordered factor lives at"):
            Outliers().evaluate(videos, levels="track")
