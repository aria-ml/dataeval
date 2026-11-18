"""Several `VideoTorchExtractor` ways share one walk of each video and each give what they give alone.

Fixtures: random float32 videos of 8x8x3 frames, thirteen, seven and thirty frames long, so clips of three, four and
five frames leave trailing partial clips; each given as a stream that decodes afresh on each walk and counts its walks
and the frames it hands out, as a rank-4 array, as a rank-4 tensor and as a list of frames; and the mock processor and
mock video models of `test_videotorch_extractor.py`, one per clip length, one of them read at a hooked layer.
"""

import weakref
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import torch

from dataeval import Embeddings
from dataeval.extractors import VideoTorchExtractor
from dataeval.extractors._videotorch import _is_mirrored, _mirror
from tests.extractors.test_videotorch_extractor import (
    MockProcessor,
    MockVideoModel,
    ReplayableStream,
    StreamFrame,
    frames_of,
    make_video,
)

pytestmark = [pytest.mark.required, pytest.mark.filterwarnings("ignore::dataeval.exceptions.ExperimentalWarning")]


def _way(num_frames: int, **kwargs: Any) -> VideoTorchExtractor:
    torch.manual_seed(num_frames)
    return VideoTorchExtractor(MockVideoModel(num_frames=num_frames), processor=MockProcessor(), device="cpu", **kwargs)


def _ways() -> dict[str, VideoTorchExtractor]:
    return {
        "a4_mean": _way(4),
        "b3_max": _way(3, clip_aggregation="max"),
        "c5_mean": _way(5),
        "d4_layer": _way(4, layer_name="encoder.1"),
    }


class _Counted(ReplayableStream):
    """A replayable stream that also counts the frames it hands out, over all its walks."""

    def __init__(self, video: np.ndarray) -> None:
        super().__init__(video)
        self.pulled = [0]

    def __iter__(self) -> Iterator[StreamFrame]:
        self.walks += 1
        return frames_of(self.video, self.pulled)


def _videos() -> list[Any]:
    return [_Counted(make_video(13)), make_video(7), torch.as_tensor(make_video(30)), list(make_video(13))]


def test_each_way_equals_its_solo_run():
    ways, videos = _ways(), _videos()
    mirrored = _mirror(ways, videos, range(len(videos)))
    for key, way in ways.items():
        assert np.array_equal(mirrored[key], np.asarray(way(videos))), key


def test_each_stream_is_walked_once_and_each_frame_pulled_once():
    stream = _Counted(make_video(13))
    _mirror(_ways(), [stream], [0])
    assert stream.walks == 1
    assert stream.pulled[0] == 13


def test_no_more_frames_are_held_than_the_longest_clip():
    live: list[weakref.ref[np.ndarray]] = []
    seen: list[int] = []

    class _Watched(ReplayableStream):
        def __iter__(self) -> Iterator[StreamFrame]:
            for i, frame in enumerate(self.video):
                pixels = np.ascontiguousarray(np.moveaxis(frame, -1, 0))
                live.append(weakref.ref(pixels))
                yield StreamFrame(pixels, i / 30, i, i)

    ways = _ways()
    for way in ways.values():
        way._model.register_forward_pre_hook(lambda _module, _args: seen.append(sum(ref() is not None for ref in live)))
    _mirror(ways, [_Watched(make_video(30))], [0])
    assert seen
    assert max(seen) <= 5


def test_a_video_short_for_one_way_raises_for_that_way():
    ways = {"b3": _way(3), "c5": _way(5)}
    with pytest.raises(ValueError, match="shorter than one clip of 5") as raised:
        _mirror(ways, [ReplayableStream(make_video(4))], [7])
    notes = " ".join(getattr(raised.value, "__notes__", []))
    assert "'c5'" in notes
    assert "dataset index 7" in notes


class _Subclass(VideoTorchExtractor):
    calls = 0

    def __call__(self, data: Any) -> Any:
        type(self).calls += 1
        return super().__call__(data)


def _items(make: Any, n: int = 3) -> Any:
    class _Videos:
        def __len__(self) -> int:
            return n

        def __getitem__(self, i: int) -> tuple[Any, int, dict[str, Any]]:
            return make(i), 0, {}

    return _Videos()


def test_a_subclass_that_overrides_call_is_not_mirrored():
    torch.manual_seed(4)
    subclass = _Subclass(MockVideoModel(num_frames=4), processor=MockProcessor(), device="cpu")
    assert not _is_mirrored(subclass)
    assert _is_mirrored(_way(4))

    streams = [ReplayableStream(make_video(9)) for _ in range(3)]
    Embeddings.compute_multi(_items(lambda i: streams[i]), {"a": _way(4), "b": _way(3), "sub": subclass}, batch_size=3)
    assert _Subclass.calls == 1
    assert [stream.walks for stream in streams] == [2, 2, 2]


def test_a_one_shot_stream_with_only_mirrored_ways_is_accepted():
    videos = [make_video(9) for _ in range(3)]
    ways = {"a": _way(4), "b": _way(3)}
    children = Embeddings.compute_multi(_items(lambda i: frames_of(videos[i])), ways, batch_size=3)
    for key, way in ways.items():
        assert np.array_equal(children[key][:], np.asarray(way([frames_of(v) for v in videos]))), key


def test_a_one_shot_stream_with_another_way_raises():
    def walker(batch: list[Any]) -> np.ndarray:
        return np.stack([np.zeros(2, dtype=np.float32) for _ in batch])

    with pytest.raises(ValueError, match="one-shot"):
        Embeddings.compute_multi(
            _items(lambda i: frames_of(make_video(9))), {"a": _way(4), "b": _way(3), "walker": walker}, batch_size=3
        )


class _WritingProcessor(MockProcessor):
    """A processor that writes into the frames it is given, as no ordinary processor does."""

    def __call__(self, frames: Any, return_tensors: str = "pt") -> Any:
        for frame in frames:
            frame *= 2.0
        return super().__call__(frames, return_tensors)


def test_a_processor_that_writes_into_its_frames_does_not_reach_another_way():
    def ways() -> dict[str, VideoTorchExtractor]:
        torch.manual_seed(0)
        return {
            "three": VideoTorchExtractor(MockVideoModel(num_frames=3), processor=_WritingProcessor(), device="cpu"),
            "four": VideoTorchExtractor(MockVideoModel(num_frames=4), processor=_WritingProcessor(), device="cpu"),
        }

    def videos() -> list[np.ndarray]:  # rank-4 arrays are walked frame by frame, so the ways' windows share frames
        return [np.random.default_rng(i).random((12, 8, 8, 3), dtype=np.float32) for i in range(2)]

    shared = ways()
    mirrored = _mirror(shared, videos(), [0, 1])
    for key, way in shared.items():
        assert np.array_equal(mirrored[key], np.asarray(way(videos()))), key


class _OtherArray:
    """An array type that is neither NumPy nor torch, sliced into clips as any such array is."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.shape = data.shape
        self.ndim = data.ndim

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return self.data

    def __getitem__(self, key: Any) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


def test_a_sliced_array_of_another_type_reaches_each_way_as_it_is():
    from dataeval.extractors._videotorch import _is_sliced

    video = _OtherArray(make_video(8))
    assert _is_sliced(video)
    seen: list[type] = []
    ways = {"a": _way(4), "b": _way(3)}
    for way in ways.values():
        split = way._split_video_into_clips

        def recording(given: Any, split: Any = split) -> Any:
            seen.append(type(given))
            return split(given)

        way._split_video_into_clips = recording  # type: ignore[method-assign]
    _mirror(ways, [video], [0])
    assert seen == [_OtherArray, _OtherArray]


class _FailsWhileDecoding(ReplayableStream):
    """A stream that hands out five frames and then fails, as a corrupt file does."""

    def __iter__(self) -> Iterator[StreamFrame]:
        for i, frame in enumerate(self.video[:5]):
            yield StreamFrame(np.ascontiguousarray(np.moveaxis(frame, -1, 0)), i / 30, i, i)
        raise OSError("the stream could not be decoded")


def test_a_stream_that_fails_while_decoding_is_named():
    with pytest.raises(OSError, match="could not be decoded") as raised:
        _mirror({"a": _way(4), "b": _way(3)}, [_FailsWhileDecoding(make_video(9))], [3])
    notes = " ".join(getattr(raised.value, "__notes__", []))
    assert "dataset index 3" in notes
    assert "'a'" in notes
    assert "'b'" in notes


def test_mirrored_ways_leave_the_dataset_arrays_unchanged():
    videos = [np.random.default_rng(i).random((8, 8, 8, 3), dtype=np.float32) for i in range(3)]
    stored = [v.copy() for v in videos]
    torch.manual_seed(0)
    writing = VideoTorchExtractor(MockVideoModel(num_frames=4), processor=_WritingProcessor(), device="cpu")
    ways = {
        "writing": writing,
        "plain": _way(4),
        "flat": lambda batch: np.stack([np.asarray(v).ravel() for v in batch]),
    }
    children = Embeddings.compute_multi(_items(lambda i: videos[i]), ways, batch_size=3)
    for kept, original in zip(videos, stored, strict=True):
        assert np.array_equal(kept, original)
    assert np.array_equal(children["flat"][:], np.stack([v.ravel() for v in stored]))
