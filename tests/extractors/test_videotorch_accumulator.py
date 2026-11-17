"""`VideoTorchExtractor`'s clip loop as start, push and finish steps gives what a call gives.

Fixtures: a random float32 video of thirteen 8x8x3 frames, so four-frame clips leave a trailing partial clip, given as a
stream that decodes afresh on each walk, a rank-4 array and a list of frames; and the mock processor and mock video
model of `test_videotorch_extractor.py`.
"""

import numpy as np
import pytest

from dataeval.extractors import VideoTorchExtractor
from tests.extractors.test_videotorch_extractor import MockProcessor, MockVideoModel, ReplayableStream, make_video

pytestmark = pytest.mark.required


def _kinds(video: np.ndarray) -> dict[str, object]:
    return {"stream": ReplayableStream(video), "array": video, "list": list(video)}


@pytest.mark.parametrize("aggregation", ["mean", "max"])
@pytest.mark.parametrize("kind", ["stream", "array", "list"])
def test_the_accumulator_equals_a_call(aggregation, kind):
    extractor = VideoTorchExtractor(
        MockVideoModel(), processor=MockProcessor(), clip_aggregation=aggregation, device="cpu"
    )
    video = _kinds(make_video(13))[kind]
    running = extractor._start()
    for clip in extractor._split_video_into_clips(video):
        extractor._push(running, clip)
    row = extractor._finish(running, 0)
    assert np.array_equal(row.cpu().numpy(), np.asarray(extractor([video]))[0])


def test_finishing_with_no_clip_raises_the_short_video_error():
    extractor = VideoTorchExtractor(MockVideoModel(), processor=MockProcessor(), device="cpu")
    with pytest.raises(ValueError, match="shorter than one clip"):
        extractor._finish(extractor._start(), 0)
