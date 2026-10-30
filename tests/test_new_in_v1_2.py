"""Run the example on the New in v1.2 page, so that the per-video finding and its address stay true.

Nothing else executes ``new-in-v1.2.md``. The block evaluates a ``dataset`` the page describes but does
not build: 25 videos of noisy frames, one of them dark throughout.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from dataeval.types import SourceIndex
from tests.quality.test_duplicates import make_tracking_dataset

pytestmark = pytest.mark.infrastructure

PAGE = Path(__file__).resolve().parent.parent / "docs" / "source" / "getting-started" / "new-in-v1.2.md"
FENCE = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)


@pytest.mark.required
def test_the_example_flags_the_dark_video_and_its_frames():
    (block,) = FENCE.findall(PAGE.read_text())
    rng = np.random.default_rng(0)
    videos = [[rng.integers(100, 140, (3, 24, 32), dtype=np.uint8) for _ in range(8)] for _ in range(25)]
    videos[7] = [rng.integers(5, 25, (3, 24, 32), dtype=np.uint8) for _ in range(8)]

    namespace = {"dataset": make_tracking_dataset(videos)}
    exec(compile(block, PAGE.name, "exec"), namespace)
    result = namespace["result"]

    assert set(result.sequences.data()["item_index"]) == {7}
    assert set(result.frames.data()["item_index"]) == {7}
    assert set(result.frames.data()["target_index"]) == set(range(8))
    assert namespace["address"] == SourceIndex(7, 0, "unit")
    frame = namespace["frame"]
    assert (frame.level, frame.item_index) == ("unit", 7)
