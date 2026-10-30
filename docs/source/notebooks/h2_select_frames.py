# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: dataeval
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Select frames from a video dataset for image tools
#
# Neighboring frames of a video are nearly the same picture. Run an image tool over all of them and the tool counts
# one scene many times, at the cost of decoding every frame. {class}`.SequenceFrames` thins each video to the frames
# you keep, and records what each kept frame stands for.
#
# Use this guide to:
#
# - keep a fixed budget of frames per video by position or by appearance
# - replay a random selection
# - read how many source frames each kept frame stands for
#
# You need `dataeval`. The guide builds its own videos.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval
except Exception:
    pass

# %%
from collections.abc import Iterator
from typing import Any, NamedTuple, cast

import numpy as np

from dataeval.config import set_max_processes
from dataeval.data import EvenlySpaced, Representative, SequenceFrames, Stride
from dataeval.protocols import DatasetMetadata, DatumMetadata, MultiobjectTrackingDatum, VideoFrame
from dataeval.types import SourceIndex

set_max_processes(1)

# %% [markdown]
# ## Build a dataset to select from
#
# Three videos of 60 frames each hold a still scene for 40 frames, then change at every frame for 20.

# %% tags=["hide-input"]
N_FRAMES, STILL = 60, 40


def scene(seed: int) -> list[np.ndarray]:
    """A still scene with sensor noise for 40 frames, then a different picture at every frame."""
    rng = np.random.default_rng(seed)
    still = rng.integers(0, 255, (3, 16, 16))
    noisy = [np.clip(still + rng.integers(-3, 4, still.shape), 0, 255) for _ in range(STILL)]
    busy = [rng.integers(0, 255, (3, 16, 16)) for _ in range(N_FRAMES - STILL)]
    return [frame.astype(np.uint8) for frame in noisy + busy]


class FrameTargets(NamedTuple):
    """The boxes in one frame, which MAITE calls a `SingleFrameObjectTrackingTarget`."""

    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    track_ids: np.ndarray


class VideoTargets(NamedTuple):
    """The boxes in every frame of one video, which MAITE calls a `MultiobjectTrackingTarget`."""

    frame_tracks: list[FrameTargets]


class Frame(NamedTuple):
    """One decoded frame, which MAITE calls a `VideoFrame`."""

    pixels: np.ndarray
    time_s: float
    pts: int
    frame_index: int


class Stream:
    """The frames of one video."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames

    def __iter__(self) -> Iterator[VideoFrame]:
        for index, pixels in enumerate(self._frames):
            yield Frame(pixels=pixels, time_s=index / 30, pts=index, frame_index=index)


class Videos:
    """A minimal MAITE multi-object tracking dataset."""

    def __init__(self, n_videos: int) -> None:
        self.metadata = DatasetMetadata({"id": "videos", "index2label": {0: "car"}})
        self._scenes = [scene(seed) for seed in range(n_videos)]

    def __len__(self) -> int:
        return len(self._scenes)

    def __getitem__(self, index: int) -> MultiobjectTrackingDatum:
        track = FrameTargets(
            boxes=np.array([[2, 2, 8, 8]], dtype=np.float32),
            labels=np.array([0], dtype=np.int64),
            scores=np.ones(1, dtype=np.float32),
            track_ids=np.array([0], dtype=np.int64),
        )
        target = VideoTargets(frame_tracks=[track] * N_FRAMES)
        return Stream(self._scenes[index]), target, cast(DatumMetadata, {"id": f"video_{index}"})


dataset = Videos(3)
print(f"{len(dataset)} videos of {N_FRAMES} frames")

# %% [markdown]
# ## Keep a budget of frames
#
# Each selector takes the budget you can afford:
#
# - {class}`.Stride` fixes the spacing, so a longer video keeps more frames. With `jitter=True` it keeps a random frame
#   from each block, so a periodic scene cannot line up with the spacing.
# - {class}`.EvenlySpaced` fixes the count, so every video keeps the same number of frames however long it is.
# - {class}`.Representative` fixes the count and picks by appearance. It describes every frame, in batches of
#   `batch_size`, and keeps those whose spread matches the video's own.

# %%
selections = {
    "Stride": SequenceFrames(dataset, Stride(10, jitter=True, seed=0)),
    "EvenlySpaced": SequenceFrames(dataset, EvenlySpaced(4, jitter=True, seed=0)),
    "Representative": SequenceFrames(dataset, Representative(6, batch_size=16)),
}
for name, frames in selections.items():
    first = [a.key for a in frames.source_indices if a.item == 0]
    print(f"{name:<15} {len(frames):>2} frames, video 0 keeps {first}")

# %% [markdown]
# Pass a `seed` to replay a jittered selection. Without one, the selection is not reproducible unless {func}`.set_seed`
# was called. {class}`.Representative` needs no seed because it is deterministic.
#
# Use {class}`.Stride` when the spacing in time matters, {class}`.EvenlySpaced` for an equal share per video, and
# {class}`.Representative` when the footage holds long stretches of stillness. It describes frames from raw pixels by
# default, so pass a pretrained {class}`.TorchExtractor` as `extractor` on real footage.

# %% [markdown]
# ## Read what a kept frame stands for
#
# Each kept frame records `frames_represented`, the number of source frames nearer to it than to any other kept frame,
# so a statistic can be weighted back to the whole video. No evaluator reads these weights yet, so a result over kept
# frames describes the selection and not the source video's distribution. {meth}`.SequenceFrames.to_source` gives the
# address of a kept frame in the source dataset.

# %%
frames = selections["Representative"]
represented = [dict(meta)["frames_represented"] for _, _, meta in frames.stream()]
sources = frames.source_indices
for address, weight in zip(sources[:6], represented[:6], strict=False):
    print(f"video {address.item}, frame {address.key:>2} stands for {weight:>4.0f} frames")

kept = frames.to_source([SourceIndex(3)])[0]
print(f"\nview position 3 is {kept!r}")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###


def weights(frames: SequenceFrames) -> list[float]:
    return [cast(dict[str, Any], meta)["frames_represented"] for _, _, meta in frames.stream()]


for name, frames in selections.items():
    for item in range(len(dataset)):
        keys = [a.key for a in frames.source_indices if a.item == item and a.key is not None]
        assert keys == sorted(keys)
        assert len(set(keys)) == len(keys)
        if name != "Stride":
            assert len(keys) == {"EvenlySpaced": 4, "Representative": 6}[name]
    for item in range(len(dataset)):
        owned = [w for a, w in zip(frames.source_indices, weights(frames), strict=True) if a.item == item]
        assert sum(owned) == N_FRAMES
again = SequenceFrames(dataset, Stride(10, jitter=True, seed=0)).source_indices
other = SequenceFrames(dataset, Stride(10, jitter=True, seed=1)).source_indices
assert again == selections["Stride"].source_indices
assert other != again
assert (
    SequenceFrames(dataset, Representative(6, batch_size=16)).source_indices
    == selections["Representative"].source_indices
)
assert kept == sources[3]

# %% [markdown]
# ## Count each video as one unit
#
# Frames of one video share a scene and a camera, so thinning them does not make them independent samples. An evaluator
# that treats them as independent overstates how much data you have. Where the question is about the scene, count the
# video as the unit. [Parity](../concepts/Parity.md) counts a video as one unit however many of its frames the view
# keeps.

# %% [markdown]
# ## Next steps
#
# - [Identify duplicate video](./tt_identify_duplicate_video.py) — Find repeated, re-encoded and leaked footage.
# - [Trace a finding back to its source](./h2_trace_findings_to_source.py) — Follow an address back to its data.
