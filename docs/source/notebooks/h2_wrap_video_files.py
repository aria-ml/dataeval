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
# # How to wrap video files as a multi-object tracking dataset
#
# DataEval reads video through the MAITE multi-object tracking protocol. Each datum is a stream of frames, the tracked
# boxes of every frame, and a metadata dictionary. Footage on disk takes two small classes to fit that: a stream that
# reopens the file, and a dataset that reads the file's properties as factors.
#
# Use this guide to:
#
# - turn video files and their tracking annotations into a dataset that {class}`.Outliers` accepts
# - serve the frame rate and duration of each file as named factors that {class}`.Metadata` validates
# - trace a finding back to the file and frame it came from
#
# You need `dataeval` and `av` (PyAV). The guide writes its own clips.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval av
except Exception:
    pass

# %%
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import NamedTuple, cast

import av
import numpy as np

from dataeval import Metadata
from dataeval.config import set_max_processes
from dataeval.data import SourceLocator
from dataeval.protocols import DatasetMetadata, DatumMetadata, MultiobjectTrackingDatum, VideoFrame
from dataeval.quality import Outliers

set_max_processes(1)

# %% [markdown]
# ## Write sample clips
#
# Three clips of 30 frames show a white box crossing a textured background. The second one has a single frame flashed
# to near white, so there is something for an evaluator to find. The box positions are the tracking annotations.

# %% tags=["hide-input"]
HEIGHT, WIDTH, FPS, N_FRAMES = 120, 160, 15, 30


def write_clip(path: Path, seed: int, flash: int | None = None) -> np.ndarray:
    """Encode one clip and return the box of its single track in every frame."""
    rng = np.random.default_rng(seed)
    background = rng.integers(60, 140, (HEIGHT // 8, WIDTH // 8, 3)).astype(np.uint8).repeat(8, 0).repeat(8, 1)
    boxes = []
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        for i in range(N_FRAMES):
            image = background.copy()
            x = 10 + 4 * i
            image[40:70, x : x + 30] = 255
            if i == flash:
                image[:] = 250
            boxes.append([x, 40, x + 30, 70])
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return np.array(boxes, dtype=np.float32)


folder = Path(tempfile.mkdtemp())
paths = [folder / f"clip_{k}.mp4" for k in range(3)]
boxes = [write_clip(path, seed=k, flash=17 if k == 1 else None) for k, path in enumerate(paths)]
print([path.name for path in paths])

# %% [markdown]
# ## Stream frames from the file
#
# A video stream is an iterable that DataEval walks from the first frame, more than once. Open the file inside
# `__iter__` so that every walk starts fresh, and close it when the walk ends, including when the consumer stops early.
# Each frame carries `pixels` as a `(C, H, W)` array, its own `frame_index`, and `time_s` and `pts` from the file.


# %%
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


class FileStream:
    """The frames of one video file, decoded as they are read."""

    open_files = 0

    def __init__(self, path: Path) -> None:
        self.path = path

    def __iter__(self) -> Generator[VideoFrame, None, None]:
        container = av.open(str(self.path))
        FileStream.open_files += 1
        try:
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                yield Frame(
                    pixels=frame.to_ndarray(format="rgb24").transpose(2, 0, 1),
                    time_s=float(frame.pts * stream.time_base),
                    pts=int(frame.pts),
                    frame_index=index,
                )
        finally:
            container.close()
            FileStream.open_files -= 1


# %% [markdown]
# ## Read the file's properties as factors
#
# The dataset opens each file once for its properties and returns them in the datum's metadata. `frame_rate` and
# `video_duration` are known factors, which {class}`.Metadata` places at the `sequence` level and checks against their
# definitions. `codec_name` has no definition, so it is read as given.


# %%
class VideoFileDataset:
    """A MAITE multi-object tracking dataset over video files."""

    def __init__(self, paths: list[Path], boxes: list[np.ndarray], frame_rates: list[float] | None = None) -> None:
        self.metadata = DatasetMetadata({"id": "video-files", "index2label": {0: "car"}})
        self._paths, self._boxes, self._frame_rates = paths, boxes, frame_rates

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> MultiobjectTrackingDatum:
        path = self._paths[index]
        with av.open(str(path)) as container:
            video = container.streams.video[0]
            info = {
                "id": path.stem,
                "frame_rate": float(video.average_rate),
                "video_duration": float(container.duration / av.time_base),
                "codec_name": video.codec_context.name,
            }
        if self._frame_rates is not None:
            info["frame_rate"] = self._frame_rates[index]
        frame_tracks = [
            FrameTargets(
                boxes=box[None],
                labels=np.array([0], dtype=np.int64),
                scores=np.ones(1, dtype=np.float32),
                track_ids=np.array([0], dtype=np.int64),
            )
            for box in self._boxes[index]
        ]
        return FileStream(path), VideoTargets(frame_tracks=frame_tracks), cast(DatumMetadata, info)


dataset = VideoFileDataset(paths, boxes)
metadata = Metadata(dataset)
print(metadata.rows_at("sequence").select("item_index", "frame_rate", "video_duration", "codec_name"))

# %% [markdown]
# A value that breaks a known factor's definition sets the factor aside, so a bad frame rate cannot reach an analysis
# unnoticed. {attr}`.Metadata.dropped_factors` names the factor, and {meth}`.Metadata.unusable_rows` places its values.

# %%
broken = Metadata(VideoFileDataset(paths, boxes, frame_rates=[-1.0, 15.0, 15.0]))
print(broken.dropped_factors)
print(broken.unusable_rows("frame_rate").sort("value").head(1))

# %% [markdown]
# ## Run an evaluator and follow a finding
#
# Pass the dataset to {class}`.Outliers`. It decodes every frame and reports frames, and with `per_target=True` tracks,
# as {class}`.SourceIndex` addresses of the dataset. {class}`.SourceLocator` resolves an address to the frame.

# %%
result = Outliers().evaluate(dataset)
frames = result.frames.outliers
address = max(frames, key=lambda a: len(frames[a]))
found = SourceLocator(dataset)[address]
print(f"{address!r} is frame {address.key} of {paths[address.item].name}")
print(f"mean brightness {found.pixels.mean():.0f} of 255, flagged on {sorted(frames[address])}")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
first, second = list(FileStream(paths[0])), list(FileStream(paths[0]))
assert len(first) == N_FRAMES
assert [f.frame_index for f in first] == [f.frame_index for f in second] == list(range(N_FRAMES))
assert first[1].time_s > first[0].time_s
assert first[1].pts > first[0].pts
held = FileStream.open_files
walk = iter(FileStream(paths[0]))
next(walk)
assert FileStream.open_files == held + 1
walk.close()
assert FileStream.open_files == held
assert metadata.dropped_factors == {}
assert metadata.rows_at("sequence")["codec_name"].to_list() == ["mpeg4"] * 3
assert broken.dropped_factors == {"frame_rate": ["invalid_known_factor"]}
assert broken.unusable_rows("frame_rate").sort("value")["item_index"][0] == 0
assert (address.item, address.key) == (1, 17)
assert found.pixels.shape == (3, HEIGHT, WIDTH)
assert found.pixels.mean() > 240

# %% [markdown]
# ## Next steps
#
# - [Identify duplicate video](./tt_identify_duplicate_video.py) — Find repeated, re-encoded and leaked footage in a
#   multi-object tracking dataset.
# - [How to trace a finding back to its source](./h2_trace_findings_to_source.py) — Follow an address through views to
#   the data on disk.
