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
# # Defer image and video decoding
#
# {class}`.Metadata` and the evaluators built on it read a dataset's targets and metadata and never use its pixels, but
# `dataset[index]` still decodes the image or the video frames before it returns them. Return a lazy object that knows
# its shape and decodes on first use, and a walk that never reads pixels decodes nothing. See
# [Metadata levels](../concepts/MetadataLevels.md) for what {class}`.Metadata` keeps.
#
# Use it for a dataset you write yourself. The downloadable datasets of `maite-datasets` take `lazy=True`.
#
# You need `dataeval`, `pillow` and `av` (PyAV). This guide writes its own files.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval pillow av
except Exception:
    pass

# %%
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import av
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from dataeval import Metadata
from dataeval.config import set_max_processes
from dataeval.core import image_stats
from dataeval.data import Resize, View
from dataeval.flags import ImageStats
from dataeval.protocols import DatasetMetadata, DatumMetadata

set_max_processes(1)

decodes: Counter[str] = Counter()  # how many images and frames each walk decodes

# %% tags=["hide-input"]
folder = Path(tempfile.mkdtemp())
rng = np.random.default_rng(0)

image_paths = []
for i in range(40):
    image_paths.append(folder / f"image_{i:02d}.png")
    Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(image_paths[-1])

HEIGHT, WIDTH, N_FRAMES = 120, 160, 20
clip_paths = []
for k in range(4):
    clip_paths.append(folder / f"clip_{k}.mp4")
    with av.open(str(clip_paths[-1]), "w") as container:
        stream = container.add_stream("mpeg4", rate=15)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        background = rng.integers(60, 140, (HEIGHT // 8, WIDTH // 8, 3)).astype(np.uint8).repeat(8, 0).repeat(8, 1)
        for i in range(N_FRAMES):
            image = background.copy()
            image[40:70, 10 + 4 * i : 40 + 4 * i] = 255
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

# %% [markdown]
# ## Return a lazy image
#
# `LazyImage` satisfies {class}`.Array`. It reads its shape from the file header and decodes on the first numpy access.
# The dataset returns one in place of the decoded array, so building {class}`.Metadata` reads no pixels.


# %%
def read_image(path: Path) -> NDArray[np.uint8]:
    """Decode an image file to a channels-first (C, H, W) array."""
    decodes["image"] += 1
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)


class LazyImage:
    """An image file that decodes on first use and keeps the array for the life of the object."""

    def __init__(self, path: Path) -> None:
        self._path, self._array = path, None
        with Image.open(path) as image:
            width, height = image.size
        self.shape, self.ndim = (3, height, width), 3

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> NDArray[Any]:
        if self._array is None:
            self._array = read_image(self._path)
        return self._array if dtype is None else self._array.astype(dtype)

    def __getitem__(self, key: Any) -> Any:
        return np.asarray(self)[key]

    def __iter__(self):
        return iter(np.asarray(self))

    def __len__(self) -> int:
        return self.shape[0]


# %% tags=["hide-input"]
class ImageDataset:
    """A classification dataset over image files that returns each image decoded or lazy."""

    def __init__(self, paths: list[Path], lazy: bool) -> None:
        self._paths, self._lazy = paths, lazy
        self.metadata = DatasetMetadata(id="images", index2label={0: "a", 1: "b"})

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> tuple[Any, NDArray[np.float32], DatumMetadata]:
        path = self._paths[index]
        image = LazyImage(path) if self._lazy else read_image(path)
        return (
            image,
            np.eye(2, dtype=np.float32)[index % 2],
            DatumMetadata(id=index, **{"weather": ("clear", "rainy")[index % 3 % 2]}),
        )


# %%
for lazy in (False, True):
    decodes.clear()
    factors = Metadata(ImageDataset(image_paths, lazy)).factor_names
    print(f"{'lazy ' if lazy else 'eager'}: {decodes['image']} images decoded to build the metadata")

# %% [markdown]
# The eager dataset decodes every image to build the metadata and the lazy one decodes none. An evaluator that reads
# the pixels, such as {func}`.image_stats`, decodes each image on demand and works unchanged. `dataset[index]` returns a
# new object on each call, so the cache lasts for one read, and a view operation that reads the pixels, such as
# {class}`.Resize`, decodes every item even when the items are lazy. Defer only the decode: work that `__getitem__`
# does for the item, such as computing features, still runs on every read.

# %%
lazy_images = ImageDataset(image_paths, lazy=True)

decodes.clear()
stats = image_stats(lazy_images, stats=ImageStats.PIXEL, normalize_pixel_values=False)
print(f"image_stats: {decodes['image']} images decoded for {stats['image_count']} images")

decodes.clear()
resized = Metadata(View(lazy_images, Resize((32, 32)))).factor_names
print(f"Resize view: {decodes['image']} images decoded to build the metadata")

# %% [markdown]
# ## Return lazy video frames
#
# Building metadata for a tracking dataset reads each frame's number, timings and size, so an eager stream decodes
# every frame to build it. `LazyFrames` reads the timings from the file's packets and yields frames whose `pixels` read
# their shape from the header and decode on first use.


# %%
class Frame(NamedTuple):
    """One frame, which MAITE calls a `VideoFrame`."""

    pixels: Any
    time_s: float
    pts: int
    frame_index: int


class LazyPixels:
    """The pixels of one frame, decoded from the start of the file on first use."""

    def __init__(self, path: Path, index: int, shape: tuple[int, int, int]) -> None:
        self._path, self._index, self._array = path, index, None
        self.shape, self.ndim = shape, 3

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> NDArray[Any]:
        if self._array is None:
            with av.open(str(self._path)) as container:
                for index, frame in enumerate(container.decode(container.streams.video[0])):
                    decodes["frame"] += 1
                    if index == self._index:
                        self._array = frame.to_ndarray(format="rgb24").transpose(2, 0, 1)
                        break
            if self._array is None or self._array.shape != self.shape:
                raise ValueError(
                    f"{self._path.name}: frame {self._index} does not match its header shape {self.shape}."
                )
        return self._array if dtype is None else self._array.astype(dtype)

    def __getitem__(self, key: Any) -> Any:
        return np.asarray(self)[key]

    def __iter__(self):
        return iter(np.asarray(self))

    def __len__(self) -> int:
        return self.shape[0]


def packet_times(packets: Any) -> list[int]:
    """Return the presentation timestamps of a stream's packets in display order."""
    sized = [packet for packet in packets if packet.size]
    if any(packet.pts is None for packet in sized):
        raise ValueError("A packet has no timestamp, so the frame timings cannot be read without decoding.")
    return sorted(packet.pts for packet in sized)


class LazyFrames:
    """The frames of one video file, with timings from the packets and pixels that decode on first use."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __iter__(self):
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            shape = (3, stream.codec_context.height, stream.codec_context.width)
            time_base, times = stream.time_base, packet_times(container.demux(stream))
        for index, pts in enumerate(times):
            yield Frame(LazyPixels(self.path, index, shape), float(pts * time_base), pts, index)


# %% tags=["hide-input"]
class FrameTargets(NamedTuple):
    """The boxes in one frame, which MAITE calls a `SingleFrameObjectTrackingTarget`."""

    boxes: NDArray[np.float32]
    labels: NDArray[np.int64]
    scores: NDArray[np.float32]
    track_ids: NDArray[np.int64]


class VideoTargets(NamedTuple):
    """The boxes in every frame of one video, which MAITE calls a `MultiobjectTrackingTarget`."""

    frame_tracks: list[FrameTargets]


class FileFrames:
    """The frames of one video file, decoded as they are walked."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __iter__(self):
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                decodes["frame"] += 1
                pts = int(frame.pts)
                yield Frame(
                    frame.to_ndarray(format="rgb24").transpose(2, 0, 1), float(pts * stream.time_base), pts, index
                )


class VideoDataset:
    """A multi-object tracking dataset over video files that walks each file with the stream class it is given."""

    def __init__(self, paths: list[Path], stream: type) -> None:
        self._paths, self._stream = paths, stream
        self.metadata = DatasetMetadata(id="clips", index2label={0: "box"})

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> tuple[Any, VideoTargets, DatumMetadata]:
        tracks = [
            FrameTargets(
                boxes=np.array([[10 + 4 * i, 40, 40 + 4 * i, 70]], dtype=np.float32),
                labels=np.zeros(1, dtype=np.int64),
                scores=np.ones(1, dtype=np.float32),
                track_ids=np.zeros(1, dtype=np.int64),
            )
            for i in range(N_FRAMES)
        ]
        path = self._paths[index]
        return self._stream(path), VideoTargets(tracks), DatumMetadata(id=path.stem)


# %%
for stream in (FileFrames, LazyFrames):
    decodes.clear()
    factors = Metadata(VideoDataset(clip_paths, stream)).factor_names
    print(f"{stream.__name__:10}: {decodes['frame']} frames decoded to build the metadata")

# %% [markdown]
# The eager stream decodes every frame and the lazy one decodes none. A lazy frame decodes from the start of the file
# when its pixels are read, so a walk that reads the pixels of many frames, such as {class}`.Outliers`, should use the
# eager stream.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
eager_images, lazy_images = ImageDataset(image_paths, lazy=False), ImageDataset(image_paths, lazy=True)

decodes.clear()
lazy_frame = Metadata(lazy_images).dataframe
assert decodes["image"] == 0
decodes.clear()
eager_frame = Metadata(eager_images).dataframe
assert decodes["image"] >= len(image_paths)
assert lazy_frame.equals(eager_frame)

from dataeval.protocols import Array  # noqa: E402

image = lazy_images[3][0]
assert isinstance(image, Array)
assert image.shape == eager_images[3][0].shape == (3, 256, 256)
assert np.array_equal(np.asarray(image), eager_images[3][0])

decodes.clear()
first, second = np.asarray(lazy_images[0][0]), np.asarray(lazy_images[0][0])
assert decodes["image"] == 2
assert np.array_equal(first, second)

decodes.clear()
image_stats(lazy_images, stats=ImageStats.PIXEL, normalize_pixel_values=False)
assert decodes["image"] >= len(image_paths)

decodes.clear()
assert Metadata(View(lazy_images, Resize((32, 32)))).factor_names
assert decodes["image"] >= len(image_paths)

decodes.clear()
lazy_clips = Metadata(VideoDataset(clip_paths, LazyFrames))
lazy_rows = lazy_clips.dataframe
assert decodes["frame"] == 0
decodes.clear()
eager_rows = Metadata(VideoDataset(clip_paths, FileFrames)).dataframe
assert decodes["frame"] >= len(clip_paths) * N_FRAMES
assert lazy_rows.equals(eager_rows)

lazy_walk, eager_walk = list(LazyFrames(clip_paths[0])), list(FileFrames(clip_paths[0]))
assert [f.pts for f in lazy_walk] == [f.pts for f in eager_walk]
assert [f.time_s for f in lazy_walk] == [f.time_s for f in eager_walk]
assert [f.frame_index for f in lazy_walk] == list(range(N_FRAMES))
assert lazy_walk[7].pixels.shape == eager_walk[7].pixels.shape == (3, HEIGHT, WIDTH)
decodes.clear()
assert np.array_equal(np.asarray(lazy_walk[7].pixels), eager_walk[7].pixels)
assert decodes["frame"] == 8  # frame 7 is the eighth frame from the start of the file

try:
    np.asarray(LazyPixels(clip_paths[0], 7, (3, HEIGHT + 1, WIDTH)))
except ValueError:
    pass
else:
    raise AssertionError("a header shape that the decoded frame does not match must raise")

assert packet_times([
    SimpleNamespace(size=5, pts=20),
    SimpleNamespace(size=5, pts=10),
    SimpleNamespace(size=0, pts=None),
]) == [10, 20]
try:
    packet_times([SimpleNamespace(size=5, pts=None)])
except ValueError:
    pass
else:
    raise AssertionError("a packet with no timestamp must raise")

# %% [markdown]
# ## Next steps
#
# - [Metadata levels](../concepts/MetadataLevels.md) — Understand why metadata is kept at the level where it was
#   measured.
# - [Keep metadata in sync with its dataset and cache it](./h2_sync_and_cache_metadata.py) — Save the metadata so a
#   dataset walk happens once.
# - [Wrap video files as a multi-object tracking dataset](./h2_wrap_video_files.py) — Serve a video file's frames and
#   properties as a tracking dataset.
