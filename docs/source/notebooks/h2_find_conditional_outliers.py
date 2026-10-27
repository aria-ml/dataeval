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
# # How to find outliers that depend on their context
#
# Some values are ordinary across the whole dataset and wrong only in context. A threshold fitted to a whole level
# passes them, because each value is common somewhere. {class}`~dataeval.quality.Outliers` judges a value against its
# context instead. See [Data integrity](../concepts/DataIntegrity.md).
#
# Use it after [How to find outliers in metadata factors](./h2_find_factor_outliers.py), when a single threshold per
# factor is not enough.
#
# You need `dataeval` and `maite-datasets`.

# %% tags=["remove_cell"]
# Google Colab Only
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval maite-datasets
except Exception:
    pass

# %% tags=["remove_cell"]
import warnings

# Reading rows bins the video timestamps, which this guide does not use.
warnings.filterwarnings("ignore", message=".*binned automatically")

# %%
from collections.abc import Iterator
from typing import cast

import numpy as np
import polars as pl
from IPython.display import display
from maite_datasets.multiobject_tracking import (
    MultiobjectTrackingTargetTuple,
    SingleFrameObjectTrackingTargetTuple,
    VideoFrameTuple,
)

from dataeval import Metadata
from dataeval.config import use_seed
from dataeval.protocols import DatasetMetadata, DatumMetadata, MultiobjectTrackingDatum, VideoFrame
from dataeval.quality import Outliers
from dataeval.types import SourceIndex

pl.Config.set_tbl_width_chars(160)

# %% [markdown]
# ## Build an example with known problems
#
# The example builds eight 200-frame videos of cars and a truck, with a bus in video 3 for 12 frames. Each argument gets
# one planted problem:
#
# | Argument | Planted in the example |
# | --- | --- |
# | `strata=` | a car box the size of a truck |
# | `factors=` with `strata=` | one snowy frame in a clear video |
# | `given=` | a frame as bright as noon at 2 a.m. |
# | `combinations=` | a large box near the horizon |
# | `neighbors=` | a frame's exposure out of line with its video |
#
# The factors are `weather`, `hour`, `brightness` and `exposure` for each frame, and `vehicle`, `box_y` and `box_size`
# for each detection.

# %% tags=["hide-input"]
N_VIDEOS, N_FRAMES, HEIGHT, WIDTH = 8, 200, 72, 96
BUS_VIDEO, BUS_FRAMES = 3, 12
LATE_VIDEO, LATE_FRAME = 7, 120
VEHICLES = {0: "car", 1: "car", 2: "truck", 3: "bus"}


def track_ids(video: int, frame: int) -> list[int]:
    """The tracks present in one frame."""
    return [0, 1, 2, 3] if video == BUS_VIDEO and frame < BUS_FRAMES else [0, 1, 2]


def frame_targets(video: int) -> list[SingleFrameObjectTrackingTargetTuple]:
    """One target per frame, holding a box for each track present."""
    targets = []
    for frame in range(N_FRAMES):
        ids = track_ids(video, frame)
        targets.append(
            SingleFrameObjectTrackingTargetTuple(
                boxes=np.tile(np.array([10, 10, 22, 22], dtype=np.float32), (len(ids), 1)),
                labels=np.zeros(len(ids), dtype=np.int64),
                scores=np.ones(len(ids), dtype=np.float32),
                track_ids=np.array(ids, dtype=np.int64),
            )
        )
    return targets


class BlankVideo:
    """A stand-in for a decoded video: the right number of frames, all zeros, one of them stamped 3 hours late."""

    def __init__(self, video: int) -> None:
        self.video = video

    def __iter__(self) -> Iterator[VideoFrame]:
        blank = np.zeros((3, HEIGHT, WIDTH), dtype=np.uint8)
        for index in range(N_FRAMES):
            late = 3 * 3600 if (self.video, index) == (LATE_VIDEO, LATE_FRAME) else 0
            yield VideoFrameTuple(pixels=blank, time_s=index / 30 + late, pts=index, frame_index=index)


class TrackingDataset:
    """A minimal MAITE multi-object tracking dataset."""

    def __init__(self) -> None:
        self.metadata = DatasetMetadata({"id": "conditional-outliers", "index2label": {0: "vehicle"}})
        self._data: list[MultiobjectTrackingDatum] = [
            (
                BlankVideo(video),
                MultiobjectTrackingTargetTuple(frame_tracks=frame_targets(video)),
                cast(DatumMetadata, {"id": f"video_{video}", "height": HEIGHT, "width": WIDTH}),
            )
            for video in range(N_VIDEOS)
        ]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> MultiobjectTrackingDatum:
        return self._data[index]


metadata = Metadata(TrackingDataset())
print(metadata.level_counts)

# %% tags=["hide-input"]
rng = np.random.default_rng(0)

video = np.repeat(np.arange(N_VIDEOS), N_FRAMES)
frame = np.tile(np.arange(N_FRAMES), N_VIDEOS)
weather = np.where(video == 5, "snow", "clear")
weather[(video == 0) & (frame == 80)] = "snow"
hour = video * 3.0 + frame * 0.015
brightness = 120 + 80 * np.cos((hour - 13) / 24 * 2 * np.pi) + rng.normal(0, 4, hour.size)
brightness[(video == 0) & (frame == 140)] = 195.0
exposure = 8.0 + video + np.random.default_rng(1).normal(0, 0.1, video.size)
exposure[(video == 4) & (frame == 50)] = 9.0
metadata.add_factors({"weather": weather, "hour": hour, "brightness": brightness, "exposure": exposure}, level="unit")

detections = [(v, f, t) for v in range(N_VIDEOS) for f in range(N_FRAMES) for t in track_ids(v, f)]
d_video, d_frame, d_track = (np.array(column) for column in zip(*detections, strict=True))
vehicle = np.array([VEHICLES[t] for t in d_track])
box_y = rng.uniform(0.2, 0.9, d_track.size)
box_size = np.where(vehicle == "car", 1.0, 2.0) * (0.02 + 0.08 * box_y) + rng.normal(0, 0.004, d_track.size)

large_car = np.flatnonzero((d_video == 2) & (d_frame == 100) & (d_track == 0))[0]
box_y[large_car], box_size[large_car] = 0.85, 2.0 * (0.02 + 0.08 * 0.85)
near_horizon = np.flatnonzero((d_video == 6) & (d_frame == 60) & (d_track == 1))[0]
box_y[near_horizon], box_size[near_horizon] = 0.25, 0.02 + 0.08 * 0.85
metadata.add_factors({"vehicle": vehicle, "box_y": box_y, "box_size": box_size}, level="instance")

# %% [markdown]
# ## Judge each value within a stratum
#
# Thresholded over all detections, the truck-sized car passes, because trucks of that size are common.

# %%
boxes = ["box_size", "box_y"]
display(Outliers().evaluate(metadata, levels="instance", factors=boxes).data())

# %% [markdown]
# Pass `strata=` to fit one threshold per value of a categorical factor, so each car is judged against cars only. A
# continuous factor cannot be a stratum. Use `given=`.

# %%
by_vehicle = Outliers().evaluate(metadata, levels="instance", factors=boxes, strata=["vehicle"])
display(by_vehicle.data().select("item_index", "target_index", "metric_name", "metric_value", "bound", "population"))

# %% [markdown]
# A threshold needs at least 20 rows. The bus stratum has 12 detections, so `populations` shows it was not judged. Check
# it before reading an empty result as clean.

# %%
display(by_vehicle.populations)

# %% [markdown]
# ### Find rare categorical values
#
# With `strata=`, `Outliers` judges a categorical factor by rarity: a value is flagged when an exact binomial test shows
# it occurs in less than 5 % of its stratum. Stratify the frames by video with the reserved `item_index` column.

# %%
rare = Outliers().evaluate(metadata, levels="unit", factors=["weather"], strata=["item_index"])
display(rare.data().select("item_index", "target_index", "observed", "population", "metric_value", "bound"))

# %% [markdown]
# The planted snow frame is flagged and the snowy video is not, because snow is its common value. A stratum needs at
# least 93 rows before one occurrence can be judged rare.

# %% [markdown]
# ## Judge one factor given another
#
# Brightness depends on the hour, so a single threshold passes a daylight value at night. Pass `given=` to judge
# `brightness` against the frames nearest in `hour`.

# %%
with use_seed(0):
    given_hour = Outliers().evaluate(metadata, levels="unit", factors=["brightness"], given={"brightness": "hour"})
display(given_hour.data().select("item_index", "target_index", "metric_name", "metric_value", "bound", "method"))

# %% [markdown]
# `metric_value` is a robust z-score against the nearest frames. The planted frame is flagged far past its bound.
# A stricter threshold for this check only:

# %%
strict = given_hour.with_threshold({"brightness given hour": ("modzscore", 6.0)})
display(strict.data().select("item_index", "target_index", "metric_name", "metric_value", "bound"))

# %% [markdown]
# `with_threshold` refits on the stored scores without scoring again.

# %% [markdown]
# ## Find unusual combinations of factors
#
# A box near the horizon at the size of a nearby car has an ordinary size and position, and only the pair is unusual.
# Pass `combinations=` to score a group of factors by its robust Mahalanobis distance within each stratum.

# %%
with use_seed(0):
    combined = Outliers().evaluate(
        metadata, levels="instance", factors=boxes, strata=["vehicle"], combinations=[("box_size", "box_y")]
    )
display(
    combined
    .data()
    .filter(pl.col("method") == "mahalanobis")
    .select("item_index", "target_index", "metric_name", "metric_value", "bound", "population")
)

# %% [markdown]
# The box near the horizon (video 6) is flagged by the pair and by neither factor alone, and the truck-sized car
# (video 2) by the pair and by `box_size`.

# %% [markdown]
# ## Judge a value against its neighbors
#
# Exposure differs from video to video, so the planted frame in video 4 passes a threshold fitted to all frames:

# %%
print(len(Outliers().evaluate(metadata, levels="unit", factors=["exposure"])))

# %% [markdown]
# Pass `neighbors=` to compare each frame with up to five frames on each side of it in its video. A detection's factor
# is compared along its track instead.

# %%
along = Outliers().evaluate(metadata, levels="unit", factors=["exposure"], neighbors=["exposure", "time_s"])
display(along.data().select("item_index", "target_index", "metric_name", "metric_value", "bound", "method"))


# %% [markdown]
# `bound` is the edge of what the neighbors allow. The planted frame (video 4) sits far below it, and the late frame in
# video 7 has two rows: `neighbors`, because its stamp is far from its neighbors', and `order`, because the stamps after
# it are earlier. A factor is checked for order when it is a date, time or duration, or is named `time_s` or `pts`. Cast
# other timestamps to `pl.Datetime` or `pl.Duration`. A slow drift, a scene cut and a run of more than five out-of-place
# frames are not flagged.


# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
def in_video(index: int) -> int:
    """A detection's position within its video, which is its `target_index`."""
    return int(np.sum(d_video[:index] == d_video[index]))


horizon_box, large_box = SourceIndex(6, in_video(near_horizon)), SourceIndex(2, in_video(large_car))
assert len(Outliers().evaluate(metadata, levels="instance", factors=boxes)) == 0
assert by_vehicle.outliers == {large_box: ["box_size"]}
bus = by_vehicle.populations.filter(pl.col("population").cast(pl.Utf8) == "vehicle=bus")
assert bus["judged"].to_list() == [False, False]
assert rare.data()["observed"].cast(pl.Utf8).to_list() == ["weather=snow"]
assert rare.outliers == {SourceIndex(0, 80, "unit"): ["weather"]}
assert SourceIndex(0, 140, "unit") in given_hour.outliers
assert strict.outliers == {SourceIndex(0, 140, "unit"): ["brightness given hour"]}
pairs = combined.data().filter(pl.col("method") == "mahalanobis")
assert {(6, horizon_box.key), (2, large_box.key)} <= set(
    zip(pairs["item_index"].to_list(), pairs["target_index"].to_list(), strict=True)
)
assert horizon_box not in by_vehicle.outliers
flagged = metadata.rows_at("instance").filter((pl.col("item_index") == 6) & (pl.col("target_index") == 181))
assert flagged["target_index"].to_list() == [horizon_box.key]
assert len(Outliers().evaluate(metadata, levels="unit", factors=["exposure"])) == 0
assert {(4, 50, "neighbors"), (7, 120, "neighbors"), (7, 120, "order")} <= set(
    along.data().select("item_index", "target_index", "method").rows()
)
assert along.data().filter(pl.col("metric_name") == "time_s along sequence")["target_index"].to_list() == [120, 120]

# %% [markdown]
# ## Next steps
#
# - [Data integrity](../concepts/DataIntegrity.md) — Learn the population floor, stratification, rarity, combinations
#   and neighbor checks.
# - [How to find outliers in metadata factors](./h2_find_factor_outliers.py) — Threshold single factors at each level.
# - [How to trace a finding back to its source](./h2_trace_findings_to_source.py) — Retrieve the frame, track or
#   detection a `SourceIndex` names.
