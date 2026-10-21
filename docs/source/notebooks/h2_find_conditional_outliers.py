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

# %% [markdown]
# ## Problem statement
#
# Some values are ordinary across the whole dataset and wrong only in context: a car box the size of a truck, one
# snowy frame in a clear video, a frame as bright as noon at 2 a.m., a large box near the horizon. A threshold fitted to
# a whole level passes all of them, because each value is common somewhere.
#
# {class}`~dataeval.quality.Outliers` judges a value against its context with five arguments:
#
# | Argument | Judges each value | Example |
# | --- | --- | --- |
# | `strata=` | within the rows that share a categorical value | a box against boxes of the same vehicle type |
# | `factors=` with `strata=` | a categorical value by how rare it is in its stratum | a weather label against its video |
# | `given=` | against rows with a similar value of another factor | brightness against frames of a similar hour |
# | `combinations=` | as a pair of factors that are unusual together | box size and vertical position |
# | `neighbors=` | against the rows before and after it in its sequence | a frame's exposure against the frames around it |

# %% [markdown]
# ### When to use
#
# Use this guide after [How to find outliers in metadata factors](./h2_find_factor_outliers.py), when a single
# threshold per factor is not enough: your factors depend on a class, a video, a capture condition, or each other.

# %% [markdown]
# ### What you will need
#
# 1. A python environment with the following packages installed:
#    - dataeval
#    - maite-datasets
# 1. A `Metadata` with the factors you want to check

# %% [markdown]
# ## Getting started
#
# Import the required libraries.

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
# ## Building a tracking dataset with known problems
#
# The code below builds eight 200-frame videos. Each frame holds two cars and a truck, each on its own track, and video 3 also
# holds a bus for its first 12 frames. Frame 120 of video 7 is stamped 3 hours late. The frames are blank: every check
# in this guide reads factors, not pixels.

# %%
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

# %% [markdown]
# ## Attaching the factors
#
# Attach the factors at the level they describe. `Metadata` orders frames by video, then frame, and detections by video,
# frame, then track, so build each array in that order.
#
# Per frame (`unit`):
#
# - `weather`: `"snow"` throughout video 5 and `"clear"` elsewhere. Frame 80 of video 0 is planted as `"snow"`.
# - `hour`: the capture hour. Video `v` covers hours `3v` to `3v + 3`.
# - `brightness`: follows the daily cycle of `hour`. Frame 140 of video 0, at about 2 a.m., is planted at daylight level.
# - `exposure`: about `8 + v` in video `v`. Frame 50 of video 4 is planted at 9.0, the exposure of video 1.
#
# Per detection (`instance`):
#
# - `vehicle`: the track's vehicle type.
# - `box_y`: the box's vertical position, from 0 at the top of the frame to 1 at the bottom.
# - `box_size`: grows with `box_y`, since nearer objects sit lower and look larger, and is twice as large for a truck.
#   Two boxes are planted:
#   - the car in video 2, frame 100, track 0, is truck-sized;
#   - the car in video 6, frame 60, track 1, sits near the horizon at the size of a car near the bottom.
#
# `box_y` and `box_size` are fractions of the frame, so boxes from frames of different resolutions compare. For an
# object detection dataset, {func}`~dataeval.core.image_stats` with `stats=ImageStats.DIMENSION_RELATIVE` and
# `per_target=True` computes them from the boxes as `rel_offset_y` and `rel_size`. Attach its result with
# `Metadata.add_factors`.

# %%
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
# ## Judging each value within a stratum
#
# Thresholded over all detections, the truck-sized car passes: trucks of that size are common.

# %%
boxes = ["box_size", "box_y"]
display(Outliers().evaluate(metadata, levels="instance", factors=boxes).data())

# %% [markdown]
# Pass `strata=` to fit one threshold per value of a categorical factor. Each car is then judged against cars only.
# The `population` column names the stratum a row was judged in. For a detection, `target_index` is its position in its
# video.

# %%
by_vehicle = Outliers().evaluate(metadata, levels="instance", factors=boxes, strata=["vehicle"])
display(by_vehicle.data().select("item_index", "target_index", "metric_name", "metric_value", "bound", "population"))

# %% [markdown]
# A stratum takes its values from the level they live at, so a frame factor such as `weather` also stratifies the
# detections in each frame. Several names cross: `strata=["vehicle", "weather"]` judges cars in snow apart from cars in
# clear weather. A null value is a stratum of its own. A continuous factor is refused as a stratum; use `given=` for it.

# %% [markdown]
# ## Checking which populations were judged
#
# A threshold fitted to a few rows flags clean values, so `Outliers` fits one only to a population of at least 20 rows.
# The bus stratum holds 12 detections. `result.populations` lists every population with its size and, where it was
# not judged, the reason.

# %%
display(by_vehicle.populations)

# %% [markdown]
# A population below the minimum produces no findings. Check `populations` before reading an empty result as clean.

# %% [markdown]
# ## Finding rare categorical values
#
# A categorical factor has no location or spread to threshold. With `strata=`, `Outliers` judges it by rarity instead:
# a value is flagged when an exact binomial test shows it occurs in less than 5 % of its stratum, after correcting for
# the number of values tested. Stratify the frames by video with the reserved `item_index` column and name the
# categorical factor.

# %%
rare = Outliers().evaluate(metadata, levels="unit", factors=["weather"], strata=["item_index"])
display(rare.data().select("item_index", "target_index", "observed", "population", "metric_value", "bound"))

# %% [markdown]
# The planted snow frame is flagged. `observed` holds the value, `metric_value` the upper confidence bound on its share
# of the video, and `bound` the 5 % share it falls below. The snowy video is not flagged: snow is its common value.
#
# A value that is rare in every video is not flagged. That is a property of the whole dataset, which
# {class}`~dataeval.bias.Balance` measures, not a defect in one frame. A stratum needs at least 93 rows before one
# occurrence can be judged rare; smaller strata are listed in `populations`.

# %% [markdown]
# ## Judging one factor given another
#
# Brightness depends on the hour, so a single threshold on brightness passes a daylight value at night. The hour is
# continuous, so it cannot be a stratum: strata from hour bins would change with the bin edges. Pass `given=` to judge
# `brightness` against the frames nearest in `hour`.

# %%
with use_seed(0):
    given_hour = Outliers().evaluate(metadata, levels="unit", factors=["brightness"], given={"brightness": "hour"})
display(given_hour.data().select("item_index", "target_index", "metric_name", "metric_value", "bound", "method"))

# %% [markdown]
# `metric_value` is a robust z-score against the 30 or more frames nearest in hour, signed so a negative value is
# darker than its neighbors. The planted frame sits at z ≈ 42. The second row is a clean frame slightly past the lower
# bound; at 1,600 frames the default threshold flags about one clean frame. Read `metric_value` against `bound` to tell
# them apart, or set a stricter threshold for this check only:

# %%
strict = given_hour.with_threshold({"brightness given hour": ("modzscore", 6.0)})
display(strict.data().select("item_index", "target_index", "metric_name", "metric_value", "bound"))

# %% [markdown]
# `with_threshold` refits on the stored scores without scoring again. A key of the mapping names one check by its
# `metric_name`; checks the mapping does not name keep the default.

# %% [markdown]
# ## Finding unusual combinations of factors
#
# A box near the horizon at the size of a nearby car has an ordinary size and an ordinary position. Only the pair is
# unusual. Pass `combinations=` to score each named group of factors by its robust Mahalanobis distance, within each
# stratum.

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
# The box near the horizon (video 6) is flagged by the pair and by neither factor alone. The truck-sized car (video 2)
# is flagged by the pair and by `box_size`. The third row is a clean box slightly past the bound.
#
# Pass `combinations="all"` to score every pair of factors at each level. It fits one robust covariance per pair per
# stratum, so it is slow on many factors and strata; name the groups you care about when you can.

# %% [markdown]
# ## Comparing samples
#
# A factor can hold a list of values per row, such as the speeds measured along a track. Declare it with
# `kinds={"speeds": "sample"}` and pass it as a group of one. Each track's speeds are compared with the population's by
# Wasserstein distance, adjusted for the sample's length.

# %%
speeds = [rng.normal(1.0, 0.1, 40) for _ in range(metadata.level_counts["track"])]
speeds[7] = np.r_[rng.normal(0.5, 0.05, 20), rng.normal(1.5, 0.05, 20)]  # video 2, track 1: two speeds
metadata.add_factors({"speeds": pl.Series([s.tolist() for s in speeds])}, level="track")

with use_seed(0):
    sampled = Outliers().evaluate(metadata, levels="track", combinations=[("speeds",)], kinds={"speeds": "sample"})
display(sampled.data().select("item_index", "target_index", "metric_name", "metric_value", "method"))

# %% [markdown]
# The track that switches between two speeds is flagged by `speeds`. The other rows are the short bus track, flagged by
# the track factors `Metadata` derives, such as `track_length`: `levels="track"` also thresholds every ordered factor
# at that level.

# %% [markdown]
# ## Judging a value against its neighbors along a sequence
#
# Exposure differs from video to video, so the planted frame in video 4, at the exposure of video 1, passes a threshold
# fitted to all frames:

# %%
print(len(Outliers().evaluate(metadata, levels="unit", factors=["exposure"])))

# %% [markdown]
# Pass `neighbors=` to compare each frame with up to five frames on each side of it in its video. A detection's factor
# is compared along its track instead.

# %%
along = Outliers().evaluate(metadata, levels="unit", factors=["exposure"], neighbors=["exposure", "time_s"])
display(along.data().select("item_index", "target_index", "metric_name", "metric_value", "bound", "method"))

# %% [markdown]
# `metric_value` is the frame's own value and `bound` the edge of what its neighbors allow, in the factor's units. The
# planted exposure frame (video 4) sits far below its bound. The two other exposure rows are clean frames slightly past
# theirs; read `metric_value` against `bound` to tell them apart. The late frame in video 7 has two rows: `neighbors`,
# because its stamp is far from its neighbors', and `order`, because the stamps after it are earlier.
#
# A factor is checked for order when its dtype is a date, time or duration, or when it is named `time_s` or `pts`.
# Cast a timestamp stored as a plain number under another name to `pl.Datetime` or `pl.Duration`.
#
# The check follows ramps and level changes, so a slow drift or a scene cut is not flagged. A run of more than five
# out-of-place frames reads as a segment of its own and is not flagged either.

# %% [markdown]
# ## Changing the strata without rebuilding
#
# A result keeps what it needs to re-detect. `stratified` judges the same factors within other strata, and `itemwise`
# removes the strata:

# %%
by_weather = by_vehicle.stratified(metadata, "weather")
print(sorted(by_weather.populations["population"].cast(pl.Utf8).unique().to_list()))
print(len(by_vehicle.itemwise()))

# %% [markdown]
# ## Reading a finding back to its row
#
# `outliers` maps each flagged row's address to the checks that flagged it. A detection is addressed as
# `SourceIndex(video, target_index)`. Read its factors back from the metadata with the same two values:

# %%
print(combined.outliers)

flagged = metadata.rows_at("instance").filter((pl.col("item_index") == 6) & (pl.col("target_index") == 181))
display(flagged.select("item_index", "target_index", "track_id", "vehicle", "box_y", "box_size"))


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
assert flagged["target_index"].to_list() == [horizon_box.key]
distribution = sampled.data().filter(pl.col("method") == "distribution")
assert distribution.select("item_index", "target_index").rows() == [(2, 1)]
assert len(Outliers().evaluate(metadata, levels="unit", factors=["exposure"])) == 0
assert {(4, 50, "neighbors"), (7, 120, "neighbors"), (7, 120, "order")} <= set(
    along.data().select("item_index", "target_index", "method").rows()
)
assert along.data().filter(pl.col("metric_name") == "time_s along sequence")["target_index"].to_list() == [120, 120]

# %% [markdown]
# ## Next steps
#
# - [Data Integrity](../concepts/DataIntegrity.md): the population floor, stratification, rarity, how combinations are
#   flagged, and how a value is judged against its neighbors along a sequence.
# - [How to find outliers in metadata factors](./h2_find_factor_outliers.py): thresholding single factors at each level.
# - [How to trace findings to their source](./h2_trace_findings_to_source.py): retrieve the frame, track or detection a
#   `SourceIndex` names.
