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
# # Check drone footage before you train a detector
#
# This tutorial shows you how to find what in real drone footage may get in the way of training a vehicle detector.
# You count what the footage holds, measure how much of it is new, how the camera moves, whether the labels stay
# consistent and how large the vehicles are, and read each result against one use case.
#
# Estimated time to complete: 20 minutes
#
# Relevant ML stages: [Data Engineering](../getting-started/roles/ML_Lifecycle.md#data-engineering)
#
# Relevant personas: [Data Engineer](../getting-started/roles/data_engineer.md), [ML Engineer](../getting-started/roles/ml_engineer.md)
#
# ## What you will do
#
# - Load a drone tracking dataset and count what each video holds
# - Measure how much of each video is new
# - Check how the camera moves
# - Check that labels stay consistent along each track
# - Check the size and crowding of the vehicles
# - Gather the findings in one table
#
# ## What you will learn
#
# - Why you count videos and vehicles, not frames and boxes, to judge how much footage you have
# - Why the frames of one video are not independent samples
# - How camera motion, label consistency and object size bear on training a detector
#
# ## What you will need
#
# - A Python environment with `dataeval[opencv]`, `maite-datasets[hf]>=0.0.25` and `matplotlib` installed
# - About 2 GB of free disk space for the 1.7 GB dataset
# - Time for the first download: about 26 minutes without a Hugging Face token, less with one
# - About 10 minutes to run the notebook on a CPU, after the download
#

# %% [markdown]
# ## Introduction
#
# A detector learns from the footage it is given. Before you spend time training one, it pays to know what that
# footage holds: how many distinct views it gives, whether the conditions you care about are covered, whether the labels
# stay consistent and whether the objects are large enough to learn. Each of these is a property of the data, so you
# can measure it before any model exists.
#
# This tutorial works through one use case. You are asked to train a detector that finds vehicles (cars, vans, trucks
# and buses) in footage from a drone flying over road intersections, by day and by night. You have been handed the
# validation videos of a public drone benchmark to start from. Each section ends with how its finding may hinder
# training for that detector. No model is trained here, so each of those statements is a risk to check, not a measured
# effect.
#

# %% [markdown]
# ## Setup
#

# %% tags=["remove_cell"]
# Google Colab Only
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval[opencv] "maite-datasets[hf]>=0.0.25" matplotlib
except Exception:
    pass

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from IPython.display import display
from maite_datasets.multiobject_tracking import VisDroneMOT
from matplotlib.patches import Rectangle

from dataeval import Metadata
from dataeval.config import set_max_processes
from dataeval.core import ego_stats, image_stats, track_stats
from dataeval.data import Indices, SequenceFrames, View
from dataeval.exceptions import ExperimentalWarning
from dataeval.flags import ImageStats
from dataeval.quality import Duplicates

set_max_processes(4)
_ = pl.Config.set_tbl_width_chars(160)
# Reading rows bins the frame counts, which this tutorial does not use.
warnings.filterwarnings("ignore", message=".*binned automatically")
# Edge directionality is experimental.
warnings.filterwarnings("ignore", category=ExperimentalWarning)

# %% [markdown]
# ## Load the footage and count what it holds
#
# You will work with VisDrone2019-MOT, the drone tracking benchmark of
# [Zhu et al. (2021)](https://arxiv.org/abs/2001.06303), through the
# [Voxel51 copy of its validation videos](https://huggingface.co/datasets/Voxel51/visdrone-mot). It is shared under the
# [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) license, which allows non-commercial use
# with attribution and asks that adaptations carry the same license. The first run downloads about 1.7 GB. The frames
# carry no timestamps, so every gap and length below is counted in frames.
#

# %%
dataset = VisDroneMOT(root="./data", image_set="val", download=True)
metadata = Metadata(dataset)

# %% [markdown]
# Each item of the dataset is one video, with the boxes, classes and track ids of every frame. {class}`.Metadata` holds
# them at four levels: the video, the frame, the track and the single detection. Keep the four vehicle classes of the
# use case, and count what each video holds.
#

# %%
VEHICLES = ["car", "van", "truck", "bus"]
vehicle_ids = [index for index, name in dataset.index2label.items() if name in VEHICLES]

boxes = (
    metadata
    .rows_at("instance")
    .filter(pl.col("class_label").is_in(vehicle_ids))
    .with_columns(
        box_width=pl.col("box").arr.get(2) - pl.col("box").arr.get(0),
        box_height=pl.col("box").arr.get(3) - pl.col("box").arr.get(1),
    )
)
frames = (
    metadata
    .rows_at("unit")
    .group_by("item_index")
    .agg(pl.len().alias("frames"), pl.col("width").first(), pl.col("height").first())
)
vehicles = boxes.group_by("item_index").agg(
    pl.len().alias("vehicle_boxes"), pl.col("track_id").n_unique().alias("vehicle_tracks")
)
videos = (
    metadata
    .rows_at("sequence")
    .select("item_index", pl.col("item_id").alias("video"), "scene_type", "time_of_day")
    .join(frames, on="item_index")
    .join(vehicles, on="item_index", how="left")
    .fill_null(0)
    .sort("item_index")
)
item_of = dict(zip(videos["video"], videos["item_index"], strict=True))
display(videos.drop("item_index"))

# %% [markdown]
# The 114,132 boxes come from 7 videos. One video, `uav0000268_05773_v`, holds 978 of the 2,846 frames, about a third
# of the footage. Three videos show roads or a sporting event rather than intersections, and together they hold 1,805
# frames, about 63%. The sporting event holds no vehicles at all. The night footage is a single video, and its 5,505
# vehicle boxes belong to 28 vehicles. The dusk footage is a single video too.
#
# Seven videos are too few for a bias analysis of these conditions. Every frame of a video shares its time of day and
# its scene, so the evidence about a condition counts videos, not frames, and seven cannot support a test. See
# [Untestable factors and cells](../concepts/Parity.md#untestable-factors-and-cells) for why, and
# [Identify bias and correlations](./tt_identify_bias.py) for the analysis on a dataset large enough for it.
#
# For the detector, everything it learns about vehicles at night would come from 28 vehicles filmed on one flight,
# and 3 of the 7 videos, holding 63% of the frames, show scenes other than the one it will work in.
#

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(dataset) == 7
assert metadata.rows_at("unit").height == 2846
assert metadata.rows_at("track").height == 758
assert metadata.rows_at("instance").height == 114132
assert boxes.height == 40286
assert boxes.select("item_index", "track_id").n_unique() == 246
by_video = {r["video"]: r for r in videos.iter_rows(named=True)}
assert by_video["uav0000268_05773_v"]["frames"] == 978
assert videos.filter(pl.col("scene_type") != "intersection")["frames"].sum() == 1805
assert videos.filter(pl.col("scene_type") != "intersection").height == 3
night = videos.filter(pl.col("time_of_day") == "night")
assert night.height == 1
assert night["vehicle_boxes"][0] == 5505
assert night["vehicle_tracks"][0] == 28
assert videos.filter(pl.col("time_of_day") == "dusk").height == 1
assert by_video["uav0000086_00000_v"]["vehicle_boxes"] == 0

# %% [markdown]
# ## Measure how much of each video is new
#
# Consecutive frames of a video are nearly the same picture. {class}`.Duplicates` measures how much of each video
# repeats the frame before it. It hashes every frame and counts a frame as **redundant** when its hash is close to its
# neighbor's. The hash keeps the coarse layout of the scene and drops small detail such as a single vehicle, so a
# redundant frame shows the same scene as the frame before it, not necessarily the same vehicles. The cell reads every
# frame, so it takes several minutes on a CPU.
#

# %%
redundancy = Duplicates().evaluate(dataset).aggregate_by_sequence().sort("sequence")
display(
    redundancy
    .join(videos.select(pl.col("item_index").alias("sequence"), "video"), on="sequence")
    .select("video", "n_frames", "redundant_fraction", "longest_run", "shared_with")
    .sort("video")
)

# %% [markdown]
# About 96% of the 2,846 frames repeat the frame before them. Every video is mostly repeats, from about 79% of the
# frames of `uav0000305_00000_v` to almost all of them. In `uav0000086_00000_v` and `uav0000268_05773_v` the longest
# run of redundant frames is the whole video: each frame stays close to the one before it from start to end. No frame
# matches a frame of another video (`shared_with` is 0), so a split that keeps each video whole puts all near-copies on
# one side. [Identify duplicate video](./tt_identify_duplicate_video.py) shows how to check a split for leakage, and
# [Data leakage](../concepts/Leakage.md) explains why a split has to separate sources, not only samples.
#
# For the detector, the 2,846 frames hold far fewer distinct views than their count suggests. A split by frame
# rather than by video would put near-copies of one scene in both training and validation, and the validation score
# could then reward recognizing frames seen in training.
#

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
frac = redundancy["redundant_fraction"].to_numpy()
assert 0.94 <= redundancy["redundant_frames"].to_numpy().sum() / redundancy["n_frames"].to_numpy().sum() <= 0.98
assert 0.785 <= frac.min() <= 0.80
assert frac.max() >= 0.995
assert (redundancy["shared_with"] == 0).all()
whole = redundancy.filter(pl.col("longest_run") == pl.col("n_frames"))["sequence"].to_list()
assert whole == sorted([item_of["uav0000086_00000_v"], item_of["uav0000268_05773_v"]])

# %% [markdown]
# ## Check how the camera moves
#
# A drone camera pans, tilts and zooms as it flies. {func}`~dataeval.core.ego_stats` measures that motion between each
# pair of consecutive frames from how the whole picture moves. Where it cannot separate the camera's own motion, for
# example from that of moving vehicles, it marks the reading as not trusted, and this section uses only the trusted
# frames. The cell reads every frame, so it takes a few minutes on a CPU.
#

# %%
camera = ego_stats(dataset)["stats"]
motion = pl.DataFrame({name: values for name, values in camera.items() if name != "motion"})

# %% [markdown]
# Fast camera motion can smear a frame. Edge directionality, an experimental statistic of
# {func}`~dataeval.core.image_stats`, compares how wide the edges of a frame are in its two directions. It is near 0
# when edges are equally sharp both ways and grows when one direction is smeared, which is what motion blur does (see
# [Blur statistics](../concepts/BlurStatistics.md)). Measure it on every frame of `uav0000182_00000_v`, one of the two
# road videos, and set it beside the speed of the camera's pan, in pixels per frame.
#

# %%
road_item = item_of["uav0000182_00000_v"]
edges = image_stats(
    SequenceFrames(View(dataset, Indices([road_item]))),
    stats=ImageStats.VISUAL_BLUR_EDGE_DIRECTIONALITY,
    per_target=False,
)
road = (
    motion
    .filter(pl.col("item_index") == road_item)
    .sort("unit_index")
    .with_columns(frame=pl.int_range(pl.len()), blur_edge_directionality=edges["stats"]["blur_edge_directionality"])
    .filter(pl.col("ego_trusted"))
)
fast = road["pan_speed"] > road["pan_speed"].quantile(0.75)
display(
    road
    .with_columns(fastest_quarter=fast)
    .group_by("fastest_quarter")
    .agg(pl.len().alias("frames"), pl.col("pan_speed").median(), pl.col("blur_edge_directionality").median())
    .sort("fastest_quarter")
)

# %%
fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
top.plot(road["frame"], road["pan_speed"], color="tab:blue")
top.set_ylabel("pan speed\n(pixels per frame)")
bottom.plot(road["frame"], road["blur_edge_directionality"], color="tab:orange")
bottom.set_ylabel("edge\ndirectionality")
bottom.set_xlabel("frame of uav0000182_00000_v")
fig.tight_layout()
plt.show()

# %% [markdown]
# The two curves rise together. In the fastest quarter of the frames, the median edge directionality is more than ten
# times its median in the other three quarters, which is the pattern motion blur along the camera's path would
# produce. The two frames below are the least and the most lopsided frames of the video.
#

# %%
road_frames = SequenceFrames(View(dataset, Indices([road_item])))
least = road.sort("blur_edge_directionality")["frame"][0]
most = road.sort("blur_edge_directionality")["frame"][-1]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for ax, index, title in zip(axes, (least, most), ("least lopsided", "most lopsided"), strict=True):
    image = np.moveaxis(np.asarray(road_frames[int(index)][0]), 0, -1)
    ax.imshow(image)
    ax.set_title(f"frame {index}, {title}")
    ax.axis("off")
fig.tight_layout()
plt.show()

# %% [markdown]
# This video also zooms the most of the seven, so the same vehicle appears at different sizes within one flight. The
# camera of the night video also pans fast at times. Compare its frames with each other rather than with daytime
# frames, since noise also moves these measurements.
#
# For the detector, frames from fast pans, and the vehicles in them, may be smeared.
# [Camera motion as a per-frame factor](../concepts/MetadataLevels.md#camera-motion-as-a-per-frame-factor) shows how
# to carry the camera's motion into the rest of an analysis.
#

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert motion.filter(pl.col("item_index") == road_item)["ego_trusted"].to_numpy().mean() >= 0.99
slow_dir = np.median(road.filter(~fast)["blur_edge_directionality"].to_numpy())
fast_dir = np.median(road.filter(fast)["blur_edge_directionality"].to_numpy())
assert slow_dir < 0.03
assert fast_dir > 0.15
assert fast_dir / slow_dir > 10
trusted = motion.filter(pl.col("ego_trusted"))
zoom = trusted.group_by("item_index").agg(pl.col("zoom_rate").abs().quantile(0.9)).sort("zoom_rate")
assert zoom[-1, "item_index"] == road_item
night_pan = np.quantile(
    trusted.filter(pl.col("item_index") == item_of["uav0000117_02622_v"])["pan_speed"].to_numpy(), 0.9
)
assert night_pan > 10

# %% [markdown]
# ## Check that labels stay consistent along each track
#
# A track follows one vehicle from frame to frame, so its class should not change and it should not drop out while the
# vehicle is in view. {func}`~dataeval.core.track_stats` summarizes every track. `label_confidence` is the share of a
# track's frames that carry its most common class, so it is 1 when the class never changes. `n_gaps` counts the
# times a track disappears and comes back, and `total_gap_length` counts the frames it is missing.
#

# %%
summary = track_stats(dataset)["stats"]
columns = ("item_index", "track_ids", "labels", "label_confidence", "n_appearances", "n_gaps", "total_gap_length")
tracks = pl.DataFrame({name: values for name, values in summary.items() if name in columns}).filter(
    pl.col("labels").is_in(vehicle_ids)
)

classes = metadata.rows_at("instance").group_by("item_index", "track_id").agg(pl.col("class_label").unique())
flips = tracks.filter(pl.col("label_confidence") < 1).join(
    classes, left_on=["item_index", "track_ids"], right_on=["item_index", "track_id"]
)
flips = flips.with_columns(
    pair=pl.Series(["/".join(sorted(dataset.index2label[c] for c in found)) for found in flips["class_label"]])
)
display(
    flips
    .join(videos.select("item_index", "video"), on="item_index")
    .select("video", "track_ids", "pair", "label_confidence", "n_appearances")
    .sort("video", "track_ids")
)

# %% [markdown]
# Six of the 246 vehicle tracks change class along the way: five switch between car and van, and one between car and
# truck. Track 87 of the night video is one of them. Below, in red, is its box in the last frame labeled car and in the
# first frame labeled van.
#

# %%
units = metadata.rows_at("unit").group_by("item_index").agg(pl.col("unit_index").min().alias("first_unit"))
detections = (
    metadata
    .rows_at("instance")
    .join(units, on="item_index")
    .with_columns(frame=pl.col("unit_index") - pl.col("first_unit"))
)


def track_path(item: int, track: int) -> pl.DataFrame:
    """Every detection of one track, in frame order."""
    return detections.filter((pl.col("item_index") == item) & (pl.col("track_id") == track)).sort("frame")


def window(item: int, around: list[list[float]], pad: int) -> tuple[int, int, int, int]:
    """The rectangle around some boxes, grown by `pad` pixels and kept inside the frame."""
    size = videos.filter(pl.col("item_index") == item).row(0, named=True)
    x0 = max(int(min(b[0] for b in around)) - pad, 0)
    y0 = max(int(min(b[1] for b in around)) - pad, 0)
    x1 = min(int(max(b[2] for b in around)) + pad, size["width"])
    y1 = min(int(max(b[3] for b in around)) + pad, size["height"])
    return x0, y0, x1, y1


def show(ax, item: int, frame: int, area: tuple[int, int, int, int], box: list[float] | None, title: str) -> None:
    """Draw one area of one frame of one video, with a box in red where there is one."""
    image = np.moveaxis(np.asarray(SequenceFrames(View(dataset, Indices([item])))[frame][0]), 0, -1)
    x0, y0, x1, y1 = area
    ax.imshow(image[y0:y1, x0:x1])
    if box is not None:
        ax.add_patch(
            Rectangle((box[0] - x0, box[1] - y0), box[2] - box[0], box[3] - box[1], fill=False, ec="red", lw=2)
        )
    ax.set_title(title)
    ax.axis("off")


night_item = item_of["uav0000117_02622_v"]
path = track_path(night_item, 87)
after = path.filter(pl.col("class_label") != pl.col("class_label").shift(1)).row(0, named=True)
before = path.filter(pl.col("frame") < after["frame"]).row(-1, named=True)
area = window(night_item, [before["box"], after["box"]], pad=60)
fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
for ax, row in zip(axes, (before, after), strict=True):
    label = dataset.index2label[row["class_label"]]
    show(ax, night_item, row["frame"], area, row["box"], f"frame {row['frame']}: {label}")
fig.tight_layout()
plt.show()

# %% [markdown]
# The vehicle, partly behind a yellow car, looks the same in both frames. The change may be a borderline call between
# car and van or a labeling slip, and either way one track carries both classes.
#
# A gap can mean that the vehicle was hidden, that it left the frame or that it was missed when the footage was
# labeled. Track 113 of `uav0000182_00000_v` drops out for 12 frames. Below are the last frame before the gap, a frame
# in the middle of it and the first frame after it, with the track's box in red where it has one.
#

# %%
road_path = track_path(road_item, 113)
back = road_path.filter(pl.col("frame") - pl.col("frame").shift(1) > 1).row(0, named=True)
last = road_path.filter(pl.col("frame") < back["frame"]).row(-1, named=True)
middle = (last["frame"] + back["frame"]) // 2
area = window(road_item, [last["box"], back["box"]], pad=80)
shots = (
    (last["frame"], last["box"], "last labeled"),
    (middle, None, "no box"),
    (back["frame"], back["box"], "labeled again"),
)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
for ax, (frame, box, label) in zip(axes, shots, strict=True):
    show(ax, road_item, frame, area, box, f"frame {frame}: {label}")
fig.tight_layout()
plt.show()

# %% [markdown]
# In the middle frame the van is still in plain view, between where it was last labeled and where it is labeled again,
# yet it carries no box. Here the gap looks like a missed label rather than a vehicle that was hidden or left the frame.
# Other gaps need the same look before you decide what they are.
# [Measure frames and tracks in a tracking dataset](./h2_measure_frames_and_tracks.py) measures gaps and the steps
# between detections in more depth.
#
# For the detector, one vehicle would be taught as a car in some frames and a van in others, and a vehicle left
# without a box would be taught as background.
#

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert tracks.height == 246
assert flips.height == 6
assert sorted(flips["pair"].to_list()) == ["car/truck"] + ["car/van"] * 5
assert (tracks["n_gaps"] > 0).sum() == 36
assert (tracks["total_gap_length"] > 50).sum() == 10
per_video = tracks.group_by("item_index").agg(pl.len(), (pl.col("n_gaps") > 0).sum().alias("gapped"))
gapped = {r["item_index"]: (r["gapped"], r["len"]) for r in per_video.iter_rows(named=True)}
assert gapped[item_of["uav0000305_00000_v"]] == (17, 48)
assert gapped[item_of["uav0000117_02622_v"]] == (0, 28)
assert after["frame"] - before["frame"] == 1
assert {dataset.index2label[before["class_label"]], dataset.index2label[after["class_label"]]} == {"car", "van"}
assert back["frame"] - last["frame"] - 1 == 12

# %% [markdown]
# ## Check the size and crowding of the vehicles
#
# A detector sees each frame at the size it is trained at, usually far smaller than 4K. Measure each vehicle box in
# pixels and as a share of the frame's width, so you can work out how wide a vehicle will be at your own input size.
# The boxes are already in the metadata, so this step reads no pixels.
#

# %%
sizes = boxes.group_by("item_index").agg(
    pl.col("box_width").median(),
    pl.col("box_height").median(),
    (pl.col("box_width") / pl.col("width")).median().alias("width_share"),
    ((pl.col("box_width") * pl.col("box_height")) < 32 * 32).mean().alias("under_32"),
)
display(
    videos
    .select("item_index", "video", "width")
    .join(sizes, on="item_index")
    .drop("item_index")
    .rename({"width": "frame_width"})
    .sort("video")
)

# %% [markdown]
# In pixels, the vehicles of `uav0000182_00000_v` are the smallest: the median box is 44 by 31 pixels,
# and about a third are smaller in area than 32 by 32 pixels. Elsewhere the median vehicle is 71 to 108 pixels wide.
# Measured against the frame, the 4K video `uav0000268_05773_v` holds the smallest: a median vehicle of 71 by 38 pixels
# spans under 2% of the frame's width, against about 3% to 5% in the other videos. Once every frame is scaled to the
# same input size, its vehicles end up the smallest. Multiply `width_share` by the width you train at to see how wide
# a vehicle becomes.
#
# Vehicles in traffic also sit close together. The `SPATIAL_COVERED_FRACTION` statistic of
# {func}`~dataeval.core.image_stats` measures how much of each box other boxes cover. It reads the boxes, not the
# picture, so it measures overlap between annotations, not whether one vehicle hides another. Measure it on every
# frame, then attach each frame's values to its video with {meth}`.SequenceFrames.to_source`.
#

# %%
frame_view = SequenceFrames(dataset)
covered = image_stats(frame_view, stats=ImageStats.SPATIAL_COVERED_FRACTION, per_image=False)
metadata.add_factors(covered, source_index=frame_view.to_source(covered["source_index"]))
crowding = (
    metadata
    .rows_at("instance")
    .filter(pl.col("class_label").is_in(vehicle_ids))
    .group_by("item_index")
    .agg((pl.col("covered_fraction").fill_nan(None) > 0.5).mean().alias("over_half"))
)
display(videos.select("item_index", "video").join(crowding, on="item_index").drop("item_index").sort("video"))

# %% [markdown]
# In the busy daytime intersection `uav0000137_00458_v`, about a quarter of the vehicle boxes are more than half covered
# by other boxes. In every other video at most about one in six are.
# [Find crowded, truncated and conflicting annotations](./h2_find_annotation_review_candidates.py) shows how to turn
# these statistics into a list of boxes to review.
#
# For the detector, the vehicles of `uav0000268_05773_v` may shrink below what it resolves once frames are scaled to
# its input size, those of `uav0000182_00000_v` are small even at full size, and where boxes overlap heavily, as in
# the busy intersection, much of a box may show another object.
#

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
s = {r["item_index"]: r for r in sizes.iter_rows(named=True)}
small, big = item_of["uav0000182_00000_v"], item_of["uav0000268_05773_v"]
assert (round(s[small]["box_width"]), round(s[small]["box_height"])) == (44, 31)
assert 0.34 <= s[small]["under_32"] <= 0.36
assert (round(s[big]["box_width"]), round(s[big]["box_height"])) == (71, 38)
assert s[big]["width_share"] < 0.02
assert all(71 <= round(r["box_width"]) <= 108 for k, r in s.items() if k != small)
assert all(0.028 <= r["width_share"] <= 0.05 for k, r in s.items() if k != big)
c = {r["item_index"]: r["over_half"] for r in crowding.iter_rows(named=True)}
busy = item_of["uav0000137_00458_v"]
assert 0.21 <= c[busy] <= 0.27
assert all(v <= 0.16 for k, v in c.items() if k != busy)

# %% [markdown]
# ## Conclusion
#
# You checked a set of drone videos for a vehicle detector before training on it. Each check measured a property of
# the footage, and each finding is a risk to weigh, not a measured effect on a model.
#
# | Finding | How it may hinder training | Measured with |
# | ------- | -------------------------- | ------------- |
# | 7 videos, with night and dusk one video each, and 63% of frames away from intersections | The detector learns night from 28 vehicles on one flight, and 3 of 7 videos show other scenes | `Metadata` |
# | About 96% of frames repeat the frame before them | Far fewer distinct views than frames, and a split by frame leaks near-copies into validation | `Duplicates` |
# | Edges grow lopsided in fast pans, as motion blur along the camera's path would make them | Vehicles in those frames may be smeared | `ego_stats` and `image_stats` |
# | 6 vehicle tracks change class, and one vehicle goes unlabeled for 12 frames in view | The same vehicle carries two classes, and an unlabeled vehicle is taught as background | `track_stats` |
# | Small vehicles in two videos and heavy box overlap in the busy intersection | Vehicles may shrink below what the detector resolves, and a box may show another object | `Metadata` and `image_stats` |
#
# The findings point to the next steps for this footage: collect more night footage and more intersections, split by
# video, review the borderline classes and the gaps, and check vehicle sizes at your input size.
#

# %% [markdown]
# ## Next steps
#
# - [Metadata levels](../concepts/MetadataLevels.md) — Learn how factors sit at the video, frame, track and detection
#   levels and roll up between them.
# - [Data leakage](../concepts/Leakage.md) — Learn why a split has to separate the sources of the data, not only its
#   samples.
# - [Identify duplicate video](./tt_identify_duplicate_video.py) — Find duplicate footage, leakage across a split and
#   redundant segments in a tracking dataset.
# - [Measure frames and tracks in a tracking dataset](./h2_measure_frames_and_tracks.py) — Measure frame timing,
#   repeats, gaps and the steps of each track.
#

# %% [markdown]
# ## On your own
#
# Repeat these checks on your own footage. Start by writing down your use case: what the detector must find, and in
# which scenes and conditions. Then wrap your videos and annotations as a multi-object tracking dataset, as
# [Wrap video files as a multi-object tracking dataset](./h2_wrap_video_files.py) shows, and run each section on it.
# Judge every finding against your use case.
#
