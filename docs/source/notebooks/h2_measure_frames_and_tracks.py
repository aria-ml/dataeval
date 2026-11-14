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
# # Measure frames and tracks in a tracking dataset
#
# Late or repeated frames, ids that jump to another object and tracks that keep dropping out are visible only across
# frames, so a check of single boxes misses them. {func}`~dataeval.core.frame_stats`,
# {func}`~dataeval.core.track_steps` and {func}`~dataeval.core.track_stats` measure them, and a
# {class}`~dataeval.Metadata` holds each result at its own level. See [Metadata levels](../concepts/MetadataLevels.md).
#
# Use them before you train or evaluate on a multi-object tracking (MOT) dataset.
#
# You need DataEval 1.3, where `frame_stats` and `track_steps` are experimental. Frames must report `time_s`, as
# [Wrap video files as a multi-object tracking dataset](./h2_wrap_video_files.py) shows.

# %% tags=["remove_cell"]
# Google Colab Only
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval
except Exception:
    pass

# %%
import warnings

import polars as pl
from IPython.display import display

from dataeval import Metadata
from dataeval.core import frame_stats, track_stats, track_steps
from dataeval.exceptions import ExperimentalWarning

# frame_stats and track_steps are experimental.
warnings.filterwarnings("ignore", category=ExperimentalWarning)
# Reading rows bins the video timestamps, which this guide does not use.
warnings.filterwarnings("ignore", message=".*binned automatically")

# %% [markdown]
# ## Build a tracking dataset with known defects
#
# This guide builds a synthetic stand-in for your own dataset: four sequences of 40 frames at 30 frames per second, each
# with two cars, two people and one planted defect.
#
# | Sequence | Defect | Where it should show |
# | --- | --- | --- |
# | 0 | the recorder stalls for half a second at frame 20 | `elapsed_time_s_std` |
# | 1 | frames 20 to 23 are copies of frame 19 | `repeat_frame_fraction` |
# | 2 | the two people swap ids at frame 20 | `step_displacement_norm` |
# | 3 | one person is annotated on every other frame | `n_gaps` and `gap_exceeds_limit` |

# %% tags=["hide-input"]
from collections.abc import Iterator
from typing import NamedTuple, cast

import numpy as np

from dataeval.protocols import DatasetMetadata, DatumMetadata, MultiobjectTrackingDatum, VideoFrame

HEIGHT, WIDTH, N_FRAMES, FPS = 72, 96, 40, 30
DEFECTS = ["stall", "frozen", "id_switch", "flicker"]


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


class Video:
    """A stand-in for a decoded video: its frames, in order."""

    def __init__(self, frames: list[Frame]) -> None:
        self.frames = frames

    def __iter__(self) -> Iterator[VideoFrame]:
        return iter(self.frames)


def make_sequence(index: int, defect: str) -> tuple[Video, VideoTargets]:
    """Two fast cars (ids 0 and 1) and two slow people (ids 2 and 3), with one planted defect."""
    rng = np.random.default_rng(100 + index)
    tracks = {}  # id -> (class, center at frame 0, center step per frame)
    for track in (0, 1):
        start = rng.uniform([8, 8], [WIDTH - 20, HEIGHT - 20])
        tracks[track] = (0, start, rng.uniform(0.8, 1.2, 2) * rng.choice([-1, 1], 2))
    for track, (low, high) in ((2, (8, 40)), (3, (56, 76))):
        start = np.array([rng.uniform(low, high), rng.uniform(8, HEIGHT - 20)])
        tracks[track] = (1, start, rng.uniform(0.1, 0.3, 2) * rng.choice([-1, 1], 2))
    frames, targets = [], []
    for frame in range(N_FRAMES):
        boxes, ids, labels = [], [], []
        for track, (label, start, step) in tracks.items():
            if defect == "flicker" and track == 3 and frame % 2:
                continue
            swapped = defect == "id_switch" and frame >= 20 and track in (2, 3)
            cx, cy = start + step * frame
            boxes.append([cx - 6, cy - 6, cx + 6, cy + 6])
            ids.append(5 - track if swapped else track)
            labels.append(label)
        targets.append(
            FrameTargets(
                boxes=np.array(boxes, dtype=np.float32),
                labels=np.array(labels, dtype=np.int64),
                scores=np.ones(len(ids), dtype=np.float32),
                track_ids=np.array(ids, dtype=np.int64),
            )
        )
        # Each frame is a flat image of its own value, so only a copied frame repeats the one before it.
        pixels = np.full((3, HEIGHT, WIDTH), (7 * frame + 40 * index) % 200 + 20, dtype=np.uint8)
        if defect == "frozen" and 20 <= frame < 24:
            pixels = frames[-1].pixels
        stall = 0.5 if defect == "stall" and frame >= 20 else 0.0
        frames.append(Frame(pixels, frame / FPS + stall, frame, frame))
    return Video(frames), VideoTargets(targets)


class TrackingDataset:
    """A minimal MAITE multi-object tracking dataset."""

    def __init__(self, defects: list[str]) -> None:
        self.metadata = DatasetMetadata({"id": "planted-defects", "index2label": {0: "car", 1: "person"}})
        self._data: list[MultiobjectTrackingDatum] = []
        for index, defect in enumerate(defects):
            video, targets = make_sequence(index, defect)
            datum_metadata = cast(DatumMetadata, {"id": f"sequence_{index}", "height": HEIGHT, "width": WIDTH})
            self._data.append((video, targets, datum_metadata))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> MultiobjectTrackingDatum:
        return self._data[index]


dataset = TrackingDataset(DEFECTS)

# %% [markdown]
# ## Measure the frames, steps and tracks
#
# Build the `Metadata` once and attach each result to it. {meth}`.Metadata.add_factors` also rolls the frame values up
# to the sequence level; pass `aggregate=False` to keep only the per-frame values.

# %%
metadata = Metadata(dataset)
# Pass stats=FrameStats.TIMING to read timestamps only and decode no frames.
metadata.add_factors(frame_stats(dataset))

steps = track_steps(dataset)
metadata.add_factors(steps)
# add_factors ignores a result's details, so attach step_status by hand.
metadata.add_factors({"step_status": steps.get("details", {})["step_status"]}, source_index=steps["source_index"])

metadata.add_factors(track_stats(dataset), level="track", key="track_id")
print(metadata.level_counts)

# %% [markdown]
# The counts are sequences, frames (`unit`), tracks and detections (`instance`).

# %% [markdown]
# ## Find the stalled and frozen sequences
#
# Read the two roll-ups on the sequence rows: a stall widens the spread of the time between frames, and copied frames
# raise the fraction of frames that repeat the one before.

# %%
sequences = metadata.rows_at("sequence")
display(
    sequences.select("item_id", pl.col("elapsed_time_s_std", "repeat_frame_fraction").round(3)),
)

# %% [markdown]
# Sequence 0 is the only one whose frames arrive unevenly, with a spread of 0.08 s, and sequence 1 the only one with
# repeats, a tenth of its frames.

# %% [markdown]
# ## Read gaps and steps per class
#
# Read the share of each `step_status` per class. A detection's status says why its step from the previous observation
# was or was not measured, and a class that drops out of its tracks has a share of `gap_exceeds_limit`.

# %%
detections = metadata.rows_at("instance").with_columns(
    pl.col("class_label").replace_strict(metadata.index2label).alias("class")
)
statuses = ["ok", "first_observation", "gap_exceeds_limit"]
rates = detections.group_by("class", maintain_order=True).agg([
    (pl.col("step_status") == status).mean().round(3).alias(status) for status in statuses
])
display(rates)

# %% [markdown]
# People have 6% `gap_exceeds_limit` and cars have none. The shares pool detections, and a track's detections are not
# independent observations, so count tracks when a sample size matters.

# %% [markdown]
# Name the track with `track_stats`. Its `n_gaps` counts the same events as `gap_exceeds_limit` while `track_steps`
# keeps its default `max_frame_delta=1`.

# %%
flickering = metadata.rows_at("track").filter(pl.col("n_gaps") > 0)
display(flickering.select("item_id", "track_id", "n_gaps", "gap_fraction"))

# %% [markdown]
# Sort the measured steps to find an id that jumped. Keep only `ok` rows, because the rows without a measured step
# are NaN and sort first.

# %%
largest = (
    detections
    .filter(pl.col("step_status") == "ok")
    .sort("step_displacement_norm", descending=True)
    .select("item_id", "track_id", "class", "time_s", "step_displacement_norm")
    .head(3)
)
display(largest)

# %% [markdown]
# The two people of sequence 2 each jump 4.5 box diagonals at 0.67 s (frame 20), and no other step exceeds 0.1. A step
# is not conditioned on shot boundaries or camera motion, so a camera pan or a cut between shots also shows as a large
# step. See {func}`~dataeval.core.track_steps`.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
import math

assert dict(metadata.level_counts) == {"sequence": 4, "unit": 160, "track": 16, "instance": 620}

# The roll-ups exist by default and none is stored with aggregate=False.
rolled = {"elapsed_time_s_median", "elapsed_time_s_std", "repeat_frame_fraction"}
assert rolled <= set(sequences.columns)
plain = Metadata(dataset)
plain.add_factors(frame_stats(dataset), aggregate=False)
assert not rolled & set(plain.rows_at("sequence").columns)
assert {"elapsed_time_s", "repeat_frame"} <= set(plain.rows_at("unit").columns)

# Only the stalled sequence has a timing spread, and only the frozen sequence has repeats.
by_item = sequences.sort("item_index")
spread = by_item["elapsed_time_s_std"].to_list()
assert abs(spread[0] - 0.08) < 0.005
assert max(spread[1:]) < 1e-9
fraction = by_item["repeat_frame_fraction"].to_list()
assert abs(fraction[1] - 4 / 39) < 1e-9
assert fraction[0] == fraction[2] == fraction[3] == 0

# People drop out, cars do not.
by_class = {row["class"]: row for row in rates.iter_rows(named=True)}
assert by_class["car"]["gap_exceeds_limit"] == 0
assert abs(by_class["person"]["gap_exceeds_limit"] - 0.063) < 0.001

# One track has gaps: the flickering person of sequence 3.
assert flickering.height == 1
assert flickering["item_index"].to_list() == [3]
assert flickering["track_id"].to_list() == [3]
assert flickering["n_gaps"].to_list() == [19]
assert detections.filter(pl.col("step_status") == "gap_exceeds_limit").height == 19

# Unfiltered, the rows without a measured step sort first.
unfiltered = detections.sort("step_displacement_norm", descending=True)["step_displacement_norm"]
assert math.isnan(unfiltered[0])

# The id switch is the two largest measured steps, and every other step is small.
steps_sorted = largest["step_displacement_norm"].to_list()
assert largest["item_id"].to_list()[:2] == ["sequence_2", "sequence_2"]
assert sorted(largest["track_id"].to_list()[:2]) == [2, 3]
assert all(abs(t - 20 / 30) < 1e-9 for t in largest["time_s"].to_list()[:2])
assert all(abs(s - 4.5) < 0.05 for s in steps_sorted[:2])
assert steps_sorted[2] < 0.1

# %% [markdown]
# ## Next steps
#
# - [Metadata levels](../concepts/MetadataLevels.md) — Learn what each level holds and how values move between them.
# - [Analyze a dataset across its levels](./tt_analyze_across_levels.py) — Move between the levels of a `Metadata` and
#   aggregate a factor from one level to the next.
# - [Find outliers in metadata factors](./h2_find_factor_outliers.py) — Threshold these factors, each against the rows
#   at its own level.
