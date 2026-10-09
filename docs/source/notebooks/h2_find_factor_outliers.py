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
# # How to find outliers in metadata factors

# %% [markdown]
# ## Problem statement
#
# Much of what describes a dataset never touches a pixel: sensor telemetry recorded with each frame, capture
# conditions, and statistics you derived yourself, such as the per-track measurements
# {func}`~dataeval.core.track_stats` produces. Each is a column of numbers with a distribution, and a value far outside
# it is worth a look - a sensor that glitched for one frame, or a track whose annotation broke down.
#
# {class}`~dataeval.quality.Outliers` thresholds these columns directly when you pass it a {class}`~dataeval.Metadata`.
# Each factor is judged against the rows at the level it was measured at: a track against other tracks, a frame against
# other frames. A frame and a track are measured at different levels, so comparing them would answer the wrong question.

# %% [markdown]
# ### When to use
#
# Use this guide when you have attached measurements to a `Metadata` - your own, or a statistic family DataEval
# computed - and want each one checked the way `Outliers` checks image statistics. For a multi-object tracking (MOT)
# dataset, the track level turns this into a re-annotation worklist: one row per suspicious track.

# %% [markdown]
# ### What you will need
#
# 1. A python environment with the following packages installed:
#    - dataeval
#    - maite-datasets
# 1. A dataset, or a `Metadata` with the factors you want to check

# %% [markdown]
# ## Getting started
#
# Let's import the required libraries.

# %% tags=["remove_cell"]
# Google Colab Only
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval maite-datasets
except Exception:
    pass

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
from dataeval.core import track_stats
from dataeval.protocols import DatasetMetadata, DatumMetadata, MultiobjectTrackingDatum, VideoFrame
from dataeval.quality import Outliers
from dataeval.types import SourceIndex

pl.Config.set_tbl_width_chars(160)

# %% [markdown]
# ## Building a tracking dataset with known problems
#
# No ready-made MOT dataset has labeled annotation defects, so this builds one: twelve 40-frame sequences, each holding
# four vehicles that drift across the frame with a little observation noise. Three sequences carry a planted defect on
# their track 0:
#
# | Sequence | Defect | Where it should show |
# | --- | --- | --- |
# | 2 | the track is annotated on every other frame only - flickering | `n_gaps` and `gap_fraction` |
# | 5 | one track id is used for two different objects, seen only at the start and the end | `gap_fraction` |
# | 7 | the boxes jump around the object from frame to frame | `speed_variance` |
#
# The video frames are blank. Nothing here reads pixels: the track statistics come from the annotation.

# %%
HEIGHT, WIDTH, N_FRAMES, N_SEQUENCES, N_TRACKS = 72, 96, 40, 12, 4
rng = np.random.default_rng(0)


def path(n_frames: int, noise: float) -> np.ndarray:
    """Box centers for one object drifting at constant velocity, observed with `noise` pixels of error."""
    start = rng.uniform([10, 10], [WIDTH - 30, HEIGHT - 30])
    velocity = rng.uniform(-0.6, 0.6, 2)
    return start + velocity * np.arange(n_frames)[:, None] + rng.normal(0, noise, (n_frames, 2))


def tracks_for(defect: str | None) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Each track as (frames it appears in, box center in each of them)."""
    tracks = {track: (np.arange(N_FRAMES), path(N_FRAMES, noise=0.6)) for track in range(N_TRACKS)}
    if defect == "flicker":
        frames = np.arange(0, N_FRAMES, 2)
        tracks[0] = (frames, tracks[0][1][frames])
    elif defect == "reused_id":
        frames = np.r_[0:3, N_FRAMES - 3 : N_FRAMES]
        tracks[0] = (frames, np.r_[path(3, 0.6), path(3, 0.6) + 40])
    elif defect == "jitter":
        tracks[0] = (np.arange(N_FRAMES), path(N_FRAMES, noise=4.0))
    return tracks


def frame_targets(defect: str | None) -> list[SingleFrameObjectTrackingTargetTuple]:
    """One target per frame, holding a 12-pixel box for every track present in it."""
    tracks = tracks_for(defect)
    targets = []
    for frame in range(N_FRAMES):
        boxes, ids = [], []
        for track, (frames, centers) in tracks.items():
            at = np.flatnonzero(frames == frame)
            if at.size:
                cx, cy = centers[at[0]]
                boxes.append([cx - 6, cy - 6, cx + 6, cy + 6])
                ids.append(track)
        targets.append(
            SingleFrameObjectTrackingTargetTuple(
                boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4),
                labels=np.zeros(len(ids), dtype=np.int64),
                scores=np.ones(len(ids), dtype=np.float32),
                track_ids=np.array(ids, dtype=np.int64),
            )
        )
    return targets


class BlankVideo:
    """A stand-in for a decoded video: the right number of frames, all zeros."""

    def __iter__(self) -> Iterator[VideoFrame]:
        blank = np.zeros((3, HEIGHT, WIDTH), dtype=np.uint8)
        for index in range(N_FRAMES):
            yield VideoFrameTuple(pixels=blank, time_s=index / 30, pts=index, frame_index=index)


class TrackingDataset:
    """A minimal MAITE multi-object tracking dataset."""

    def __init__(self, defects: dict[int, str]) -> None:
        self.metadata = DatasetMetadata({"id": "planted-defects", "index2label": {0: "vehicle"}})
        self._data: list[MultiobjectTrackingDatum] = [
            (
                BlankVideo(),
                MultiobjectTrackingTargetTuple(frame_tracks=frame_targets(defects.get(index))),
                cast(DatumMetadata, {"id": f"sequence_{index}", "height": HEIGHT, "width": WIDTH}),
            )
            for index in range(N_SEQUENCES)
        ]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> MultiobjectTrackingDatum:
        return self._data[index]


dataset = TrackingDataset({2: "flicker", 5: "reused_id", 7: "jitter"})

# %% [markdown]
# ## Attaching the factors
#
# Build the `Metadata`, then attach the factors to check, each at its own level. `track_stats` measures every track of
# every sequence and lands at the track level, keyed by `track_id`.
#
# Add one factor of your own too: a per-frame exposure time from the sensor, at the frame (`unit`) level, with a
# single glitched reading in frame 20 of sequence 2.

# %%
metadata = Metadata(dataset)
metadata.add_factors(track_stats(dataset), level="track", key="track_id")

exposure_ms = rng.normal(8.0, 0.4, metadata.level_counts["unit"])
exposure_ms[2 * N_FRAMES + 20] = 30.0
metadata.add_factors({"exposure_ms": exposure_ms}, level="unit")

print(metadata.level_counts)

# %% [markdown]
# ## Which factors can be thresholded
#
# A threshold needs a quantity: a location and a spread. `Outliers` decides which factors are quantities from each
# column's raw type, not from how `Metadata` chose to bin it:
#
# - **Ordered** - integers, floats, dates and durations - are thresholded. That includes integer counts such as
#   `n_gaps`, which the binning heuristic calls discrete.
# - **Categorical** - strings, booleans, and integers with a declared vocabulary - are not. `track_stats` declares its
#   `labels` column a vocabulary, so a class id is never given a mean. `entry_at_edge` and `exit_at_edge` are booleans.
#
# Name a categorical factor and `Outliers` raises rather than returning a result:

# %%
try:
    Outliers().evaluate(metadata, factors=["labels"])
except ValueError as error:
    print(error)

# %% [markdown]
# Only raw values are read - never bin codes - so re-binning the metadata cannot change a result.

# %% [markdown]
# ## Thresholding every factor
#
# With no `factors=`, every ordered factor is checked, each at its own level. The `level` column says which population
# a row was judged in.

# %%
result = Outliers().evaluate(metadata)
display(result.aggregate_by_metric())

# %% [markdown]
# Several track factors describe the same thing - `n_appearances`, `track_length` and `total_gap_length` all move when
# a track loses frames - so one defect raises several rows. Name the factors to check to get one row per question.

# %%
signals = ["n_gaps", "gap_fraction", "speed_variance", "exposure_ms"]
focused = Outliers().evaluate(metadata, factors=signals)
display(focused.data().select("item_index", "target_index", "level", "metric_name", "metric_value", "percentile"))

# %% [markdown]
# Each planted defect is flagged, and nothing else is:
#
# - sequence 2, track 0 - `n_gaps` and `gap_fraction`: the flickering track, with 19 gaps covering half its span
# - sequence 5, track 0 - `gap_fraction` only: one id spanning two objects, a single gap covering most of its span
# - sequence 7, track 0 - `speed_variance`: boxes that jump from frame to frame
# - sequence 2, frame 20 - `exposure_ms`: the glitched sensor reading, judged against frames, not tracks
#
# The two gap signals distinguish the defects: many short gaps is a track that keeps dropping out, one long gap is an
# id that was reused. For a track row `target_index` holds the `track_id`, and for a frame row the frame's
# `unit_index`.

# %% [markdown]
# ## Reading a finding back to its track
#
# `outliers` maps each flagged row's address to the metrics that flagged it. A track is addressed as
# `SourceIndex(item, track_id, "track")`, which is enough to find it again in the metadata:

# %%
print(focused.outliers)

flickering = metadata.rows_at("track").filter((pl.col("item_index") == 2) & (pl.col("track_id") == 0))
display(flickering.select("item_index", "track_id", "n_appearances", "track_duration", "n_gaps", "gap_fraction"))

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert focused.outliers == {
    SourceIndex(2, 0, "track"): ["gap_fraction", "n_gaps"],
    SourceIndex(5, 0, "track"): ["gap_fraction"],
    SourceIndex(7, 0, "track"): ["speed_variance"],
    SourceIndex(2, 20, "unit"): ["exposure_ms"],
}
assert "labels" not in set(result.data()["metric_name"].cast(pl.Utf8))

# %% [markdown]
# ## Next steps
#
# - [Data Integrity](../concepts/DataIntegrity.md) — What makes a factor thresholdable, and why bins are never read.
# - [Metadata Levels](../concepts/MetadataLevels.md) — Levels, keys, and the built-in track statistics.
# - [How to find duplicate video](./h2_deduplicate_video.py) — Duplicates and redundancy in the same kind of dataset.
# - [How to trace findings to their source](./h2_trace_findings_to_source.py) — Retrieve the frame, track or detection a
#   `SourceIndex` names.
