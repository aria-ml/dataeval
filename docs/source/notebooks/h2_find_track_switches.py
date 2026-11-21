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
# # Find tracks whose ids may have switched
#
# An identity switch is a track id that moves to another object. The experimental {func}`.track_switches` scores each
# meeting of two tracks of one class by whether their crop embeddings fit better crossed than straight. See
# [Label noise and mislabeling](../concepts/DataIntegrity.md#label-noise-and-mislabeling) and the function's row in
# [Validation evidence by evaluator](../reference/ValidationEvidence.md).
#
# Use it to find where to check a tracking dataset's ids. A flag marks a place to look, not an error. An empty table
# does not mean the ids are right: a switch with no second track nearby is not scored, and {func}`.track_steps` shows
# it as a large step.
#
# You need DataEval 1.3, `dataeval-plots>=0.0.20`, `maite-datasets[hf]>=0.0.25` and `torchvision`.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval "dataeval-plots>=0.0.20" "maite-datasets[hf]>=0.0.25"
except Exception:
    pass

# %%
import warnings

import dataeval_plots as dep
import numpy as np
import polars as pl
import torch
from IPython.display import display
from maite_datasets.multiobject_tracking import VisDroneMOT
from torchvision.models import ResNet18_Weights, resnet18

from dataeval import Embeddings
from dataeval.config import set_device
from dataeval.core import track_switches
from dataeval.data import DetectionCrops, Indices, SequenceFrames, View
from dataeval.exceptions import ExperimentalWarning
from dataeval.extractors import TorchExtractor

set_device("cuda" if torch.cuda.is_available() else "cpu")
# track_switches is experimental.
warnings.filterwarnings("ignore", category=ExperimentalWarning)

# %% [markdown]
# ## Load one video as frames
#
# Load the validation videos of VisDrone2019-MOT, the drone tracking benchmark of
# [Zhu et al. (2021)](https://arxiv.org/abs/2001.06303), through the
# [Voxel51 copy](https://huggingface.co/datasets/Voxel51/visdrone-mot), shared under the
# [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) license, which downloads about 1.7 GB on the
# first run. Keep one video as frames with {class}`.SequenceFrames`. To check every video, pass `dataset` without the
# `View`.

# %%
dataset = VisDroneMOT(root="./data", image_set="val", download=True)
frames = SequenceFrames(View(dataset, Indices([6])))
print(f"{frames[0][2]['source_id']}: {len(frames)} frames")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert frames[0][2]["source_id"] == "uav0000339_00001_v"
assert len(frames) == 275

# %% [markdown]
# ## Crop and embed each detection
#
# Crop each box and embed the crops with a pretrained ResNet18, an ImageNet classifier, as
# [Embed object detection crops and visualize class clusters](./h2_embed_detection_crops.py) shows. `min_size=16`
# drops boxes whose shorter side is under 16 pixels, and a dropped box cannot meet another.

# %%
crops = DetectionCrops(frames, min_size=16, square="pad")
weights = ResNet18_Weights.DEFAULT
extractor = TorchExtractor(
    resnet18(weights=weights, progress=False), transforms=weights.transforms(), layer_name="avgpool"
)
embeddings = Embeddings(crops, extractor=extractor, batch_size=64)
print(f"{len(crops)} crops, {crops.n_dropped} dropped")

# %% [markdown]
# ## Score the encounters
#
# Pass each crop's video and frame from `frames.frame_map`, and its box and class from the crops. Only tracks of one
# class whose boxes meet are paired, and the rows come highest score first.

# %%
video, frame = frames.frame_map[crops.item_indices].T
result = track_switches(embeddings, crops.track_ids, frame, video, boxes=crops.boxes, labels=crops.labels)
encounters = pl.DataFrame(result)
print(f"{encounters.height} encounters, {encounters['switch_score'].is_not_nan().sum()} scored")
display(encounters.filter(pl.col("flagged")).select("label", "track_a", "track_b", "frame", "switch_score"))

# %% [markdown]
# A score above 0 means that, summed over the two tracks, the crops after the meeting lie closer to the other track's
# crops before it than to their own. An encounter is not scored when no meeting has 10 crops of each track on both
# sides, the default `window`.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert (len(crops), len(np.unique(crops.track_ids))) == (11286, 71)
assert (crops.track_ids >= 0).all()  # so np.unique counts tracks only
# Two encounters are too near a track's end to score.
assert (encounters.height, int(encounters["switch_score"].is_not_nan().sum())) == (64, 62)
# The third score is about -0.025, far from 0 against the 1e-4 difference between a CPU and a GPU run.
assert int(encounters["flagged"].sum()) == 2
assert encounters["switch_score"][2] < -0.01
top = encounters.row(0, named=True)
assert (top["track_a"], top["track_b"], top["flagged"]) == (27, 28, True)

# %% [markdown]
# ## Look at the top row
#
# Draw each track's box 5 frames before the flagged frame, at it and 5 frames after, with `track_a` on the top row and
# `track_b` below, cut to the area around the pair. If a track has no box at one of those frames, choose others.

# %%
top = encounters.row(0, named=True)
pair = (top["track_a"], top["track_b"])
shown = [top["frame"] - 5, top["frame"], top["frame"] + 5]
here = video == top["item_index"]
row = {(t, f): np.flatnonzero(here & (crops.track_ids == t) & (frame == f))[0] for t in pair for f in shown}
corners = crops.boxes[list(row.values())]
height, width = np.asarray(frames[int(crops.item_indices[top["row_a"]])][0]).shape[-2:]
area = (
    max(int(corners[:, 0].min()) - 30, 0),
    max(int(corners[:, 1].min()) - 30, 0),
    min(int(corners[:, 2].max()) + 30, width),
    min(int(corners[:, 3].max()) + 30, height),
)
dep.plot(
    frames,
    indices=crops.item_indices[list(row.values())].tolist(),
    images_per_row=3,
    titles=[f"track {t}, frame {f}" for t, f in row],
    boxes=[[box] for box in corners.tolist()],
    region=area,
    box_color="red",
    figsize=(9, 3.6),
)

# %% [markdown]
# In frame 102 track 27 is on the person in the dark shirt and track 28 on the person in red behind them, and by frame
# 112 each box is on the other person. Look at the other flagged row, `encounters.row(1, named=True)`, the same way,
# since a flag can also be a box that drifted off its object, an occluder that both tracks pass behind, or nothing
# wrong.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert shown == [102, 107, 112]  # the frames the text names
centers = {key: (crops.boxes[r][0] + crops.boxes[r][2]) / 2 for key, r in row.items()}
# The two boxes cross.
assert centers[27, 102] < centers[28, 102]
assert centers[27, 112] > centers[28, 112]

# %% [markdown]
# ## Next steps
#
# - [Label noise and mislabeling](../concepts/DataIntegrity.md#label-noise-and-mislabeling) — What an identity switch
#   is and what a flag can mean.
# - [Measure frames and tracks in a tracking dataset](./h2_measure_frames_and_tracks.py) — Measure the step of each
#   detection, which shows a jump to a distant object.
# - [Trace a finding back to its source](./h2_trace_findings_to_source.py) — Retrieve the frame or detection that a
#   finding names.
