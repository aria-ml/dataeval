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
# # Find misspelled class names and empty images in a detection dataset
#
# This tutorial shows you how to find the defects that a dataset can carry from its source to your code: class names
# that are not on the dataset's class list, a box whose class was never named, and images with nothing in them. You
# find each one with DataEval, decide what it is by looking at it, and fix what can be fixed.
#
# Estimated time to complete: 15 minutes
#
# Relevant ML stages: [Data Engineering](../getting-started/roles/ML_Lifecycle.md#data-engineering)
#
# Relevant personas: [Data Engineer](../getting-started/roles/data_engineer.md)
#
# ## What you will do
#
# - Load drone video frames with four planted defects
# - Find the class names that are not on the class list
# - Find the images with no content
# - Look at the misnamed boxes and decide what they are
# - Fix the labels and leave out the empty images
#
# ## What you will learn
#
# - How to find class names outside a dataset's class list, and how to tell a typo from a box with no name
# - How to find images with no content with `Outliers`, and how to tell them from ordinary frames it also flags
# - How to fix labels with `Relabel` without losing a box silently, and how to leave out what cannot be fixed
#
# ## What you will need
#
# - A Python environment with `dataeval`, `maite-datasets[hf]>=0.0.25` and `dataeval-plots>=0.0.19` installed
# - About 2 GB of free disk space for the 1.7 GB dataset
# - Time for the first download: about 26 minutes without a Hugging Face token, less with one
# - Less than 1 minute to run the notebook on a CPU, after the download
#

# %% [markdown]
# ## Introduction
#
# A dataset makes claims about itself. Its class list says which names a box can carry, and each image file claims to
# hold a picture. A defect is a box or an image that breaks one of those claims: a box whose name is not on the list,
# or an image with nothing in it.
#
# A dataset loader can hide such defects. It might correct a misspelled name for you, guess a class for a box with no
# name, or put a black image where a file could not be read. Each of those fixes hides the defect from you, and a guess
# can be wrong. A loader that passes the data through as it was published leaves the defects for you to find. This
# tutorial shows you how to find them with DataEval, how to decide what each one is, and how to fix what can be fixed,
# with every change on the record.
#
# The defects here are modeled on ones in a published drone dataset, DroneVehicle. Its training labels hold the
# misspelled class name `feright` and a box named `*`, and some of its image files are empty or cut short.
#

# %% [markdown]
# ## Setup
#

# %% tags=["remove_cell"]
# Google Colab Only
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval "maite-datasets[hf]>=0.0.25" "dataeval-plots>=0.0.19"
except Exception:
    pass

# %%
import difflib

import dataeval_plots as dep
import numpy as np
import polars as pl
from IPython.display import display
from maite_datasets.multiobject_tracking import VisDroneMOT

from dataeval import Metadata, Ontology
from dataeval.config import set_max_processes
from dataeval.core import label_alignment
from dataeval.data import DetectionCrops, EvenlySpaced, Indices, Relabel, SequenceFrames, View
from dataeval.quality import Outliers

set_max_processes(4)
_ = pl.Config.set_tbl_width_chars(160)
_ = pl.Config.set_tbl_rows(20)

# %% [markdown]
# ## Load the frames
#
# You will work with VisDrone2019-MOT, the drone tracking benchmark of
# [Zhu et al. (2021)](https://arxiv.org/abs/2001.06303), through the
# [Voxel51 copy of its validation videos](https://huggingface.co/datasets/Voxel51/visdrone-mot). It is shared under the
# [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) license, which allows non-commercial use
# with attribution and asks that adaptations carry the same license. The first run downloads about 1.7 GB.
# [Check drone footage before you train a detector](./tt_check_drone_footage.py) works with the same videos.
#
# The videos are a tracking dataset. {class}`.SequenceFrames` presents them as a detection dataset of frames, and
# {class}`.EvenlySpaced` keeps 3 frames spread across each of the 7 videos.
#

# %%
dataset = VisDroneMOT(root="./data", image_set="val", download=True)
frames = SequenceFrames(dataset, EvenlySpaced(3))
CLASSES = [dataset.index2label[index] for index in sorted(dataset.index2label)]
print(f"{len(frames)} frames, classes: {CLASSES}")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(dataset) == 7
assert len(frames) == 21
assert len(CLASSES) == 10

# %% [markdown]
# The next cell plants four defects in the frames:
#
# - one truck box is renamed `truvk`, a typo;
# - one bus box is renamed `*`, a box whose class was never written down;
# - one frame is replaced by a black image of the same size, which is what some loaders give when they catch a read
#   error and put a placeholder in its place;
# - one frame is replaced by an image with no pixels, which is what some loaders give when they pass an empty file
#   through.
#
# The cell is hidden, because planting defects is not part of the work. From here on, `planted` stands for a dataset
# as you might receive it, and you look for the defects as if you did not know where they are.
#

# %% tags=["hide-input"]
TRUCK, BUS = CLASSES.index("truck"), CLASSES.index("bus")
# (frame, box) of the planted names, and the two frames replaced by images with no content.
TYPO, NAMELESS = (5, 12), (15, 11)
BLACK, EMPTY = 0, 6


class PlantedDefects:
    """Pass the frames through, with two boxes renamed and two images replaced."""

    def __init__(self, frames, *, typo, nameless, black, empty):
        self._frames, self._typo, self._nameless, self._black, self._empty = frames, typo, nameless, black, empty
        names = frames.metadata["index2label"]
        self.metadata = {"id": "visdrone-planted", "index2label": {**names, 10: "truvk", 11: "*"}}

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, index):
        image, target, datum = self._frames[index]
        labels = np.array(target.labels, copy=True)
        for (frame, box), label in ((self._typo, 10), (self._nameless, 11)):
            if index == frame:
                labels[box] = label
        if index == self._black:
            image = np.zeros_like(np.asarray(image))
        if index == self._empty:
            image = np.zeros((3, 0, 0), dtype=np.uint8)
        return image, target._replace(labels=labels), datum


planted = PlantedDefects(frames, typo=TYPO, nameless=NAMELESS, black=BLACK, empty=EMPTY)

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert np.asarray(frames[TYPO[0]][1].labels)[TYPO[1]] == TRUCK
assert np.asarray(frames[NAMELESS[0]][1].labels)[NAMELESS[1]] == BUS
assert {BLACK, EMPTY}.isdisjoint({TYPO[0], NAMELESS[0]})

# %% [markdown]
# ## Find the class names outside the class list
#
# Start with the names. Count the boxes of each class name the data uses, with {class}`.Metadata`:
#


# %%
def box_counts(data, column="boxes"):
    """Count the boxes of each class name in a detection dataset."""
    names = data.metadata["index2label"]
    rows = Metadata(data).rows_at("instance")
    counts = rows.group_by("class_label").len(name=column).sort("class_label")
    return counts.select(pl.col("class_label").replace_strict(names, return_dtype=pl.String).alias("class"), column)


counts = box_counts(planted)
display(counts)

# %% [markdown]
# Two names are not VisDrone classes: `truvk` and `*`, with one box each. With ten classes you can spot them by eye.
# With hundreds of names, or names spread over many files, let {func}`.label_alignment` compare the names in the data
# with the class list. Write the class list as an {class}`.Ontology` of the ten classes:
#

# %%
VISDRONE = Ontology.from_hierarchy(CLASSES)
alignment = label_alignment(counts["class"].to_list(), VISDRONE)
print("Names not on the class list:", list(alignment["unaligned_source"]))

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert sorted(alignment["unaligned_source"]) == ["*", "truvk"]
assert counts.filter(pl.col("class").is_in(["*", "truvk"]))["boxes"].to_list() == [1, 1]

# %% [markdown]
# `label_alignment` matched the ten names spelled as on the class list and reports the two that are not. To see what
# each one might have meant, rank the class names by how close their spelling is:
#

# %%
print({name: difflib.get_close_matches(name, CLASSES, n=3, cutoff=0) for name in alignment["unaligned_source"]})

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert difflib.get_close_matches("truvk", CLASSES, n=1, cutoff=0) == ["truck"]
assert all(difflib.SequenceMatcher(None, "*", name).ratio() == 0 for name in CLASSES)

# %% [markdown]
# `truvk` is one letter away from `truck`, so it reads as a typo. `*` shares no character with any class name, so its
# ranking carries no information: it is a box whose class was never written down. A spelling ranking can only propose
# a class. The box itself decides, as the section after next shows.
#
# The ranking is kept apart from `label_alignment` on purpose. Given a matcher, `label_alignment` accepts what the
# matcher proposes, so the matcher decides rather than you. [Align two label spaces](./h2_align_label_spaces.py) shows
# how to pass a matcher when you trust it to decide.
#

# %% [markdown]
# ## Find the empty images
#
# Next, the pictures. {class}`.Outliers` measures statistics of every image, such as its brightness, contrast and
# size, and flags the images whose values sit far from the rest. Each row of its output is one flagged statistic of one
# frame: its value, and the bound past which `Outliers` calls a value unusual. [Clean a dataset](./tt_clean_dataset.py)
# covers it in depth.
#

# %%
outliers = Outliers().evaluate(planted)
flagged = outliers.data().select("item_index", "metric_name", "metric_value", "bound").sort("item_index", "metric_name")
display(flagged)

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
flagged_items = sorted(set(flagged["item_index"].to_list()))
ordinary = [item for item in flagged_items if item not in (BLACK, EMPTY)]
assert {BLACK, EMPTY} <= set(flagged_items)
assert len(flagged_items) == 5
assert set(flagged.filter(pl.col("item_index") == BLACK)["metric_name"].to_list()) == {
    "contrast",
    "depth",
    "entropy",
    "zeros",
}
assert flagged.filter(pl.col("item_index") == EMPTY)["metric_name"].to_list() == ["missing"]
black_values = dict(zip(*flagged.filter(pl.col("item_index") == BLACK)[["metric_name", "metric_value"]], strict=True))
assert black_values["zeros"] == 1
assert black_values["depth"] == 1
assert black_values["contrast"] == 0
assert black_values["entropy"] == 0
assert flagged.filter(pl.col("item_index") == EMPTY)["metric_value"].to_list() == [1]
assert set(flagged.filter(pl.col("item_index").is_in(ordinary))["metric_name"].to_list()) == {"saturated_fraction"}
assert all(0.25 <= value <= 0.40 for value in flagged.filter(pl.col("item_index").is_in(ordinary))["metric_value"])

# %% [markdown]
# Five frames are flagged.
#
# - Frame 0 is flagged on four statistics. All of its values are zero (`zeros` is 1), so it has no `contrast`, no
#   spread of values (`entropy`), and its values need only one bit (`depth`). It is the black frame.
# - Frame 6 is flagged on `missing`, the share of its values that are missing, which is 1. The frame has no pixels at
#   all, as its shape shows below.
# - Frames 12, 13 and 14 are flagged on `saturated_fraction`, the share of values at the top of their range: about a
#   third of each frame.
#

# %%
print("Shape of frame 6:", planted[6][0].shape)
print("Videos of frames 12, 13 and 14:", {planted[frame][2]["source_id"] for frame in (12, 13, 14)})

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert planted[EMPTY][0].shape == (3, 0, 0)
assert EMPTY == 6
assert BLACK == 0
assert ordinary == [12, 13, 14]
assert len({planted[item][2]["source_id"] for item in ordinary}) == 1

# %% [markdown]
# A frame with no pixels cannot be drawn. Draw the black frame next to one of the frames flagged on
# `saturated_fraction`:
#

# %%
dep.plot([planted[0][0], planted[12][0]], figsize=(10, 3), images_per_row=2, titles=["Frame 0", "Frame 12"])

# %% [markdown]
# The three frames flagged on `saturated_fraction` come from one video. In it the camera looks toward the horizon, and
# a white, overcast sky fills about a third of the picture. That is a property of the footage, not a defect, so keep
# those frames. `Outliers` flags what is unusual, and you decide whether unusual is wrong. The black frame and the frame
# with no pixels are wrong: neither holds a picture of the scene its boxes describe. Write down the frames to leave
# out:
#

# %%
empty_frames = [0, 6]

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert empty_frames == sorted([BLACK, EMPTY])

# %% [markdown]
# ## Look at the boxes and decide
#
# A misspelled name says what it meant, but a box with no name does not. Look at both boxes next to examples of the
# classes they could belong to. {class}`.DetectionCrops` cuts each box out of its frame as a small image. Cut them from
# a view that leaves out the two empty frames, since neither holds a picture to cut from:
#

# %%
clean = View(planted, Indices(empty_frames, exclude=True))
crops = DetectionCrops(clean)
labels = np.asarray(crops.labels)
names = planted.metadata["index2label"]
label_of = {name: index for index, name in names.items()}
shown = [int(np.flatnonzero(labels == label_of[name])[0]) for name in ("truvk", "*")]
for name in ("car", "van", "truck", "bus"):
    shown += [int(index) for index in np.flatnonzero(labels == label_of[name])[:3]]
dep.plot(
    [crops[index][0] for index in shown],
    figsize=(14, 4.5),
    images_per_row=7,
    titles=[names[int(labels[index])] for index in shown],
)

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(shown) == 14
# The `*` box and the middle bus example are one tracked vehicle, in two frames of one video.
assert crops.track_ids[shown[1]] == crops.track_ids[shown[-2]]
assert crops.item_indices[shown[-2]] > crops.item_indices[shown[1]]
assert (
    clean[int(crops.item_indices[shown[1]])][2]["source_id"]
    == clean[int(crops.item_indices[shown[-2]])][2]["source_id"]
)

# %% [markdown]
# The `truvk` box shows a truck with an open cargo bed, like the trucks in the grid, which confirms what the spelling
# suggested. The `*` box shows a long green vehicle from above with a hatch in its roof. The middle bus example shows
# the same vehicle, partly in view at the bottom of its crop: the dataset tracks it under one track id, and labels it
# bus in a later frame of its video. Write down what you decided:
#

# %%
reviewed = {"truvk": "truck", "*": "bus"}

# %% [markdown]
# ## Fix the labels and leave out the empty frames
#
# {class}`.Relabel` rewrites each class name through a mapping. Here the mapping is the one `label_alignment` found
# for the ten names spelled as on the class list, plus your two decisions. With `on_unmatched="raise"`, `Relabel` stops
# with an error if any name is left without a class, rather than dropping its boxes silently. With `drop_empty=False`,
# it also keeps any image that has no boxes, which it would otherwise drop.
#
# The two empty frames cannot be repaired from the data you have. Leave them out with {class}`.Indices` and
# `exclude=True`, as the crops did, and keep the rest:
#

# %%
fixed = View(
    planted,
    [
        Indices(empty_frames, exclude=True),
        Relabel({**alignment["class_remap"], **reviewed}, target=VISDRONE, on_unmatched="raise", drop_empty=False),
    ],
)
left_out = View(planted, Indices(empty_frames))
table = (
    box_counts(planted, "planted")
    .join(box_counts(left_out, "on the left-out frames"), on="class", how="left")
    .join(box_counts(fixed, "fixed"), on="class", how="left")
    .fill_null(0)
)
display(table)
print(f"Frames: {len(planted)} planted, {len(fixed)} fixed")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
before = dict(zip(table["class"], table["planted"], strict=True))
out = dict(zip(table["class"], table["on the left-out frames"], strict=True))
after = box_counts(fixed)
after = dict(zip(after["class"], after["boxes"], strict=True))
assert set(after) <= set(CLASSES)
assert after["truck"] == before["truck"] - out["truck"] + 1
assert after["bus"] == before["bus"] - out["bus"] + 1
assert sum(after.values()) == sum(before.values()) - sum(out.values())
assert sum(out.values()) == 144
assert len(fixed) == len(planted) - 2

# %% [markdown]
# `truvk` and `*` are gone, and truck and bus each have one box more than the frames you kept held. Every other
# difference is a box on the two frames you left out, 144 in all. No box was dropped without you deciding it. The two
# frames are left out, not repaired: if you need them, go back to the source of the data for the files.
#

# %% [markdown]
# ## Conclusion
#
# You received a detection dataset with four defects and found each one with DataEval.
#
# | Defect | Found with | Fixed by |
# | ------ | ---------- | -------- |
# | `truvk`, a misspelled class name | `label_alignment`, then a spelling ranking and a look at the box | `Relabel`, to truck |
# | `*`, a box with no class name | `label_alignment`, then a look at the box | `Relabel`, to bus |
# | A black frame | `Outliers`: `zeros`, `contrast`, `entropy` and `depth` | Left out with `Indices` |
# | A frame with no pixels | `Outliers`: `missing` | Left out with `Indices` |
#
# `Outliers` also flagged three ordinary frames, whose sky fills a third of the picture. Looking at them showed that
# they are not defects.
#
# ## What you learned
#
# - A name outside the class list is found by comparing the names in the data with the list. A spelling ranking
#   proposes what a typo meant, and the box decides.
# - An image with no content stands out in its statistics, and so can an ordinary image. Look before you remove.
# - A fix that raises on anything it cannot map keeps every box on the record, and a view leaves out what cannot be
#   fixed without changing the source.
# - A misspelling that lands on another valid class name cannot be found by its name. Only the box can show it, or, in
#   a tracking dataset, a class that changes along a track.
# - If a crop does not show its class, leave the box out or send it back rather than guess. The planted bus is an easy
#   case, and a real box with no name may not be.
#

# %% [markdown]
# ## Next steps
#
# - [Ontology](../concepts/Ontology.md) — Learn how a class list becomes an ontology that names can be aligned with.
# - [Data integrity](../concepts/DataIntegrity.md) — Learn which defects in images and labels to check for, and why.
# - [Align two label spaces](./h2_align_label_spaces.py) — Map one label vocabulary onto another, with a matcher for
#   typos.
# - [Conform and merge datasets with different label vocabularies](./h2_conform_and_merge_datasets.py) — Bring two
#   datasets to one vocabulary and merge them.
#

# %% [markdown]
# ## On your own
#
# Repeat these steps on a detection dataset of your own. Write its class list as an `Ontology` and compare the names
# in the data with it. Run `Outliers` over its images, and look at every flagged image before you leave one out. Look
# at the boxes of every name outside the class list, and write down what each one is. Then fix the names with `Relabel`
# and `on_unmatched="raise"` on a view that leaves out what cannot be fixed, and compare the box counts before and
# after.
#
