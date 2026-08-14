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
# # Beyond the Bounding Box: Comparing Objects to Their Surroundings
#
# Learn how to isolate an object's statistics from its environment and identify potential background shortcuts.
#
# Estimated time to complete: 15 minutes
#
# Relevant ML stages: [Data Engineering](../getting-started/roles/ML_Lifecycle.md#data-engineering)
#
# Relevant personas: Data Scientist, ML Engineer, T&E Engineer

# %% [markdown]
# ## What you'll do
#
# - Compute image statistics for the whole frame, individual bounding boxes, and the masked background.
# - Calculate a per-detection separation score to find camouflaged objects.
# - Compare scene-relative ratios across different lighting conditions.
# - Evaluate dataset balance and mutual information to detect background shortcut risks.

# %% [markdown]
# ## What you'll learn
#
# - Learn to separate hard-object detection challenges from hard-scene conditions.
# - Learn to use scene-relative ratios to normalize objects across variable light.
# - Learn to verify whether scenery content is acting as an unintended shortcut for your model.

# %% [markdown]
# ## What you'll need
#
# - Environment Requirements
#   - `dataeval` or `dataeval[all]`
#   - `maite-datasets`
#   - `matplotlib`

# %% [markdown]
# ## Background
#
# A detection's statistics describe the object *and* the conditions it was captured in,
# mixed together. A swimmer photographed at noon and the same swimmer at dusk have
# different brightness, but the same relationship to the water around them. So a raw
# per-target number rarely answers the question you actually have.
#
# Two reference points make those questions answerable, and
# {func}`.compute_stats` produces both in one pass:
#
# - **The background** - every pixel the image's boxes do *not* cover, which is the
#   scene the objects were found in. Comparing a target against it asks
#   *"would anything see this object at all?"*
# - **The other targets** - comparing detections to each other asks
#   *"do these classes actually look different, or is the model being handed a shortcut?"*
#
# Reach for these techniques when you are triaging detector misses to separate "hard object"
# from "hard scene" issues, or when you suspect a class is being identified by its surroundings
# rather than by the object itself.
#
# This tutorial works both questions on a maritime search-and-rescue dataset (SeaDrone), where
# they have a direct operational meaning: a swimmer that does not stand out from the water is
# a swimmer a detector will miss.

# %% [markdown]
# ## Getting started

# %%
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval maite-datasets matplotlib
except Exception:
    pass

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from maite_datasets.object_detection import SeaDrone

from dataeval import Metadata
from dataeval.bias import Balance
from dataeval.config import set_max_processes
from dataeval.core import compute_ratios, compute_stats
from dataeval.data import Limit, View
from dataeval.flags import ImageStats

set_max_processes(4)

# Show every row of the comparison tables below rather than polars' default window - the
# rows this guide reasons about would otherwise be the ones elided.
pl.Config.set_tbl_rows(20)

# Balance's mutual-information estimator warns when a factor has many distinct values,
# which the continuous image statistics here do by construction. Not a finding.
warnings.filterwarnings("ignore", message="The number of unique classes is greater", category=UserWarning)

# %% [markdown]
# ## Load the dataset
#
# SeaDrone is aerial drone footage over water, annotated with swimmers, boats, jetskis,
# buoys, and life-saving appliances. It is a good fit for this question because the
# objects are small and the background - open water - is both large and highly variable
# in brightness, from dark swell to sun glare.

# %%
dataset = View(SeaDrone(root="./data", image_set="val", download=True), Limit(120))

print(f"Images: {len(dataset)}")
print(f"Image shape: {dataset[0][0].shape}")

# %% [markdown]
# ## Compute all three views in one pass
#
# `per_image`, `per_target`, and `per_background` are independent. Enabling all three
# gives, for every image: the whole frame, each annotated box, and the frame with those
# boxes masked out.
#
# Background values come back on the same rows as the whole-image ones, under names
# prefixed with `background_`, because they describe the same thing a whole-image
# statistic describes - the image.

# %%
stats = compute_stats(
    dataset,
    stats=ImageStats.VISUAL_BRIGHTNESS | ImageStats.VISUAL_CONTRAST | ImageStats.DIMENSION_SIZE,
    per_image=True,
    per_target=True,
    per_background=True,
    normalize_pixel_values=False,
)

print(sorted(stats["stats"]))

# %% [markdown]
# Handing the whole result to {meth}`.Metadata.add_factors` places each value on the row
# its `source_index` names. Per-image values land at the `unit` level and per-detection
# values at the `instance` level, so they arrive as separate `unit_` and `instance_`
# factors.

# %%
metadata = Metadata(dataset)
metadata.add_factors(stats)

print(sorted(n for n in metadata.factor_names if "brightness" in n or "fraction" in n))

# %% [markdown]
# Note there is no `instance_background_brightness`. A detection has no background of its
# own - the background belongs to the image - so that split holds no values and is not
# written. It is reported in {attr}`.Metadata.dropped_factors` rather than left as a
# column of nulls.

# %%
print(dict(metadata.dropped_factors))

# %% [markdown]
# ## Read `background_fraction` first
#
# Every other background number is only as trustworthy as the share of the image it was
# measured over. `background_fraction` is that share, and it is returned whenever
# `per_background=True` whether or not you asked for it.
#
# Check it before anything else: a background statistic measured over a few percent of an
# image is noise in measurement's clothing.

# %%
units = metadata.rows_at("unit")
coverage = units["unit_background_fraction"]

print(f"background fraction  min {coverage.min():.3f}   median {coverage.median():.3f}   max {coverage.max():.3f}")

# %% [markdown]
# On this dataset the boxes cover almost nothing - the median image is over 99%
# background. This has a practical consequence worth stating plainly. Compare the whole image
# and background brightness values to observe this:

# %%
drift = (units["unit_background_brightness"] - units["unit_brightness"]).abs().mean()
print(f"mean |background - whole image| brightness: {drift:.3f}")

# %% [markdown]
# The background is essentially the whole frame. So on SeaDrone, masking the objects out
# barely changes the *scene* statistic - what it buys you is the guarantee that the scene
# value is not contaminated by the objects you are about to compare against it.
#
# On a dataset with large or numerous objects - a close-up product line, a crowded street
# scene - `background_fraction` drops well below 1 and the two diverge sharply. That is
# where masking changes the answer rather than just securing it, and it is also where you
# must watch for images whose fraction approaches 0.

# %% [markdown]
# ## Question 1: target vs background
#
# Because unit-level factors propagate down to the instance rows beneath them, one frame
# holds each detection's own brightness *beside* the brightness of the scene behind it.
# That is the whole point of storing the background at the level it was measured at.
#
# Their difference is a per-detection separation score: how far the object sits from its
# surroundings. Near zero means the object is, in brightness terms, camouflaged.

# %%
labels = metadata.index2label
detections = metadata.rows_at("instance").with_columns(
    (pl.col("instance_brightness") - pl.col("unit_background_brightness")).alias("separation"),
    pl.col("class_label").replace_strict(labels, return_dtype=pl.Utf8).alias("class_name"),
)

by_class = (
    detections
    .group_by("class_name")
    .agg(
        pl.len().alias("n"),
        pl.col("separation").mean().round(1).alias("mean_separation"),
        pl.col("separation").std().round(1).alias("sd"),
        pl.col("instance_size").median().alias("median_px"),
    )
    .sort("mean_separation")
)
print(by_class)

# %% [markdown]
# Two things fall out of this table.
#
# **Only one class is brighter than the water it sits on.** Boats read positive; swimmers,
# buoys, jetskis, and life-saving appliances all read negative - they are darker than their
# background. A detector tuned on bright-object contrast has the sign wrong for four of
# the five classes.
#
# **The hardest classes are hard twice over.** Life-saving appliances separate most
# strongly but are among the smallest objects on screen, and jetskis - despite being
# large - sit closest to zero with the widest spread, meaning they are sometimes bright
# and sometimes dark against the water with no consistent polarity.

# %%
# Diverging bars around a meaningful zero: color encodes the sign, length the magnitude.
# The per-detection spread rides in its own band above each bar rather than on top of it,
# so neither the marks nor the value labels are read through a cloud of dots.
BRIGHTER, DARKER, INK, MUTED = "#2a78d6", "#e34948", "#0b0b0b", "#8a8a85"

ordered = by_class.sort("mean_separation")
names = ordered["class_name"].to_list()
means = ordered["mean_separation"].to_numpy()
counts = ordered["n"].to_list()
y = np.arange(len(names))

rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(8, 3.8))

ax.barh(y, means, height=0.34, color=[BRIGHTER if m > 0 else DARKER for m in means], zorder=3)
for i, name in enumerate(names):
    pts = detections.filter(pl.col("class_name") == name)["separation"].to_numpy()
    jitter = rng.uniform(-0.055, 0.055, len(pts))
    ax.scatter(pts, i + 0.30 + jitter, s=7, color=MUTED, alpha=0.35, linewidths=0, zorder=2)

ax.axvline(0, color=INK, linewidth=1, zorder=4)

separation = detections["separation"].to_numpy()
pad = 0.02 * (separation.max() - separation.min())
for i, (m, n) in enumerate(zip(means, counts, strict=True)):
    ax.text(
        m + (pad if m > 0 else -pad),
        i,
        f"{m:+.1f}",
        va="center",
        ha="left" if m > 0 else "right",
        fontsize=9,
        color=INK,
        zorder=5,
    )
    ax.text(
        0.995,
        (i + 0.42) / len(names),
        f"n={n}",
        transform=ax.transAxes,
        va="center",
        ha="right",
        fontsize=8,
        color=MUTED,
    )

ax.set_yticks(y, names)
ax.tick_params(axis="y", length=0)
ax.set_ylim(-0.5, len(names) - 0.1)
ax.set_xlabel("object brightness − background brightness\n← darker than water          brighter than water →")
ax.set_title("How far each class sits from the water behind it", loc="left", fontsize=11)
ax.grid(axis="x", color=MUTED, alpha=0.25, linewidth=0.6)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
plt.show()

# %% [markdown]
# The dots show why the class means are not the whole story: every class has detections
# sitting on the zero line, whatever its average. Those are the individually hard cases.

# %%
camouflaged = detections.filter(pl.col("separation").abs() < 5)
print(
    f"{len(camouflaged)} of {len(detections)} detections ({len(camouflaged) / len(detections):.1%}) "
    f"are within ±5 brightness of their background"
)
print(camouflaged.group_by("class_name").agg(pl.len().alias("n")).sort("n", descending=True))

# %% [markdown]
# These are the first cases to check when a detector misses. You can look up their exact
# image and target indices directly from the metadata:

# %%
print(camouflaged.select("item_index", "target_index", "class_name", "separation").head(8))

# %% [markdown]
# ## Question 2: target vs target
#
# Comparing detections to each other has a trap in it: raw statistics are dominated by
# the conditions of the image each detection came from. Two boats in different lighting
# differ more from each other than a boat and a swimmer in the same frame.
#
# {func}`.compute_ratios` removes that by expressing each box relative to the image it
# sits in, which is what makes detections from different scenes comparable.

# %%
ratios = compute_ratios(stats)
print(sorted(ratios["stats"]))

# %% [markdown]
# Background statistics are deliberately absent from the ratio output: a box has no
# background of its own, so there is nothing to divide.
#
# Adding the ratios as their own factors puts the raw and the scene-relative view side by
# side.

# %%
relative = Metadata(dataset)
relative.add_factors({f"relative_{k}": v for k, v in ratios["stats"].items()}, source_index=ratios["source_index"])

comparison = (
    relative
    .rows_at("instance")
    .with_columns(pl.col("class_label").replace_strict(labels, return_dtype=pl.Utf8).alias("class_name"))
    .group_by("class_name")
    .agg(
        pl.len().alias("n"),
        pl.col("relative_brightness").mean().round(3).alias("brightness_vs_image"),
        pl.col("relative_contrast").mean().round(3).alias("contrast_vs_image"),
        pl.col("relative_size").mean().round(4).alias("size_vs_image"),
    )
    .sort("brightness_vs_image")
)
print(comparison)

# %% [markdown]
# Read as ratios, `1.0` means "indistinguishable from the frame average". Boats sit above
# it, everything else below - the same ordering the separation score gave, now on a scale
# that is comparable across scenes shot in different light.

# %% [markdown]
# ## Does the background predict the class?
#
# The two questions meet here. If the scene behind an object carries information about
# *which* class it is, a model can score well by reading the water instead of the object -
# and will fail the moment the deployment scenery changes.
#
# {class}`.Balance` measures exactly that, as mutual information between each factor and
# the class label. Restrict the metadata to the statistics of interest so the dataset's own
# capture metadata does not crowd the output.

# %%
capture_metadata = [
    "altitude",
    "compass_heading",
    "date_time",
    "drone",
    "frame",
    "gimbal_heading",
    "gimbal_pitch",
    "height",
    "id",
    "image_id",
    "latitude",
    "longitude",
    "object_id",
    "object_size",
    "speed",
    "storage",
    "width",
    "xspeed",
    "yspeed",
    "zspeed",
]
focused = Metadata(dataset, exclude=capture_metadata)
focused.add_factors(stats)

classwise = (
    Balance()
    .evaluate(focused)
    .classwise.with_columns(pl.col("factor_name").cast(pl.Utf8), pl.col("class_name").cast(pl.Utf8))
    .filter(pl.col("factor_name").str.starts_with("unit_background"))
    .sort("factor_name", "class_name")
)
print(classwise)

# %% [markdown]
# Evaluating dataset balance helps you identify dependencies between class labels and background-only factors.
# Here, `unit_background_contrast` exhibits a clear relationship with the class label, revealing that the sea
# state or wave pattern itself is partly predictive of which object is on screen.
#
# This represents a dataset shortcut: a model could achieve high accuracy on this data by learning to
# classify the water instead of the object, which is a major deployment risk.
#
# Crucially, `unit_background_fraction` remains completely independent of the class label.
# This serves as a vital sanity check, confirming that the findings reflect the actual scenery rather
# than a mathematical artifact of bounding box size.

# %% [markdown]
# ## What to do with this
#
# - **Triage missed detections with the separation score.** A missed detection with near-zero
#   separation is a hard-scene problem, not a model-capacity problem, and no amount of
#   retraining on the same imagery will fix it.
# - **Treat a background factor flagged by {class}`.Balance` as a deployment risk.** Test on
#   held-out scenery, not just held-out images, and check whether per-class performance
#   tracks the background factor.
# - **Always read `background_fraction` first.** It decides whether the rest of the
#   background numbers mean anything, and on datasets with large objects it will vary far
#   more than it does here.

# %% [markdown]
# ## Related
#
# - [Data integrity](../concepts/DataIntegrity.md) - the full statistics vocabulary and
#   what `per_background` computes.
# - [Metadata levels](../concepts/MetadataLevels.md) - why the background is stored at the
#   `unit` level and how it reaches the instance rows.
# - [How to specify custom statistics on object detection datasets](h2_custom_image_stats_object_detection.py) -
#   choosing flags and reading `source_index`.
