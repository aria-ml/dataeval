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
# # How to trace a finding back to its source

# %% [markdown]
# ## Problem statement
#
# An evaluator tells you *that* something is wrong and *where*, but not *what it looks like*. {class}`.Outliers` hands
# back a mapping keyed on {class}`.SourceIndex` — an address naming one row of your dataset — and the next question is
# always the same one: show me that row.
#
# This guide is the way back. It covers how to read an address, how to retrieve the image, frame, track or detection it
# names, how to move between the levels around it, and how to hand a finding back to {class}`.Metadata` as a factor.

# %% [markdown]
# ### When to use
#
# - Eyeballing what a linter flagged, before deciding whether it is a real problem
# - Tracing a finding through a {class}`.View` back to the untransformed data on disk
# - Walking a video's level graph: this detection, in that frame, on that track
# - Turning findings into factors so bias and balance analysis can see them

# %% [markdown]
# ### What you will need
#
# 1. An object detection dataset (we'll use SeaDrone from maite-datasets)
# 1. A Python environment with the following packages installed:
#    - `dataeval`
#    - `maite-datasets`
#    - `matplotlib`

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
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from maite_datasets.multiobject_tracking import to_multiobject_tracking_dataset
from maite_datasets.object_detection import SeaDrone

from dataeval import Metadata
from dataeval.config import set_max_processes
from dataeval.core import compute_stats
from dataeval.data import Limit, Resize, SourceLocator, View
from dataeval.flags import ImageStats
from dataeval.quality import Outliers
from dataeval.types import SourceIndex

set_max_processes(1)

# %%
dataset = View(SeaDrone(root="./data", image_set="val", download=True), Limit(50))
print(dataset)

# %% [markdown]
# ## An address is what an evaluator hands you
#
# Run {class}`.Outliers` over the detections and look at what comes back. With `per_target=True` the keys are
# {class}`.SourceIndex` objects rather than plain integers, because an item index alone cannot name one box out of
# several.

# %%
stats = compute_stats(
    dataset, stats=ImageStats.DIMENSION | ImageStats.VISUAL, per_target=True, normalize_pixel_values=False
)
outliers = Outliers().from_stats(stats, per_target=True)

for address, metrics in list(outliers.outliers.items())[:5]:
    print(f"{address!r:<24} {', '.join(metrics)}")

# %% [markdown]
# Each address has three fields — `item`, `key` and `level` — and here only the first two are set. That is the
# **task-generic** spelling: an address with a key names one of its item's labels, whatever level that turns out to be.
# {func}`.compute_stats` emits it because it does not need to know whether it measured an image or a video.
#
# See [Metadata levels](../concepts/MetadataLevels.md#naming-a-row-keys-or-addresses) for the whole vocabulary.

# %% [markdown]
# Both kinds of address are in there. `per_image` is on by default and `per_target` was switched on above, so the
# result holds whole-image findings — keyed `SourceIndex(24)`, no key — alongside per-detection ones. **A null key is
# an item's own row and a key is one of its labels**, which is all you need to tell them apart:

# %%
detections = [a for a in outliers.outliers if a.key is not None]
whole_images = [a for a in outliers.outliers if a.key is None]
print(f"{len(detections)} detection findings, {len(whole_images)} whole-image findings")

# Pick a flagged detection whose image holds several, so there is something to compare it against. Which detections a
# linter flags depends on the data, so fall back to the first detection of the busiest image when it flags none.
per_item = Counter(index.item for index in stats["source_index"] if index.key is not None)
if detections:
    address = max(detections, key=lambda a: per_item[a.item])
else:
    address = next(a for a in stats["source_index"] if a.item == per_item.most_common(1)[0][0] and a.key is not None)

print(f"\nitem  = {address.item}")
print(f"key   = {address.key}")
print(f"level = {address.level}   (unstated)")
print(f"str   = {address}")

# %% [markdown]
# ## Following one address
#
# {class}`.SourceLocator` binds a dataset once and resolves addresses against it. **Hand it the same object you
# computed the statistics over** — an address's `item` is a position in whatever {func}`.compute_stats` walked, so a
# locator over something else resolves it to different data.

# %%
locator = SourceLocator(dataset)

print(f"item_level = {locator.item_level}")
print(f"levels     = {locator.levels}")

found = locator[address]
print(f"\n{address!r} is a {found.level}")
print(f"  box   = {found.box}")
print(f"  label = {found.label}")
print(f"  image = {found.pixels.shape}")

# An accessor a level does not carry raises rather than answering None, so a mistake is a
# message instead of a null propagating into a plot. Branch on `.level` for mixed findings.
whole_image = whole_images[0] if whole_images else SourceIndex(address.item)
try:
    _ = locator[whole_image].box
except TypeError as error:
    print(f"\n{whole_image!r}.box -> {error}")

# %% [markdown]
# The address resolved to `instance` without stating a level, because this is an image task and a keyed address names a
# label. The same tuple on a video dataset would resolve the same way — that is what
# {meth}`.SourceIndex.resolve` is for.
#
# Retrieval is lazy: building a `SourceItem` reads nothing, so a thousand addresses cost nothing until you look at
# them. `found.pixels` is the **whole image**, because context is most of what you want when judging whether a finding
# is real. {meth}`.SourceItem.crop` is the cut-out.


# %%
def show(axis, pixels, title, box=None):
    """Draw a (C, H, W) array, optionally with a box on it."""
    axis.imshow(np.transpose(pixels, (1, 2, 0)).astype(np.uint8))
    axis.set_title(title, fontsize=9)
    axis.axis("off")
    if box is not None:
        x0, y0, x1, y1 = box
        axis.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="red", linewidth=1.5))


fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
show(axes[0], found.pixels, f"item {found.item_index}: the scene", box=found.box)
show(axes[1], found.crop(), "crop(): the detection")
show(axes[2], found.crop(region="context", padding=1.0), "crop(context, padding=1.0)")
plt.tight_layout()
plt.show()

# %% [markdown]
# `crop()` shares its implementation with {class}`.DetectionCrops`, so the same four arguments cut the same pixels
# in both places. The *defaults* differ on purpose — `crop()` leaves the detection's own aspect ratio alone, while
# `DetectionCrops` squares crops for a model — so to reproduce exactly what an embedding saw, pass the policy that
# view was built with — {attr}`.DetectionCrops.policy` hands it over so you never restate the four arguments:
#
# ```python
# crops = DetectionCrops(dataset)
# found.crop(policy=crops.policy)  # byte-identical to the crop crops[i] holds
# ```

# %% [markdown]
# ## Moving between levels
#
# One address names one row and says nothing about what that row sits inside — that is what lets a single tuple reach
# every level of a graph that is a diamond rather than a chain. The locator supplies the parentage the address leaves
# out.
#
# {meth}`.SourceItem.at` climbs to a containing row; {meth}`.SourceItem.within` descends to the rows inside one.

# %%
scene = found.at("unit")
print(f"{found.address!r} sits in {scene.address!r}")

siblings = scene.within("instance")
print(f"item {scene.item_index} holds {len(siblings)} detection(s):")

flagged = set(outliers.outliers)
for sibling in siblings:
    mark = "<- flagged" if sibling.address in flagged else ""
    print(f"  {sibling.address!r:<20} key={sibling.key:<4} label={sibling.label:<3} {mark}")

# %% [markdown]
# That last loop is the point of an address being a *value*: the keys of an evaluator's result and the addresses the
# locator produces are the same objects, so membership just works. Note that the minimal spelling is what matches —
# `SourceIndex(3, 7)` and `SourceIndex(3, 7, "instance")` name one row but are different dictionary keys, which is why
# producers leave the level unstated.

# %% [markdown]
# ## What the view did
#
# Statistics are computed over a view, so a finding is a finding about *transformed* data. {attr}`.SourceItem.datum` is
# what was measured; {attr}`.SourceItem.source_datum` is what sits underneath, with no transform applied.

# %%
resized = View(dataset, Resize(128))
on_resized = SourceLocator(resized)[SourceIndex(found.item_index, found.key)]

print(f"measured shape  = {on_resized.pixels.shape}")
print(f"source shape    = {on_resized.source_pixels.shape}")
print(f"measured box    = {np.round(on_resized.box, 1)}")
print(f"source box      = {np.round(on_resized.source_datum[1].boxes[found.key], 1)}")
print(f"source item     = {on_resized.source_item_index}")

# %% [markdown]
# An outlier that is only an outlier after a transform is a finding about the transform. `source_item_index` walks the
# whole view chain, not one link, so it names a row of the original dataset however deep the views are stacked.

# %% [markdown]
# ## All four levels, on video
#
# An image dataset has two levels — the item and its labels — so an address never has to state one. Video has four,
# and the two in the middle have no unkeyed spelling: a frame and a track both carry keys, and only the level tells
# them apart.

# %%
videos = to_multiobject_tracking_dataset(
    videos=[np.zeros((3, 3, 48, 48), dtype=np.uint8)],
    labels=[[[0, 1], [], [1]]],
    bboxes=[[[[1, 1, 8, 8], [10, 10, 20, 20]], [], [[12, 12, 22, 22]]]],
    track_ids=[[[5, 9], [], [9]]],
    metadata=None,
    classes=["diver", "boat"],
    name="toy-tracking",
)

tracking = SourceLocator(videos)
print(f"item_level = {tracking.item_level}")
print(f"levels     = {tracking.levels}")

# %%
sequence = tracking[SourceIndex(0)]
print(f"{sequence.address!r} is a {sequence.level} and holds:")
for level in ("unit", "track", "instance"):
    inside = sequence.within(level)
    print(f"  {len(inside)} {level}: {[item.address for item in inside]}")

# %% [markdown]
# Read those keys carefully, because each level names its rows with a different column:
#
# - `unit` keys on `unit_index` — the frame's own number
# - `track` keys on `track_id` — what the tracker assigned, so `5` and `9` rather than `0` and `1`
# - `instance` keys on `target_index`, which counts detections across the **whole sequence**, not within a frame
#
# A detection carries its own parentage, so climbing from one reaches both of its parents:

# %%
detection = tracking[SourceIndex(0, 2)]
print(f"{detection.address!r} label={detection.label} box={detection.box}")
print(f"  seen in    {detection.at('unit').address!r}")
print(f"  belongs to {detection.at('track').address!r}")
print(f"  came from  {detection.at('sequence').address!r}")

track = detection.at("track")
observations = track.within("instance")
print(f"\ntrack {track.key} was observed {len(observations)} times:")
for observation in observations:
    print(f"  {observation.address!r} in frame {observation.at('unit').key}, box {observation.box}")

# %% [markdown]
# ```{note}
# A `track_id` is unique only *within* a sequence — two videos reusing id `9` hold two unrelated tracks — so a track
# never spans items, and everything inside it comes from the one datum. A frame and a track are *siblings* rather than
# one containing the other; ask each for its `instance` rows to find where they meet.
# ```

# %% [markdown]
# ## Feeding a finding back
#
# The addresses an evaluator returns and the addresses {meth}`.Metadata.add_factors` accepts are the same vocabulary,
# so a finding goes back in as a factor without a translation step. Bias, balance and diversity analysis can then see
# it as an ordinary column.

# %%
metadata = Metadata(dataset)
metadata.add_factors(stats)

addresses = list(stats["source_index"])
metadata.add_factors(
    {"is_outlier": [address in flagged for address in addresses]},
    source_index=addresses,
)

rows = metadata.rows_at("instance")
print(rows.select("item_index", "target_index", "instance_is_outlier").head(5))
print(f"\nflagged {rows['instance_is_outlier'].sum()} of {rows.height} instance rows")

# %% [markdown]
# The addresses spanned two levels, so the one array became **two columns**: `unit_is_outlier` on the image rows and
# `instance_is_outlier` on the detection rows. That is the same split {meth}`.Metadata.add_factors` applied to the
# statistics themselves, and it is why a factor's name carries the level it was measured at.

# %% [markdown]
# ## What you learned
#
# 1. **An evaluator's keys are addresses.** {class}`.Outliers` and {class}`.Duplicates` return {class}`.SourceIndex`
#    objects, and the minimal spelling — level unstated — is what producers emit and what compares equal.
# 1. **{class}`.SourceLocator` follows one.** Bind it to the object you measured, index it with an address, and read
#    the image, frame, track or detection it names. Nothing is read until you look.
# 1. **{meth}`~.SourceItem.at` and {meth}`~.SourceItem.within` supply the parentage an address leaves out**, in both
#    directions, over all four levels.
# 1. **{attr}`~.SourceItem.source_datum` is the data before the view touched it**, which is how you tell a finding
#    about your data from a finding about your pipeline.
# 1. **Findings go back in as factors**, keyed on the addresses that produced them.

# %% [markdown]
# ## Next steps
#
# - [Metadata levels](../concepts/MetadataLevels.md) — the level graph an address names a row in
# - [](./h2_custom_image_stats_object_detection.py) — choosing flags and reading `source_index`
# - [](./h2_visualize_cleaning_issues.py) — the wider linting workflow
# - [Migrating to v1.2](../migration/v1.2.md) — what changed when `SourceIndex` became an address
