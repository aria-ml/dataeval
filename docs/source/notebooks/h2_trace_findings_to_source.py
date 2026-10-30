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
# # Trace a finding back to its source
#
# An evaluator tells you that something is wrong and where, but not what it looks like. {class}`.Outliers` returns a
# mapping keyed on {class}`.SourceIndex`, an address naming one row of your dataset, and the next question is to see
# that row. See [Metadata levels](../concepts/MetadataLevels.md#naming-a-row-keys-or-addresses).
#
# Use this guide to:
#
# - see what a linter flagged before you decide whether it is a real problem
# - trace a finding through a {class}`.View` back to the untransformed data on disk
# - turn findings into factors so bias and balance analysis can see them
#
# You need `dataeval`, `maite-datasets` and `matplotlib`. This guide uses SeaDrone.

# %% tags=["remove_cell"]
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
from maite_datasets.object_detection import SeaDrone

from dataeval import Metadata
from dataeval.config import set_max_processes
from dataeval.core import image_stats
from dataeval.data import Limit, Resize, SourceLocator, View
from dataeval.flags import ImageStats
from dataeval.quality import Outliers
from dataeval.types import SourceIndex

set_max_processes(1)

# %%
dataset = View(SeaDrone(root="./data", image_set="val", download=True), Limit(50))
print(dataset)

# %% [markdown]
# ## Read an address
#
# Run {class}`.Outliers` over the detections and look at what comes back. With `per_target=True` the keys are
# {class}`.SourceIndex` objects and not plain integers, because an item index alone cannot name one box out of several.

# %%
stats = image_stats(
    dataset, stats=ImageStats.DIMENSION | ImageStats.VISUAL, per_target=True, normalize_pixel_values=False
)
outliers = Outliers().from_stats(stats, per_target=True)

for address, metrics in list(outliers.outliers.items())[:5]:
    print(f"{address!r:<24} {', '.join(metrics)}")

# %% [markdown]
# Each address has three fields: `item`, `key` and `level`. {func}`.image_stats` leaves `level` unstated, so a null key
# is an item's own row and a key is one of its labels.

# %% [markdown]
# The result holds whole-image findings, keyed `SourceIndex(24)` with no key, alongside per-detection ones:

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
# ## Follow one address
#
# {class}`.SourceLocator` binds a dataset once and resolves addresses against it. Hand it the same object you computed
# the statistics over, because an address's `item` is a position in whatever {func}`.image_stats` walked.

# %%
locator = SourceLocator(dataset)

print(f"item_level = {locator.item_level}")
print(f"levels     = {locator.levels}")

found = locator[address]
print(f"\n{address!r} is a {found.level}")
print(f"  box   = {found.box}")
print(f"  label = {found.label}")
print(f"  image = {found.pixels.shape}")


# %% [markdown]
# The address resolved to `instance` without stating a level, because this is an image task and a keyed address names a
# label. An accessor a level does not carry, such as `.box` on a whole-image address, raises a `TypeError`.
#
# Retrieval is lazy: building a `SourceItem` reads nothing. `found.pixels` is the whole image, because context is most
# of what you need to judge a finding, and {meth}`.SourceItem.crop` is the cut-out.


# %% tags=["hide-input"]
def show(axis, pixels, title, box=None):
    """Draw a (C, H, W) array, optionally with a box on it."""
    axis.imshow(np.transpose(pixels, (1, 2, 0)).astype(np.uint8))
    axis.set_title(title, fontsize=9)
    axis.axis("off")
    if box is not None:
        x0, y0, x1, y1 = box
        axis.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="red", linewidth=1.5))


# %%
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
show(axes[0], found.pixels, f"item {found.item_index}: the scene", box=found.box)
show(axes[1], found.crop(), "crop(): the detection")
show(axes[2], found.crop(region="context", padding=1.0), "crop(context, padding=1.0)")
plt.tight_layout()
plt.show()

# %% [markdown]
# `crop()` shares its implementation with {class}`.DetectionCrops`, but the defaults differ: `crop()` leaves the
# detection's aspect ratio alone, while `DetectionCrops` squares crops for a model. To reproduce what an embedding saw,
# pass {attr}`.DetectionCrops.policy`:
#
# ```python
# crops = DetectionCrops(dataset)
# found.crop(policy=crops.policy)  # byte-identical to the crop crops[i] holds
# ```

# %% [markdown]
# ## Move between levels
#
# A `SourceItem` knows the rows around it. {meth}`.SourceItem.at` climbs to a containing row and
# {meth}`.SourceItem.within` descends to the rows inside one. Video has four levels, and a frame and a track both carry
# keys that only the level tells apart. See [Metadata levels](../concepts/MetadataLevels.md).

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
# The keys of an evaluator's result and the addresses the locator produces are the same objects, so membership works.
# Only the minimal spelling matches: `SourceIndex(3, 7)` and `SourceIndex(3, 7, "instance")` name one row but are
# different dictionary keys, which is why producers leave the level unstated.

# %% [markdown]
# ## See what the view did
#
# Statistics are computed over a view, so a finding is about transformed data. {attr}`.SourceItem.datum` is what was
# measured and {attr}`.SourceItem.source_datum` is what sits underneath, with no transform applied.

# %%
resized = View(dataset, Resize(128))
on_resized = SourceLocator(resized)[SourceIndex(found.item_index, found.key)]

print(f"measured shape  = {on_resized.pixels.shape}")
print(f"source shape    = {on_resized.source_pixels.shape}")
print(f"measured box    = {np.round(on_resized.box, 1)}")
print(f"source box      = {np.round(on_resized.source_datum[1].boxes[found.key], 1)}")
print(f"source item     = {on_resized.source_item_index}")

# %% [markdown]
# `source_item_index` walks the whole view chain, so it names a row of the original dataset however deep the views are
# stacked.

# %% [markdown]
# ## Feed a finding back
#
# The addresses an evaluator returns and the addresses {meth}`.Metadata.add_factors` accepts are the same vocabulary, so
# a finding goes back in as a factor without a translation step. Bias, balance and diversity analysis can then see it as
# an ordinary column.

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
# The addresses spanned two levels, so the one array became two columns, `unit_is_outlier` on the image rows and
# `instance_is_outlier` on the detection rows, each named for the level it was measured at.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert address.level is None
assert address.key is not None
assert found.level == "instance"
whole_image = whole_images[0] if whole_images else SourceIndex(address.item)
try:
    _ = locator[whole_image].box
    raise AssertionError("a whole-image address has no box")
except TypeError:
    pass
assert on_resized.pixels.shape != on_resized.source_pixels.shape
assert on_resized.source_item_index == found.item_index
assert "instance_is_outlier" in rows.columns
assert "unit_is_outlier" in metadata.rows_at("unit").columns

# %% [markdown]
# ## Next steps
#
# - [Metadata levels](../concepts/MetadataLevels.md) — Learn the level graph that an address names a row in.
# - [Specify custom statistics on object detection datasets](./h2_custom_image_stats_object_detection.py) —
#   Choose flags and read `source_index`.
# - [Find outliers in metadata factors](./h2_find_factor_outliers.py) — Threshold measured factors at each level
#   and trace the flagged rows.
