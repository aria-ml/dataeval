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
# # How to specify custom statistics on object detection datasets
#
# When you analyze an object detection dataset, you often need
# [image statistics](../concepts/DataIntegrity.md#image-statistics-as-a-linting-vocabulary) at three granularities: whole images,
# individual bounding boxes, and the background that no box covers. {func}`.image_stats` with {class}`.ImageStats` flags measures each
# region, and {class}`.Metadata` places every value on the row it describes.
#
# Use it when you want control over which statistics to compute, want to separate what was annotated from the scene it was annotated
# in, or want to avoid computing statistics you do not need.
#
# You need `dataeval` and `maite-datasets`. This guide uses SeaDrone.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval maite-datasets
except Exception:
    pass

# %%
import polars as pl
from maite_datasets.object_detection import SeaDrone

from dataeval import Metadata
from dataeval.config import set_max_processes
from dataeval.core import image_stats
from dataeval.data import Limit, View
from dataeval.flags import ImageStats

# Statistics are calculated across a process pool. Capping it keeps this example's memory
# footprint predictable; leave it unset to use every available core.
set_max_processes(4)

# %% [markdown]
# ## Load the dataset
#
# SeaDrone is an object detection dataset of aerial images captured by drones over marine environments. A subset keeps the computation
# time reasonable.

# %%
# Load the SeaDrone dataset
sd_dataset = SeaDrone(root="./data", image_set="val", download=True)

# Limit to first 50 images for demonstration
dataset = View(sd_dataset, Limit(50))


# %% [markdown]
# ## Compute the statistics
#
# Choose the statistics with {class}`.ImageStats` flags combined by `|`, and the regions with three arguments: `per_image` for whole
# images, `per_target` for each bounding box, and `per_background` for the pixels no bounding box covers. Pass `normalize_pixel_values`
# explicitly: its default changed to `False` in v1.1, and omitting it raises a deprecation warning.
#
# Background values are returned on the same rows as the whole-image ones, under names prefixed with `background_`. Each row carries a
# {class}`.SourceIndex` with an `item` and a `key`: the key is the bounding box index, and `None` for a whole image.

# %%
results_background = image_stats(
    data=dataset,
    stats=ImageStats.VISUAL_SHARPNESS,
    per_image=True,
    per_target=True,
    per_background=True,
    normalize_pixel_values=False,
)

print(f"Computed statistics: {sorted(results_background['stats'])}")

# %% [markdown]
# Only {attr}`.ImageStats.PIXEL` and {attr}`.ImageStats.VISUAL` statistics are computed for the background; a hash or dimension flag is
# skipped for it. Read `background_fraction`, the share of the image left unmasked, before the rest: a background statistic measured
# over a few percent of an image is unreliable, and where boxes cover an image entirely every background statistic for it is NaN. See
# [Data Integrity](../concepts/DataIntegrity.md#image-statistics-as-a-linting-vocabulary).

# %%
# The result spans image rows and target rows; only the image rows carry a background, so the
# target rows hold NaN here and have to be dropped before the fraction can be summarized.
fraction = pl.Series(results_background["stats"]["background_fraction"]).drop_nulls().drop_nans()

print(f"background_fraction: n={len(fraction)}  min={fraction.min():.3f}  median={fraction.median():.3f}")

# %% [markdown]
# On SeaDrone the annotations are tiny against a 4K frame, so nearly all of every image survives masking. A dataset whose boxes cover
# most of the frame would report a small fraction here.

# %% [markdown]
# ## Land the results on the rows they describe
#
# {meth}`.Metadata.add_factors` places each value on the row it belongs to. Hand it the whole result rather than its `stats` mapping, so
# it reads the `source_index` too.

# %%
metadata = Metadata(dataset)
supplied = set(metadata.factor_names)
metadata.add_factors(results_background)

print("levels      :", metadata.levels)
print("level counts:", metadata.level_counts)
print("new factors :", sorted(set(metadata.factor_names) - supplied))

# %% [markdown]
# One `sharpness` statistic became three factors, each named for the level it landed at:
#
# - `instance_sharpness` — measured over each annotated object
# - `unit_sharpness` — measured over each whole image
# - `unit_background_sharpness` — measured over each image's unannotated pixels
#
# The background factors carry `unit_` because they live there: one value per image. A value DataEval cannot place, such as a
# background value on a detection row, is recorded under `dropped_factors`.

# %%
print("dropped:", metadata.dropped_factors)

print(
    metadata
    .rows_at("unit")
    .select("item_index", "unit_sharpness", "unit_background_sharpness", "unit_background_fraction")
    .head(5)
)

# %% [markdown]
# Compare the whole-image and background sharpness:

# %%
units = metadata.rows_at("unit")
sharper = units.select((pl.col("unit_sharpness") > pl.col("unit_background_sharpness")).sum()).item()

print(f"images sharper as a whole than their own background: {sharper}/{len(units)}")
print(f"mean sharpness  object={metadata.rows_at('instance')['instance_sharpness'].mean():.1f}")
print(f"                image ={units['unit_sharpness'].mean():.1f}")
print(f"                back  ={units['unit_background_sharpness'].mean():.1f}")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
# Verify the statistics and the regions they were measured over
import numpy as np

assert set(results_background["stats"]) == {"sharpness", "background_sharpness", "background_fraction"}
assert fraction.to_numpy().min() > 0.9  # nearly all of every image survives masking on SeaDrone
assert np.median(fraction.to_numpy()) > 0.99

# Verify where the values landed: three sharpness factors and a background fraction, and no background on a detection
assert metadata.levels == ("unit", "instance")
assert metadata.level_counts["unit"] == len(dataset) == 50
assert metadata.level_counts["instance"] > 50
assert set(metadata.factor_names) - supplied == {
    "instance_sharpness",
    "unit_sharpness",
    "unit_background_sharpness",
    "unit_background_fraction",
}
assert "instance_background_sharpness" in metadata.dropped_factors

# %% [markdown]
# ## Next steps
#
# - [Data Integrity](../concepts/DataIntegrity.md) — Analyze image-level, target-level, and background-level statistics to identify data quality issues.
# - [Analyze a dataset across its levels](./tt_analyze_across_levels.py) — Perform multi-level dataset analysis across unit, instance, and background factors.
# - [How to add intrinsic factors to Metadata](./h2_add_intrinsic_factors.py) — Compute and attach intrinsic factors such as dimensions and pixel statistics to dataset metadata.
