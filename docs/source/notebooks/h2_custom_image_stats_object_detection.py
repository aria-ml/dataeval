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

# %% [markdown]
# ## Problem statement
#
# When working with object detection datasets, you often need to analyze
# [image statistics](../concepts/DataIntegrity.md#image-statistics-as-a-linting-vocabulary) at different granularities:
#
# - Image-level statistics: Properties of entire images
# - Target-level statistics: Properties of individual bounding boxes within images
# - Background-level statistics: Properties of the pixels no bounding box covers
#
# This guide will show you how to use {func}`.compute_stats` with custom {class}`.ImageStats` flags to capture statistics
# on each of those three regions, and how to hand the result to {class}`.Metadata` so every value lands on the row it
# describes.

# %% [markdown]
# ### When to use
#
# Use this approach when you need fine-grained control over which statistics to compute, especially when:
#
# - Working with object detection datasets with bounding boxes
# - Analyzing both full images and cropped regions (boxes)
# - Separating what was annotated from the scene it was annotated in
# - Optimizing computation by selecting only relevant statistics

# %% [markdown]
# ### What you will need
#
# 1. An object detection dataset (we'll use SeaDrone from maite-datasets)
# 1. A Python environment with the following packages installed:
#    - `dataeval`
#    - `maite-datasets`

# %% [markdown]
# ## Getting started
#
# First import the required libraries needed to set up the example.

# %%
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
from dataeval.core import compute_stats
from dataeval.data import Limit, View
from dataeval.flags import ImageStats

# Statistics are calculated across a process pool. Capping it keeps this example's memory
# footprint predictable; leave it unset to use every available core.
set_max_processes(4)

# %% [markdown]
# ## Load the dataset
#
# Begin by loading an object detection dataset. For this example we are using SeaDrone, an object detection dataset
# containing aerial images captured by drones over marine environments.
#
# We'll use a subset of the dataset to keep computation time reasonable.

# %%
# Load the SeaDrone dataset
sd_dataset = SeaDrone(root="./data", image_set="val", download=True)

# Limit to first 50 images for demonstration
dataset = View(sd_dataset, Limit(50))

print(f"Dataset size: {len(dataset)} images")
print(f"Sample image shape: {dataset[0][0].shape}")
print(f"Sample targets (boxes): {len(dataset[0][1].boxes)} boxes in first image")

# %% [markdown]
# ## Statistics on full images only
#
# Let's calculate statistics on the full images with a custom set of basic statistics.
#
# The {class}`.ImageStats` enum provides fine-grained control over which statistics to compute.
#
# You can combine flags using the `|` (bitwise OR) operator.
#
# `normalize_pixel_values` is passed explicitly throughout this guide. Its default changed to `False` in v1.1, and
# omitting it raises a deprecation warning until the parameter settles.

# %%
# Calculate custom individual statistics for full images only (per_image=True, per_target=False)
results_image_only = compute_stats(
    data=dataset,
    stats=ImageStats.PIXEL_MEAN | ImageStats.DIMENSION_ASPECT_RATIO | ImageStats.VISUAL_SHARPNESS,
    per_image=True,
    per_target=False,
    normalize_pixel_values=False,
)

print(f"Computed statistics: {list(results_image_only['stats'])}")
print(f"\nNumber of results: {len(results_image_only['source_index'])}")
print(f"Total images processed: {results_image_only['image_count']}")

# %% [markdown]
# ### Understanding SourceIndex
#
# The `source_index` field contains {class}`.SourceIndex` objects that track where each statistic came from:
#
# - `item`: The item index in the dataset
# - `target`: The bounding box index (None for full images)
# - `channel`: The channel index, populated only by the deprecated per-channel row path (band groups
#   requested with `channels=` come back as columns instead, so this stays None)
#
# That triple is what lets a single result carry values measured over different regions without ever confusing them.

# %%
# Display first 5 source indices
print("First 5 SourceIndex entries (image-level only):")
for i, src in enumerate(results_image_only["source_index"][:5]):
    print(f"  {i}: item={src.item}, target={src.target}, channel={src.channel}")

print(f"\nAll entries have target=None: {all(src.target is None for src in results_image_only['source_index'])}")

# %% [markdown]
# ## Statistics on bounding boxes only
#
# Now let's compute statistics for just the bounding boxes within the images.

# %%
# Calculate basic pixel statistics for targets only (per_image=False, per_target=True)
results_target_only = compute_stats(
    data=dataset,
    stats=ImageStats.PIXEL_BASIC,
    per_image=False,
    per_target=True,
    normalize_pixel_values=False,
)

print(f"Computed statistics: {list(results_target_only['stats'])}")
print(f"Number of target-level results: {len(results_target_only['source_index'])}")
print(f"Total targets processed: {sum(results_target_only['object_count'])}")

# Display source indices for targets from first image
print("\nSourceIndex entries for targets in first few images:")
for i, src in enumerate(results_target_only["source_index"][:5]):
    print(f"  {i}: image={src.item}, target={src.target}, channel={src.channel}")

# %% [markdown]
# ## Statistics on both full images and bounding boxes
#
# We can also compute statistics at both levels simultaneously.

# %%
# Calculate basic dimension statistics for full images and boxes (per_image=True, per_target=True)
results_both = compute_stats(
    data=dataset,
    stats=ImageStats.DIMENSION_BASIC,
    per_image=True,
    per_target=True,
    normalize_pixel_values=False,
)

print(f"Number of results (images + boxes): {len(results_both['source_index'])}")
print(f"Total images processed: {results_both['image_count']}")
print(f"Total boxes processed: {sum(results_both['object_count'])}")
print(f"Statistics calculated for each image: {list(results_both['stats'])}")

# Separate image-level and box-level results
image_indices = [i for i, src in enumerate(results_both["source_index"]) if src.target is None]
target_indices = [i for i, src in enumerate(results_both["source_index"]) if src.target is not None]

print(f"\nImage-level results: {len(image_indices)}")
print(f"Target-level results: {len(target_indices)}")

# %% [markdown]
# ## Statistics on the background
#
# The two regions above between them cover every annotated thing and every whole frame, but not the thing you often
# actually want to characterize: **the scene an object was found in**. Set `per_background=True` to measure the pixels
# that no bounding box covers.
#
# Background values are returned alongside the whole-image ones, on the same rows, under names prefixed with
# `background_`. They are per-image values — the background is a property of the frame, not of anything inside it.

# %%
results_background = compute_stats(
    data=dataset,
    stats=ImageStats.VISUAL_SHARPNESS,
    per_image=True,
    per_target=True,
    per_background=True,
    normalize_pixel_values=False,
)

print(f"Computed statistics: {sorted(results_background['stats'])}")

# %% [markdown]
# Two constraints are worth knowing before you read any of those numbers.
#
# Only {attr}`.ImageStats.PIXEL` and {attr}`.ImageStats.VISUAL` statistics are computed for the background. A hash or a
# dimension statistic in your flags is still computed for the image and its boxes and simply skipped for the background,
# which has no meaningful hash and no geometry of its own.
#
# More importantly, `background_fraction` — the share of the image left unmasked — is always returned, and should be
# read before the rest. **A background statistic measured over a few percent of an image is noise wearing a
# measurement's clothes.** Boxes are rounded outwards and unioned into the mask, so the background excludes slightly
# more than the annotations strictly cover; where boxes cover an image entirely, every background statistic for it is
# NaN.

# %%
# The result spans image rows and target rows; only the image rows carry a background, so the
# target rows hold NaN here and have to be dropped before the fraction can be summarized.
fraction = pl.Series(results_background["stats"]["background_fraction"]).drop_nulls().drop_nans()

print(f"background_fraction: n={len(fraction)}  min={fraction.min():.3f}  median={fraction.median():.3f}")

# %% [markdown]
# On SeaDrone the annotations are tiny against a 4K frame, so nearly all of every image survives masking and the
# background is measured over plenty of pixels. That is the comfortable case. A dataset whose boxes cover most of the
# frame would report a small fraction here, and its background statistics should be treated with suspicion.

# %% [markdown]
# ## Landing the results on the rows they describe
#
# Splitting a result by hand — the `image_indices` and `target_indices` comprehensions above — works, but it does not
# scale past inspection, and it leaves you holding two arrays with no record of what they mean.
#
# {meth}`.Metadata.add_factors` does that split for you. Hand it the whole result rather than its `stats` mapping and
# it reads the `source_index` too, placing each value on the row it belongs to.

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
# The background factors carry `unit_` because that is where they live: one value per image. Anything DataEval could
# not place — a background value on a detection row, which does not exist — is recorded rather than silently kept.

# %%
print("dropped:", metadata.dropped_factors)

print(
    metadata
    .rows_at("unit")
    .select("item_index", "unit_sharpness", "unit_background_sharpness", "unit_background_fraction")
    .head(5)
)

# %% [markdown]
# Reading those two sharpness columns against each other is the whole point of computing the background separately.

# %%
units = metadata.rows_at("unit")
sharper = units.select((pl.col("unit_sharpness") > pl.col("unit_background_sharpness")).sum()).item()

print(f"images sharper as a whole than their own background: {sharper}/{len(units)}")
print(f"mean sharpness  object={metadata.rows_at('instance')['instance_sharpness'].mean():.1f}")
print(f"                image ={units['unit_sharpness'].mean():.1f}")
print(f"                back  ={units['unit_background_sharpness'].mean():.1f}")

# %% [markdown]
# ## Key takeaways
#
# From this analysis, we've learned:
#
# 1. **Custom Statistics Selection**: The {class}`.ImageStats` flags allow fine-grained control over which statistics to
#    compute, optimizing performance by avoiding unnecessary calculations.
#
# 1. **Granular Analysis**: Using `per_image`, `per_target`, and `per_background`, we can analyze statistics over three
#    different regions of the same dataset:
#
#    - Full images provide context about overall scene properties
#    - Bounding boxes reveal properties of individual objects
#    - Backgrounds describe the scene an object was captured in, with the objects removed
#
# 1. **Guarding the background**: `background_fraction` says how much of an image survived masking, and a background
#    statistic should not be read without it. Hash and dimension statistics are not computed for the background at all.
#
# 1. **SourceIndex Tracking**: The {class}`.SourceIndex` objects allow us to precisely track which image, box, and channel
#    each statistic corresponds to.
#
# 1. **Handing results to Metadata**: Passing a whole result to {meth}`.Metadata.add_factors` uses that tracking to place
#    every value on the row it describes, turning one statistic into one factor per level.

# %% [markdown]
# ## Conclusion
#
# This notebook demonstrated how to use {func}`.compute_stats` with custom {class}`.ImageStats` flags to perform flexible,
# efficient analysis on object detection datasets, over whole images, individual boxes, and the background behind them.
#
# These techniques are valuable for:
#
# - Dataset quality assessment
# - Identifying biases or artifacts
# - Understanding object characteristics
# - Separating a target's properties from those of the scene it sits in
# - Optimizing preprocessing pipelines
# - Detecting outliers or anomalies
#
# Once the factors are on the metadata, the question becomes which population you are analyzing them over — an object,
# an image, or the scene behind it. [Analyze a dataset across its levels](./tt_analyze_across_levels.py) works that
# through end to end on this same dataset.

# %% [markdown]
# ## Next steps
#
# - [Acting on Results](../concepts/ActingOnResults.md) — Learn strategies for addressing dataset issues identified during evaluation.
# - [Clustering](../concepts/Clustering.md) — Understand clustering techniques for grouping similar data points and detecting patterns.
# - [Data Integrity](../concepts/DataIntegrity.md) — Analyze image-level, target-level, and background-level statistics to identify data quality issues.
# - [Dataset Bias and Coverage](../concepts/DatasetBias.md) — Evaluate bias and coverage across metadata factors in your dataset.
# - [Analyze a dataset across its levels](./tt_analyze_across_levels.py) — Perform multi-level dataset analysis across unit, instance, and background factors.
# - [Detecting common augmentations as duplicates](./tt_augmentation_duplicates.py) — Find near-identical images created through synthetic transformations and augmentations.
# - [Introduction to data cleaning](./tt_clean_dataset.py) — Clean and prepare datasets for model training by finding duplicates, outliers, and corrupted data.
# - [How to add intrinsic factors to Metadata](./h2_add_intrinsic_factors.py) — Compute and attach intrinsic factors such as dimensions and pixel statistics to dataset metadata.
# - [How to bin factors by level](./h2_bin_factors_by_level.py) — Discretize continuous factor values into discrete bins for hierarchical analysis.
# - [How to run clustering analysis](./h2_cluster_analysis.py) — Cluster embeddings and factors to discover structural groupings in your data.
# - [How to visualize cleaning issues](./h2_visualize_cleaning_issues.py) — Plot and visualize dataset anomalies, duplicates, and quality issues.
