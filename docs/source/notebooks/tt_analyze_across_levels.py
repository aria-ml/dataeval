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
# # Analyze a dataset across its levels
#
# Estimated time to complete: 30 minutes
#
# Relevant ML stages: [Data Engineering](../getting-started/roles/ML_Lifecycle.md#data-engineering)
#
# Relevant personas: [Data Engineer](../getting-started/roles/data_engineer.md), [T&E Engineer](../getting-started/roles/te_engineer.md)

# %% [markdown]
# ## What you'll do
#
# - You will build a {class}`.Metadata` object over an aerial object detection dataset and inspect the data levels
#   found inside it.
# - You will investigate how one factor can report two different distributions depending on the level you read it from.
# - You will move between those levels with {meth}`.Metadata.at` and `inherited`.
# - You will run {func}`.compute_stats` over the objects, the whole frames, and the background to create additional
#   metadata factors.
# - You will aggregate per-detection analyses into per-image analyses with {meth}`.Metadata.agg`.
# - You will create new metadata subsets two different ways: {meth}`.Metadata.where` and {meth}`.Metadata.having`.
# - You will cache the whole metadata structure with {meth}`.Metadata.save`, so you can compute once and analyze
#   every which way.

# %% [markdown]
# ## What you'll learn
#
# - You'll learn why an image-level factor read from detection rows is weighted by detection count, and what that
#   does to every statistic computed over it.
# - You'll learn when to aggregate values upward, and why DataEval will not do it for you.
# - You'll learn that a statistic measured over an object, over its whole image, and over the background behind it are
#   three different factors that land at two different levels — and how to tell when the third one is worth reading.
# - You'll learn why filtering metadata puts it out of sync with its own dataset, and how to bring the two back
#   into sync.

# %% [markdown]
# ## What you'll need
#
# - Environment Requirements
#   - `dataeval`
#   - `maite-datasets`
# - Basic familiarity with [polars](https://docs.pola.rs) expressions
# - Roughly 300 MB of disk for the dataset download
#
# Reading [Metadata Levels](../concepts/MetadataLevels.md) first is helpful but not required. This tutorial builds the
# same ideas from data.

# %% [markdown]
# ## Introduction
#
# When you analyze an object detection dataset, you are condensing rich visual information into a handful of tabular
# data points. If you are not careful about how this data is structured, hidden variables — such as the frequency of
# objects within a single scene — can unknowingly skew your evaluation results.
#
# For example, image brightness is a global property, whereas a bounding box applies to only a single detection.
# Flattening these distinct metrics into a single table forces a destructive compromise. You must either duplicate
# image-level facts across every detection (which artificially weights images that happen to contain more objects)
# or average the detection facts up to the image (which destroys individual bounding boxes as units of analysis).
#
# DataEval eliminates this compromise. It stores each metric at the exact granularity it was measured — known as a
# **level** — and natively records how these levels relate.
#
# In this tutorial, you will learn how to use levels in DataEval to analyze datasets without losing granular context
# or introducing aggregation bias.

# %% [markdown]
# ### Setup
#
# You'll begin by importing the libraries used throughout this tutorial.

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
from dataeval.bias import Balance
from dataeval.config import set_max_processes
from dataeval.core import compute_stats
from dataeval.data import Indices, Limit, Shuffle, View
from dataeval.exceptions import MetadataFormatError
from dataeval.flags import ImageStats

# Read and measure four frames at a time rather than one. See Step 5.
set_max_processes(4)

# %% [markdown]
# ## Step 1: Load the data and inspect its levels
#
# You are going to work with SeaDrone, an aerial dataset in which a drone flies over water and annotates the swimmers,
# boats and buoys it sees. Every image carries the drone's telemetry at the moment of capture — `altitude`, `speed`,
# `gimbal_pitch` — and every annotated object carries its own `object_size`.
#
# If this data is already on your computer, change `"./data"` to wherever it is stored and set `download` to `False`.
#
# You will construct a [dataset view](./h2_build_dataset_views.py) using {class}`.View` with the {class}`.Shuffle` and
# {class}`.Limit` operations to sample 200 of its images so the tutorial runs quickly. They are drawn with a seeded
# shuffle rather than taken from the front of the split: SeaDrone is ordered by capture, so the first 200 images come
# from only a couple of sorties and are not representative of the collection. The seed keeps everyone's sample
# identical.
#
# ```{note}
# **200 is a deliberately small sample, chosen so this page rebuilds in a couple of minutes rather than ten.** It is
# not a recommended analysis size. Every effect this tutorial demonstrates — the gap between a per-image and a
# per-detection mean, the sharpness difference between objects and background, the correlation between altitude and
# object count — is a structural property of how levels work, not an artifact of the sample, and each one holds (and
# generally sharpens) on the full split. Raise or drop the `Limit` to check for yourself; the exact figures quoted
# below will move, but every conclusion drawn from them will not.
# ```

# %%
dataset = View(SeaDrone(root="./data", image_set="val", download=True), [Shuffle(seed=0), Limit(200)])
metadata = Metadata(dataset)

print("levels      :", metadata.levels)
print("level counts:", metadata.level_counts)

# %% [markdown]
# Every count quoted in the prose below — and every count derived from it later — follows from that second line, so the
# tutorial checks it rather than trusting it. If your copy of the dataset disagrees, this fails immediately and says so,
# instead of leaving the text quietly describing numbers you are not seeing.

# %%
EXPECTED_COUNTS = {"unit": 200, "instance": 1305}

if dict(metadata.level_counts) != EXPECTED_COUNTS:
    raise AssertionError(
        f"This tutorial's prose was written against {EXPECTED_COUNTS} (maite-datasets 0.0.21), "
        f"but your dataset yields {dict(metadata.level_counts)}. Every idea below still holds — "
        "levels, weighting, aggregation — but the specific figures quoted will not match your output."
    )

# %% [markdown]
# This metadata holds **two levels at once**: 200 rows where one row is one image, and 1305 rows where one row is one
# detection. This is not one table filtered two ways, but one table holding two different things — an altitude is a
# fact about a flight, an object size is a fact about a swimmer.
#
# A `Metadata` object can describe several **levels** at once, and which ones it holds depends on the kind of dataset
# (see [What a level is](../concepts/MetadataLevels.md#what-a-level-is) for a detailed breakdown). For an object
# detection dataset, the image level is named `unit` and the detection level is named `instance`. Use
# {meth}`.Metadata.rows_at` to look at either one.

# %%
print("one row per image:")
print(metadata.rows_at("unit").select("item_index", "altitude", "speed").head(3))

print("\none row per detection:")
print(metadata.rows_at("instance").select("item_index", "class_label", "object_size").head(3))

# %% [markdown]
# Notice the difference between the two `item_index` columns. The first three detections all come from image 0,
# because one image may hold many detections. Across these 200 images there are 1305 detections, so an image holds
# about six on average. That spreading out from images to detections is what the rest of this tutorial is about.

# %% [markdown]
# ## Step 2: Analyze the same factor at two levels
#
# `altitude` is recorded once per image. But a detection can report an altitude too — the altitude of the image it was
# found in. DataEval stores the value once and hands it down through a process called downwards propagation
# (see [Propagation: values move down, never up](../concepts/MetadataLevels.md#propagation-values-move-down-never-up)),
# so you can read it from either level.
#
# SeaDrone writes `-1` where the drone's telemetry was unavailable, so exclude those rows first. A level tells you where
# the data lives; it does not promise that the data was recorded.

# %%
valid = pl.col("altitude") >= 0
per_image = metadata.rows_at("unit").filter(valid)["altitude"]
per_detection = metadata.rows_at("instance").filter(valid)["altitude"]

print(f"per image     n={len(per_image):5d}  mean={per_image.mean():.1f} m  std={per_image.std():.1f}")
print(f"per detection n={len(per_detection):5d}  mean={per_detection.mean():.1f} m  std={per_detection.std():.1f}")

# %% [markdown]
# **The two means disagree, and neither is wrong** — they simply measure different things.
#
# The drone did not fly higher on average than it flew. A high-altitude image simply sees more objects — from 80 m you
# frame a lot of water — so high flights contribute more detection rows than low ones. Averaging altitude across
# detections therefore weights each flight by how many things it happened to see.
#
# This aggregation bias extends far beyond altitude. Any image-level factor computed from detection rows is skewed
# by object frequency. This distortion impacts any downstream metrics, such as the correlation metrics reported by
# `Balance`, the bin thresholds for continuous factors, and the class distributions evaluated by `Diversity`. When
# analyzing an object detection dataset, unrecognized aggregation bias can result in unexpected bias evaluation results.

# %% [markdown]
# ## Step 3: Choose your level with `at()`
#
# {meth}`.Metadata.at` returns a copy of the metadata specified at the given level. The stored data does not change;
# what changes is which rows the array-shaped accessors project.

# %%
print("factor_data at instance (default):", metadata.factor_data.shape)
print("factor_data at unit               :", metadata.at("unit").factor_data.shape)

# %% [markdown]
# These two representations address fundamentally different analytical questions:
#
# - **At the `instance` level** — "Across all annotated objects, how do these factors co-vary?" Because every
#   individual object is treated as a separate observation, crowded images will have a disproportionate influence on
#   the overall statistics. This is the correct level when your focus is on object-specific behavior.
# - **At the `unit` level** — "Across all images, how do these factors co-vary?" Each image or frame is treated as a
#   single observation, meaning every image contributes equally regardless of the number of detections within it.
#
# Neither level is inherently "more correct" than the other. The error lies in answering an image-level question with
# detection-level data (or vice versa).
#
# ```{important}
# Prefer using {meth}`.Metadata.at` over directly modifying the {attr}`.Metadata.view` attribute in place.
# Because evaluators maintain a reference to the original `Metadata` object, changing its view dynamically
# can silently alter the behavior of previously initialized evaluators. In contrast, {meth}`.Metadata.at` returns a lightweight,
# independent copy that shares underlying binning and structuring computations. This allows you to evaluate multiple
# levels of the same dataset concurrently without side effects.
# ```
#
# Moving the metadata view *upward* introduces a fundamental constraint. Because class labels are defined at the
# detection (`instance`) level, attempting to access {attr}`.Metadata.class_labels` at the image (`unit`) level
# will raise an error rather than fabricating an arbitrary summary.

# %%
try:
    labels_per_image = metadata.at("unit").class_labels
except ValueError as error:
    print("as expected:", str(error).split(".")[0])

# %% [markdown]
# An image holds several detections, or none, so there is no single class label to return. You will come back to this in
# Step 7, where it turns bias analysis at the image level into an interesting question rather than an impossible one.

# %% [markdown]
# ## Step 4: Isolate columns with `inherited`
#
# While {meth}`.Metadata.at` selects the *rows* you analyze, the `inherited` parameter controls the *columns* (factors).
#
# By default, evaluating a specific level includes all accessible factors: its natively defined factors plus any factors
# propagated downwards from a higher level. Setting `inherited=False` strictly isolates the analysis to the factors
# defined natively **at that level**.

# %%
own_only = Metadata(dataset, inherited=False)

print("at instance, inherited=True :", len(metadata.factor_names), "factors")
print("at instance, inherited=False:", sorted(own_only.factor_names))

# %% [markdown]
# Two of SeaDrone's factors — `object_id` and `object_size` — belong to an annotated object rather than the image it
# resides in. Setting `inherited=False` isolates these factors. This enables you to measure variation strictly among
# the objects themselves, without the results being diluted by 18 replicated columns of image-level drone telemetry.
#
# Together, these two parameters form a simple matrix (as detailed in
# [Choosing which factors to analyze: inherited](../concepts/MetadataLevels.md#choosing-which-factors-to-analyze-inherited)).
# The view defines the rows, `inherited` restricts the columns, and the intersection dictates the analysis scope:
#
# | view | `inherited=True` (default) | `inherited=False` |
# | --- | --- | --- |
# | `at("unit")` — 200 rows | 18 image factors | *the same 18* |
# | `at("instance")` — 1305 rows | those 18 **+** `object_id`, `object_size` | `object_id`, `object_size` |
#
# The row count remains unaffected by `inherited`. At the coarsest level (`unit`), the two configurations yield
# identical results because there is no higher level from which to inherit factors.

# %% [markdown]
# ## Step 5: Measure your own factors
#
# Every factor analyzed so far is provided by the dataset itself. SeaDrone's telemetry files describe the drone's
# state, and its annotation files declare the bounding boxes. Because these data points were recorded prior to
# evaluation, they are referred to as **extrinsic** factors.
#
# {func}`.compute_stats` generates factors of the opposite type: **intrinsic** factors, which are derived directly
# from the pixel data. You can specify which statistics to extract by providing {class}`.ImageStats` flags.
#
# For object detection datasets, {func}`.compute_stats` measures these statistics across up to three distinct regions.
# Selecting the correct region involves the same conceptual framing as choosing the metadata level in Step 2:
#
# - `per_image` computes statistics over the entire image frame.
# - `per_target` computes statistics strictly within each annotated bounding box.
# - `per_background` computes statistics over all pixels *outside* the bounding boxes (the scene minus the objects).
#
# ```{note}
# This cell reads and measures 200 4K frames and takes a couple of minutes. It is by far the most expensive step in
# the tutorial — and the reason the sample is capped at 200 — so Step 10 shows you how to pay for it once.
#
# It is also the reason the first cell called {func}`.set_max_processes` with 4: {func}`.compute_stats` reads each
# frame independently, so it splits the work across worker processes. Four workers cut this step to roughly a third
# of its single-process time, for a few hundred megabytes more memory — one frame's pixels are the largest thing any
# worker holds, and the workers are forked, so most of what they occupy is shared with the parent rather than copied.
#
# Note that a positive count is taken literally rather than capped at your core count, so four workers are started
# whether or not there are four cores to run them on. Pass a negative number to size the pool relative to the machine
# instead — `-1` for one worker per core, `-2` to leave a core free — or leave it unset, the default, to keep
# everything in one process.
# ```

# %%
stats = compute_stats(
    dataset,
    stats=ImageStats.VISUAL_SHARPNESS | ImageStats.VISUAL_BRIGHTNESS,
    per_image=True,
    per_target=True,
    per_background=True,
    normalize_pixel_values=False,
)

print("measured:", sorted(stats["stats"]))

# %% [markdown]
# The output dictionary contains the raw statistic names. It does not say which level they belong to. However, the
# accompanying `source_index` records the level of each measurement. When you pass the entire result object to
# {meth}`.Metadata.add_factors`, DataEval reads these source indices and assigns each measurement to its respective
# level.

# %%
extrinsic = set(metadata.factor_names)
metadata.add_factors(stats)

print("new factors:", sorted(set(metadata.factor_names) - extrinsic))
print("dropped    :", metadata.dropped_factors)

# %% [markdown]
# The single `sharpness` statistic is mapped into three distinct factors, with the naming convention indicating their
# level:
#
# - `instance_sharpness` is assigned to the detection (`instance`) level, representing an individual annotated object.
# - `unit_sharpness` is assigned to the image (`unit`) level.
# - `unit_background_sharpness` is also assigned to the image (`unit`) level. Backgrounds are a property of the full
#   frame, not of any individual object. Consequently, there is no such thing as an object-level background — which is
#   what `dropped_factors` reports, mapping each skipped factor name to the reason it was skipped.
#
# If you query the `instance_sharpness` column from the image-level view, the values will be null. Just as
# `class_labels` could not summarize multiple objects per image in Step 3, there is no single "object sharpness" for
# an image containing multiple varying detections.

# %%
print(
    metadata
    .rows_at("unit")
    .select("item_index", "unit_sharpness", "unit_background_sharpness", "instance_sharpness")
    .head(3)
)

print("\nand from the detection rows, where the image value propagates down:")
print(metadata.rows_at("instance").select("item_index", "unit_sharpness", "instance_sharpness").head(3))

# %% [markdown]
# ### Inspect `background_fraction` to validate backgrounds
#
# When using `per_background`, DataEval always returns a `background_fraction` factor: the share of the image left
# unmasked after the annotations are removed. This factor is essential for validation, because a background statistic
# measured over a few percent of an image is noise wearing a measurement's clothes. Read `background_fraction` before
# trusting any background measurement.

# %%
units = metadata.rows_at("unit")
print(
    units.select(
        pl.col("unit_background_fraction").min().alias("smallest"),
        pl.col("unit_background_fraction").median().alias("median"),
    )
)

# %% [markdown]
# The median image retains over 99% of its pixels for the background, and even the densest image retains 80%. Because
# SeaDrone annotates small objects within large 4K frames, its background statistics are computed over plenty of
# pixels. Conversely, if a dataset contains bounding boxes that cover most of the frame, the resulting low
# `background_fraction` would render its background statistics unreliable.
#
# ### Analyze region differences
#
# With validated backgrounds, you can now compare the regions. Sharpness is a good first metric for aerial imagery,
# because it tells you whether the objects are resolvable against the scene around them.

# %%
print(f"sharpness  object     : {metadata.rows_at('instance')['instance_sharpness'].mean():.1f}")
print(f"           whole image: {units['unit_sharpness'].mean():.1f}")
print(f"           background : {units['unit_background_sharpness'].mean():.1f}")

# %% [markdown]
# **The annotated objects are roughly 2.5× sharper than the scene around them.** The background (water) is smooth,
# while the objects (swimmers or boats) produce high-frequency edges. That gap is the signal an object detector
# lives on.
#
# Because of this gap, using an image-level sharpness threshold (e.g., to discard blurry frames) will predominantly
# measure the sharpness of the water, not the objects. If you need to verify that your *objects* are resolvable, you
# must read `instance_sharpness` — a separate factor at the detection level, and the only one that describes the
# targets themselves.
#
# Measuring the background separately provides a secondary, subtler utility. Consider the relationship between the
# background and the whole-image measurement:

# %%
sharper = units.select((pl.col("unit_sharpness") > pl.col("unit_background_sharpness")).sum()).item()
print(f"images sharper as a whole than their own background: {sharper} / {len(units)}")

# %% [markdown]
# In 197 out of 200 images, the whole-image sharpness score exceeds the background-only score. Even though the
# annotations occupy a median of only 0.5% of the total pixel area, they carry enough high-frequency edges to drag the
# whole-image number measurably upward.
#
# This demonstrates why the background is worth isolating: `unit_sharpness` is contaminated by the very objects you
# were trying to describe the scene around.
#
# ### When the background is not worth measuring
#
# Calculating `per_background` requires a masking pass over every image, which carries a computational cost, and it
# does not always tell you anything new. Compare the same regional splits for brightness:

# %%
gap = (pl.col("unit_brightness") - pl.col("unit_background_brightness")).abs()

print(f"brightness whole image: {units['unit_brightness'].mean():.1f}")
print(f"           background : {units['unit_background_brightness'].mean():.1f}")
print(f"largest gap on any single image: {units.select(gap.max()).item():.1f}")

# %% [markdown]
# The difference is negligible. Across all 200 images, the two brightness readings never diverge by more than
# 2 grey levels.
#
# This occurs because brightness is an average over pixels. Masking out 0.5% of them leaves the mean virtually
# unchanged, because the objects are a rounding error in the pixel count. Sharpness is not an average in that sense —
# it responds to localized high-gradient extremes, so the same handful of object pixels moves it a lot.
#
# **`background_fraction` predicted both outcomes.** A statistic that averages over area needs the background to be a
# meaningfully different area before it will report anything new; one that responds to extremes does not.

# %% [markdown]
# ## Step 6: Aggregate values upward with `agg()`
#
# By design, DataEval propagates values strictly *downward* through the hierarchy. It will not automatically average
# object-level properties (such as bounding box size) and assign the result to the parent image. This prevents
# unintended data conflation and implicit aggregation bias.
#
# To explicitly generate these summaries, use {meth}`.Metadata.agg`. This method rolls child rows upward into the
# specified parent level and registers the result as an ordinary factor there.

# %%
enriched = metadata.agg(
    "instance",
    "unit",
    pl.len().alias("n_objects"),
    pl.col("object_size").mean().alias("mean_object_size"),
    pl.col("instance_sharpness").mean().alias("mean_object_sharpness"),
)

new_factors = sorted(set(enriched.at("unit").factor_names) - set(metadata.at("unit").factor_names))
print("new factors at the image level:", new_factors)
print(enriched.rows_at("unit").select("item_index", "altitude", "n_objects", "mean_object_size").head(5))

# %% [markdown]
# The factor `n_objects` is now an explicit, per-image factor capturing object density. All level-based rules apply to
# it normally: it bins over 200 distinct image values rather than 1305 detection values, it propagates back down to
# detections when accessed from the `instance` view, and it carries exactly one unit of weight per image.
#
# This upward aggregation unlocks analyses that were previously impossible.

# %%
crowding = enriched.rows_at("unit").filter(valid)
print("altitude vs objects-per-image, correlation:")
print(round(float(crowding.select(pl.corr("altitude", "n_objects")).item()), 3))

# %% [markdown]
# The positive correlation directly confirms the mechanism discussed in Step 2: higher flights frame more objects.
# Because this dynamic is now quantified as a formal factor, you can condition on it, bin by it, or pass it to a bias
# evaluator — which is exactly what the next step covers.
#
# Consider the second new factor: `mean_object_sharpness`. This is a **fourth** way of answering "how sharp is this
# image?", separate from the three regional measurements in Step 5.

# %%
print(
    enriched
    .rows_at("unit")
    .select("item_index", "n_objects", "unit_sharpness", "unit_background_sharpness", "mean_object_sharpness")
    .head(5)
)

# %% [markdown]
# Three of these columns are direct pixel measurements, whereas one is an aggregated summary. `unit_sharpness` and
# `unit_background_sharpness` were measured over literal regions of the image. In contrast, `mean_object_sharpness` is
# the average of the detection rows contained within the image.
#
# Consider rows 1 and 2: Image 1's `mean_object_sharpness` of 61.7 is the sharpness of a single detected object.
# Image 2's score of 33.1 blends five distinct object measurements together. Both report a single float, and only the
# companion `n_objects` column tells you whether the value is a lone measurement or a blended average.
#
# Because averaging components is fundamentally different from measuring the whole, DataEval keeps measured and
# aggregated factors in separate columns, so you never have to wonder which one you are holding.
#
# ```{note}
# The `agg` method contains a deliberate restriction to prevent statistical distortion. It will refuse to aggregate a
# factor that was *inherited* from above (e.g., averaging an image's altitude across all detections within it). Doing
# so would weight each image by its detection count all over again, compounding the aggregation bias discussed in
# Step 2. If you need to perform such an operation, pass `unique_by=` to name the entity that should be counted once.
# This safeguard is not triggered when counting rows via `pl.len()`, as it reads no column values at all.
# ```

# %% [markdown]
# ## Step 7: Ask a bias question at each level
#
# The {class}`.Balance` evaluator calculates the normalized mutual information between each factor and a target
# conditioning axis. When evaluating at the detection (`instance`) level, the standard target axis is the object's
# class label.

# %%
per_object = Balance().evaluate(metadata)
ranked = per_object.balance.sort("mi_value", descending=True)
print(ranked.head(5))

print("\nobject_size, for comparison:")
print(ranked.filter(pl.col("factor_name") == "object_size"))

# %% [markdown]
# As demonstrated in Step 3, evaluating at the image (`unit`) level yields an error because images lack a single class
# label. To work around this, specify an alternative conditioning axis. By supplying the aggregated `n_objects` factor
# to the `label=` parameter, you can measure what is associated with image density instead: "What factors correlate
# most strongly with a crowded image?"

# %%
per_image_balance = Balance(label="n_objects").evaluate(enriched.at("unit"))
print(per_image_balance.balance.sort("mi_value", descending=True).head(5))

# %% [markdown]
# **The two evaluations produce distinct profiles, and both are correct.**
#
# At the object level, `class_label` tops the table at 1.0 against itself, and the rest of the ranking is led
# by `storage` and `object_id` — identifiers recording which sortie an object was annotated in and which object it
# was — with `altitude` and `gimbal_pitch` in between. Which class you are looking at is largely a function of which
# flight you are looking at. Note where `object_size` lands by comparison: near the bottom of the table, about a
# tenth of `storage`'s score. The intuitive guess — that a boat and a swimmer are told apart by how big the box is —
# is not what this dataset reports.
#
# At the image level, the `n_objects` axis tops the table at 1.0 against itself, and `storage` leads the rest.
# Object crowding is fundamentally a property of the specific flight sortie, not of the individual
# objects. At the detection level that signal is diluted across every individual object in the dataset.
#
# This demonstrates the core utility of metadata levels: running the exact same evaluator over the exact same dataset
# while isolating completely different populations. The image-level analysis is impossible until you choose the `unit`
# population and name a valid conditioning axis for it.

# %% [markdown]
# ## Step 8: Subset populations using `where` and `having`
#
# Both {meth}`.Metadata.where` and {meth}`.Metadata.having` accept a Polars predicate and return a filtered `Metadata`
# instance containing a subset of the rows. Their mechanical difference lies in **the hierarchical direction the filter
# propagates**. The clearest way to demonstrate this is by applying identical predicates to both methods.
#
# In SeaDrone, class 1 represents a `swimmer`. However, "all swimmer detections" and "all images containing swimmers"
# define structurally different populations.

# %%
swimmer_rows = metadata.where(pl.col("class_label") == 1, level="instance")
swimmer_images = metadata.having(pl.col("class_label") == 1, level="instance")

print("unfiltered:", metadata.level_counts)
print("where     :", swimmer_rows.level_counts)
print("having    :", swimmer_images.level_counts)

# %% [markdown]
# - **`where` retains matching rows in isolation.** The individual swimmer detections survive the filter, while the
#   overall image count remains entirely untouched. The `where` method never filters upwards, meaning the parent
#   images are preserved regardless of whether their child detections were excluded.
# - **`having` retains parent entities that contain a match.** The filter propagates upwards, keeping only the images
#   that hold at least one swimmer. Crucially, once a parent image is retained, *all* of its child detections are
#   retained as well, not just the swimmers. This functions as a standard "images containing class X" filter.
#
# Both behaviors are governed by a single, strict hierarchical rule:
#
# ```{important}
# **A row is preserved only if every ancestor it possesses is also preserved.**
# ```
#
# Applying a filter at the image level ensures the cut propagates perfectly downward: if you retain only high-altitude
# images, you retain exactly the detections contained within those specific frames.

# %%
high = metadata.where(pl.col("altitude") > 60, level="unit")
print("altitude > 60 m:", high.level_counts)

# %% [markdown]
# ## Step 9: Synchronize metadata with the dataset
#
# Filtering metadata only purges *rows* from the internal dataframes; it does not remove the underlying *items* from
# the dataset object that the metadata is bound to. Consequently, all 200 original images remain in the dataset, and
# anything subsequently computed from that dataset — embeddings above all — still describes the original 200.
#
# Pairing a filtered metadata object with an unfiltered dataset would be a silent misalignment. To prevent this,
# DataEval flags filtered metadata instances. Evaluators that pair metadata with embeddings — {class}`.Outliers`,
# {class}`.Coverage` and {class}`.Prioritize` — check that flag and refuse a filtered instance outright, rather than
# trying to detect a mismatch that is undetectable in the dangerous case.

# %%
print("is_filtered:", high.is_filtered)

# %% [markdown]
# To synchronize the dataset with your filtered metadata, use {meth}`.Metadata.selected_items`. This method returns
# the array of surviving item indices. By wrapping these indices in {class}`.Indices` and building a dataset
# {class}`.View`, you establish a synchronized slice of the original dataset without modifying it.

# %%
items = high.selected_items()
matching_dataset = View(dataset, Indices(items.tolist()))

print("surviving items :", len(items))
print("matching dataset:", len(matching_dataset))

# %% [markdown]
# Any metrics or embeddings derived from `matching_dataset` will now perfectly correspond row-for-row with the `high`
# metadata instance.
#
# This synchronization succeeds only because the `high` filter was applied cleanly along item boundaries (the image
# level). Conversely, the `where` filter in Step 8 kept 907 of 1305 detections while preserving all 200 images. A
# dataset can only return full images; it cannot selectively return "three out of four" detections from a single
# frame. Rather than build a dataset slice that cannot correspond to its metadata, `selected_items` enforces this
# boundary and raises when you ask it for one.

# %%
try:
    swimmer_rows.selected_items()
except ValueError as error:
    print("as expected:", str(error).split(".")[0])

# %% [markdown]
# ## Step 10: Cache the metadata structure
#
# Constructing the initial `Metadata` object was cheap because it parsed telemetry and annotation files, not pixel
# data. Step 5 introduced the true bottleneck: reading all 200 high-resolution frames off disk, decoding them, and
# measuring pixel statistics across three regions apiece. Every operation since has been arithmetic over the resulting
# tabular rows.
#
# To avoid repeating this expensive work, use {meth}`.Metadata.save` to serialize the structure to disk, and
# {meth}`.Metadata.load` to restore it.

# %%
enriched.save("./data/seadrone-metadata.dem")
reloaded = Metadata.load("./data/seadrone-metadata.dem", dataset)

print("levels     :", reloaded.levels)
print("counts     :", reloaded.level_counts)
print("agg factors:", sorted(set(reloaded.at("unit").factor_names) - set(metadata.at("unit").factor_names)))

# %% [markdown]
# The full structure survives serialization: both levels, the links between them, the intrinsic factors measured by
# `compute_stats`, and the upward derivations generated by `agg`. The minutes of pixel measurement are now a file
# you can reload in well under a second.
#
# Notably, binning parameters are *not* baked into the serialization. The file stores raw factor values, so one file
# serves whatever `continuous_factor_bins` you later want from it.
#
# ```{warning}
# The serialization format is strictly a **computational cache, not an interchange format**. It mirrors DataEval's
# internal layout, which may evolve between releases. Files written by an older version may be rejected by a newer
# installation. This is an explicit safety mechanism: failing outright is preferable to silently injecting stale rows
# into a mutated schema.
#
# Production scripts should wrap their loads in a `try` that catches {exc}`.MetadataFormatError`, to handle an
# incompatible layout gracefully. If you require metadata that outlives a DataEval upgrade, write
# {attr}`.Metadata.dataframe` to a Parquet file instead.
# ```

# %%
try:
    metadata = Metadata.load("./data/seadrone-metadata.dem", dataset)
except MetadataFormatError:
    metadata = Metadata(dataset)
    metadata.save("./data/seadrone-metadata.dem")

print("ready:", metadata.level_counts)

# %% [markdown]
# ## What you learned
#
# - Metadata holds several levels at once, one kind of row per level, and `level_counts` tells you how many of each.
# - An image-level factor read from detection rows is weighted by detections per image. That weighting quietly
#   reshapes bin edges, correlations, and diversity scores.
# - {meth}`.Metadata.at` chooses the rows and `inherited` chooses the columns. The two are independent.
# - {func}`.compute_stats` measures intrinsic factors over up to three regions — object, whole image, and the
#   background behind the annotations — and {meth}`.Metadata.add_factors` places each at its own level from the
#   result's `source_index`. Read `background_fraction` before you read any background statistic.
# - {meth}`.Metadata.agg` moves values up only when you ask, and refuses the cases where the answer would be weighted
#   by fan-out. An averaged-up factor and a measured-whole factor are different columns on purpose.
# - {class}`.Balance` runs at either level, but only after you name an axis it can condition on: the class label at
#   `instance`, an aggregated factor such as `n_objects` at `unit`. The two answers describe different populations.
# - {meth}`.Metadata.where` keeps matching rows; {meth}`.Metadata.having` keeps the entities that have a match. Both
#   obey one rule: a row survives only if every ancestor it has survives.
# - A filtered metadata no longer matches its dataset. {meth}`.Metadata.selected_items` realigns them, and refuses
#   when no dataset subset corresponds.
# - {meth}`.Metadata.save` caches the structure so the dataset is walked once.
#
# ## Next steps
#
# - [Metadata Levels](../concepts/MetadataLevels.md) — The full level model, including the four-level graph used by
#   tracking datasets.
# - [Identify bias and correlations](./tt_identify_bias.py) — The bias evaluators used here, in depth.
# - [How to reason about factor binning across levels](./h2_bin_factors_by_level.py) — Why each factor is discretized at
#   its own level, with the numbers.
# - [How to specify custom statistics on object detection datasets](./h2_custom_image_stats_object_detection.py) — The
#   `compute_stats` flags and region switches from Step 5, on their own terms.
#
# ## On your own
#
# Try applying multi-level metadata analysis to your own object detection or tracking dataset:
#
# - Create a {class}`.Metadata` object and inspect `metadata.levels` and `metadata.level_counts` to see the hierarchy of your data.
# - Compare an image-level factor at both the `unit` and `instance` levels using {meth}`.Metadata.at` to evaluate potential aggregation bias.
# - Use {func}`.compute_stats` with `per_image=True`, `per_target=True`, and `per_background=True` to compute regional factors across your dataset.
# - Aggregate detection-level metrics up to the image level using {meth}`.Metadata.agg` and run {class}`.Balance` to check for dataset correlations.
