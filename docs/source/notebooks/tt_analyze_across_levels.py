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
# Learn how metadata holds the images and the objects of a dataset together and how to analyze a factor at the level
# you choose.
#
# Estimated time to complete: 30 minutes
#
# Relevant ML stages: [Data Engineering](../getting-started/roles/ML_Lifecycle.md#data-engineering)
#
# Relevant personas: [Data Engineer](../getting-started/roles/data_engineer.md), [T&E Engineer](../getting-started/roles/te_engineer.md)

# %% [markdown]
# ## What you will do
#
# - You will build a {class}`.Metadata` object over an aerial object detection dataset and inspect the data levels
#   found inside it.
# - You will investigate how one factor can report two different distributions depending on the level you read it from.
# - You will move between those levels with {meth}`.Metadata.at` and `inherited`.
# - You will run {func}`.image_stats` over the objects, the whole frames, and the background to create additional
#   metadata factors.
# - You will aggregate per-detection analyses into per-image analyses with {meth}`.Metadata.agg`.
# - You will create new metadata subsets two different ways: {meth}`.Metadata.where` and {meth}`.Metadata.having`.

# %% [markdown]
# ## What you will learn
#
# - You will learn why an image-level factor read from detection rows is weighted by detection count, and what that does
#   to every statistic computed over it.
# - You will learn when to aggregate values upward, and why DataEval will not do it for you.
# - You will learn that a statistic measured over an object, over its whole image, and over the background behind it
#   gives three different factors that land at two different levels. You will also learn how to tell when the third one
#   is worth reading.

# %% [markdown]
# ## What you will need
#
# - Environment Requirements
#   - `dataeval`
#   - `maite-datasets`
# - Basic familiarity with [polars](https://docs.pola.rs) expressions
# - Roughly 1.1 GB of disk for the dataset download
#
# Reading [Metadata levels](../concepts/MetadataLevels.md) first is helpful but not required. This tutorial builds the
# same ideas from data.

# %% [markdown]
# ## Introduction
#
# When you analyze an object detection dataset, you condense visual information into a handful of tabular data points.
# If you are not careful about how you structure this data, hidden variables, such as the frequency of objects within a
# single scene, can skew your evaluation results without your noticing.
#
# For example, image brightness is a global property, whereas a bounding box applies to only a single detection.
# Flattening these metrics into a single table forces a compromise. You must either duplicate image-level facts across
# every detection (which artificially weights images that happen to contain more objects) or average the detection facts
# up to the image (which destroys individual bounding boxes as units of analysis).
#
# DataEval avoids this compromise. It stores each metric at the granularity it was measured, known as a **level**, and
# records how the levels relate.
#
# In this tutorial, you will learn how to use levels in DataEval to analyze datasets without losing granular context
# or introducing aggregation bias.

# %% [markdown]
# ## Setup
#
# Import the libraries used throughout this tutorial.

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
from dataeval.core import image_stats
from dataeval.data import Limit, Shuffle, View
from dataeval.flags import ImageStats

# Read and measure four frames at a time rather than one. See the "Measure your own factors" section.
set_max_processes(4)

# %% [markdown]
# ## Load the data and inspect its levels
#
# You will work with SeaDrone, an aerial dataset in which a drone flies over water and captures images of the swimmers,
# boats and buoys it sees. Every image carries the drone's telemetry at the moment of capture (`altitude`, `speed` and
# `gimbal_pitch`), and every annotated object carries its own `object_size`.
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
# 200 is a small sample, chosen so that this page rebuilds in a couple of minutes rather than ten. It is not a
# recommended analysis size. Every effect this tutorial demonstrates is a structural property of how levels work and not
# an artifact of the sample: the gap between a per-image and a per-detection mean, the sharpness difference between
# objects and background, and the correlation between altitude and object count. Each one holds, and generally sharpens,
# on the full split. Raise or drop the `Limit` to check for yourself. The figures quoted below will move, but the
# conclusions drawn from them will not.
# ```

# %%
dataset = View(SeaDrone(root="./data", image_set="val", download=True), [Shuffle(seed=0), Limit(200)])
metadata = Metadata(dataset)

print("levels      :", metadata.levels)
print("level counts:", metadata.level_counts)

# %% [markdown]
# Every count quoted in the prose below, and every count derived from it later, follows from that second line, so the
# tutorial checks it instead of trusting it. If your copy of the dataset disagrees, this check fails immediately and
# says so. Without it, the text would describe numbers you are not seeing.

# %%
EXPECTED_COUNTS = {"unit": 200, "instance": 1305}

if dict(metadata.level_counts) != EXPECTED_COUNTS:
    raise AssertionError(
        f"This tutorial's prose was written against {EXPECTED_COUNTS} (maite-datasets 0.0.21), "
        f"but your dataset yields {dict(metadata.level_counts)}. Every idea below still holds — "
        "levels, weighting, aggregation — but the specific figures quoted will not match your output."
    )

# %% [markdown]
# This metadata holds two levels at once: 200 rows where one row is one image, and 1305 rows where one row is one
# detection. This is one table holding two different things, not one table filtered two ways. An altitude is a fact
# about a flight, and an object size is a fact about a swimmer.
#
# A `Metadata` object can describe several levels at once, and which ones it holds depends on the kind of dataset
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
# about six and a half on average. That spreading out from images to detections is what the rest of this tutorial is
# about.

# %% [markdown]
# ## Analyze the same factor at two levels
#
# `altitude` is recorded once per image. But a detection can report an altitude too: the altitude of the image it
# was found in. DataEval stores the value once and hands it down through a process called downwards propagation
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
# The two means disagree, and neither is wrong. They measure different things.
#
# The drone did not fly higher on average than it flew. A high-altitude image sees more objects: from 80 m you frame a
# lot of water. High flights contribute more detection rows than low ones, so averaging altitude across detections
# weights each flight by how many things it happened to see.
#
# This aggregation bias affects more than altitude. Any image-level factor computed from detection rows is skewed by
# object frequency, and the skew carries into downstream metrics: the correlation metrics reported by `Balance` and the
# class distributions evaluated by `Diversity`. In an object detection dataset, unrecognized aggregation bias can
# produce unexpected bias evaluation results.

# %% [markdown]
# (binning-levels)=
# ### Binning follows the level
#
# Bias evaluators read bin codes and not values, so where the bin edges fall decides what every result means. DataEval
# cuts each factor's bins at the factor's own level. The altitude edges are cut over the 200 image values and not over
# the 1305 detection rows they propagate to, so crowded images do not pull the edges toward themselves, and an image
# with no detections still gets a bin. The code is then propagated down, so an image's bin is the same number on the
# image row and on every detection inside it. {attr}`.Metadata.factor_info` reports the level a factor was defined at,
# which is also the level it was binned at.

# %%
info = metadata.factor_info["altitude"]
print(f"altitude is defined at the {info.level!r} level, binned: {info.is_binned}")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
from dataclasses import dataclass

import numpy as np

from dataeval.protocols import DatasetMetadata, DatumMetadata


def edges(values):
    cut = Metadata.from_factors(
        {"altitude": values}, class_labels=np.zeros(len(values), dtype=int), auto_bin_method="uniform_count"
    )
    return np.array(cut.encoding("altitude").edges)


# the edges come from the 200 image values and not from the 1305 detection rows
image_values = metadata.rows_at("unit")["altitude"].to_numpy()
detection_values = metadata.rows_at("instance")["altitude"].to_numpy()
own = np.array(Metadata(dataset, auto_bin_method="uniform_count").encoding("altitude").edges)
assert np.array_equal(own, edges(image_values))
assert not np.array_equal(own, edges(detection_values))

# an image's bin is the same number on its own row and on each of its detections
by_item = dict(zip(metadata.rows_at("unit")["item_index"], metadata.rows_at("unit")["altitude↕"], strict=True))
detections = metadata.rows_at("instance")
assert all(by_item[item] == code for item, code in zip(detections["item_index"], detections["altitude↕"], strict=True))


# an image with no detections still gets a bin
@dataclass
class BoxTarget:
    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


class Sparse:
    counts = (2, 1, 2, 0)
    metadata = DatasetMetadata(id="sparse", index2label={0: "a", 1: "b"})

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, index):
        count = self.counts[index]
        target = BoxTarget(np.tile([[0.0, 0.0, 10.0, 10.0]], (count, 1)), np.arange(count) % 2, np.ones(count))
        return np.zeros((3, 8, 8), np.uint8), target, DatumMetadata(id=index, **{"altitude_m": float(100 * index)})


sparse = Metadata(Sparse(), auto_bin_method="uniform_count")
info = sparse.factor_info["altitude_m"]
column = "altitude_m↕" if info.is_binned else "altitude_m#" if info.is_digitized else "altitude_m"
assert sparse.level_counts == {"unit": 4, "instance": 5}
assert sparse.rows_at("unit")[column].null_count() == 0

# %% [markdown]
# ## Choose your level with `at()`
#
# {meth}`.Metadata.at` returns a copy of the metadata specified at the given level. The stored data does not change;
# what changes is which rows the array-shaped accessors project.

# %%
print("factor_data at instance (default):", metadata.factor_data.shape)
print("factor_data at unit               :", metadata.at("unit").factor_data.shape)

# %% [markdown]
# These two representations answer different analytical questions:
#
# - At the `instance` level, the question is "Across all annotated objects, how do these factors co-vary?" Every
#   individual object is treated as a separate observation, so crowded images have a disproportionate influence on the
#   overall statistics. This is the correct level when your focus is on object-specific behavior.
# - At the `unit` level, the question is "Across all images, how do these factors co-vary?" Each image or frame is
#   treated as a single observation, so every image contributes equally regardless of the number of detections within
#   it.
#
# Neither level is "more correct" than the other. The error lies in answering an image-level question with
# detection-level data (or vice versa).
#
# ```{important}
# Use {meth}`.Metadata.at` instead of modifying the {attr}`.Metadata.view` attribute in place. Because evaluators keep a
# reference to the original `Metadata` object, changing its view can silently alter the behavior of previously
# initialized evaluators. {meth}`.Metadata.at` returns a lightweight, independent copy that shares the underlying
# binning and structuring computations, so you can evaluate multiple levels of the same dataset concurrently without
# side effects.
# ```
#
# Moving the metadata view upward introduces a constraint. Because class labels are defined at the
# detection (`instance`) level, attempting to access {attr}`.Metadata.class_labels` at the image (`unit`) level
# will raise an error rather than fabricating an arbitrary summary.

# %%
try:
    labels_per_image = metadata.at("unit").class_labels
except ValueError as error:
    print("as expected:", str(error).split(".")[0])

# %% [markdown]
# An image holds several detections, or none, so there is no single class label to return. You will come back to this in
# the "Ask a bias question at each level" section, where you name another conditioning axis to ask the bias question at
# the image level.

# %% [markdown]
# ## Isolate columns with `inherited`
#
# While {meth}`.Metadata.at` selects the rows you analyze, the `inherited` parameter controls the columns (factors).
#
# By default, evaluating a specific level includes all accessible factors: its natively defined factors plus any factors
# propagated downwards from a higher level. Setting `inherited=False` isolates the analysis to the factors defined
# natively at that level.

# %%
own_only = Metadata(dataset, inherited=False)

print("at instance, inherited=True :", len(metadata.factor_names), "factors")
print("at instance, inherited=False:", sorted(own_only.factor_names))

# %% [markdown]
# Two of SeaDrone's factors, `object_id` and `object_size`, belong to an annotated object and not to the image it
# resides in. Setting `inherited=False` isolates these factors. This enables you to measure variation among the objects
# themselves, without the results being diluted by the replicated columns of image-level drone telemetry.
#
# Together, these two parameters form a simple matrix (as detailed in [Choosing which factors to analyze:
# inherited](../concepts/MetadataLevels.md#choosing-which-factors-to-analyze-inherited)). The view defines the rows,
# `inherited` restricts the columns, and the intersection dictates the analysis scope:
#
# | View | `inherited=True` (default) | `inherited=False` |
# | --- | --- | --- |
# | `at("unit")`, 200 rows | the image factors | the same |
# | `at("instance")`, 1305 rows | the image factors + `object_id`, `object_size` | `object_id`, `object_size` |
#
# The row count remains unaffected by `inherited`. At the coarsest level (`unit`), the two configurations yield
# identical results because there is no higher level from which to inherit factors.

# %% [markdown]
# ## Measure your own factors
#
# Every factor analyzed so far is provided by the dataset itself. SeaDrone's telemetry files describe the drone's
# state, and its annotation files declare the bounding boxes. Because these data points were recorded before
# evaluation, they are called **extrinsic** factors.
#
# {func}`.image_stats` generates factors of the opposite type: **intrinsic** factors, which are derived directly
# from the pixel data. You can specify which statistics to extract by providing {class}`.ImageStats` flags.
#
# For object detection datasets, {func}`.image_stats` measures these statistics across up to three regions. Choosing a
# region works like choosing the metadata level in the "Analyze the same factor at two levels" section:
#
# - `per_image` computes statistics over the entire image frame.
# - `per_target` computes statistics within each annotated bounding box.
# - `per_background` computes statistics over all pixels outside the bounding boxes (the scene minus the objects).
#
# ```{note}
# This cell reads and measures 200 4K frames and takes a couple of minutes. That is why the sample is capped at 200, and
# why the first cell called {func}`.set_max_processes` with 4: {func}`.image_stats` reads each frame independently, so
# it splits the work across worker processes. [Configure global DataEval defaults](./h2_configure_defaults.py)
# explains how to size the pool.
# ```

# %%
stats = image_stats(
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
# The single `sharpness` statistic is mapped into three factors, and the naming convention indicates their level:
#
# - `instance_sharpness` is assigned to the detection (`instance`) level, representing an individual annotated object.
# - `unit_sharpness` is assigned to the image (`unit`) level.
# - `unit_background_sharpness` is also assigned to the image (`unit`) level. Backgrounds are a property of the full
#   frame and not of any individual object, so there is no object-level background. That is what `dropped_factors`
#   reports. It maps each skipped factor name to the reason it was skipped.
#
# If you query the `instance_sharpness` column from the image-level view, the values will be null. Just as
# `class_labels` could not summarize multiple objects per image in the "Choose your level with `at()`" section, there is
# no single "object sharpness" for an image containing several detections.

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
# The three sharpness factors are measured over different regions of the same frame. `unit_background_fraction` says how
# much of the frame the background covers, and it should be read before any background statistic.
# [Compare objects to their surroundings](./tt_compare_objects_to_surroundings.py) compares the regions.

# %% [markdown]
# ## Aggregate values upward with `agg()`
#
# By design, DataEval propagates values only downward through the hierarchy. It does not average object-level properties
# (such as bounding box size) and assign the result to the parent image. This prevents unintended data conflation and
# implicit aggregation bias.
#
# To generate these summaries explicitly, use {meth}`.Metadata.agg`. This method rolls child rows upward into the
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
# it normally: it bins over 200 image values rather than 1305 detection values, it propagates back down to detections
# when accessed from the `instance` view, and it carries one unit of weight per image.
#
# This upward aggregation enables analyses that were not possible before.

# %%
crowding = enriched.rows_at("unit").filter(valid)
print("altitude vs objects-per-image, correlation:")
print(round(float(crowding.select(pl.corr("altitude", "n_objects")).item()), 3))

# %% [markdown]
# The positive correlation supports the mechanism discussed in the "Analyze the same factor at two levels" section:
# higher flights frame more objects. Because this dynamic is now quantified as a factor, you can condition on it, bin by
# it, or pass it to a bias evaluator. The next section passes it to one.
#
# Consider the second new factor, `mean_object_sharpness`. This is a fourth way of answering "how sharp is this image?",
# separate from the three regional measurements in the "Measure your own factors" section.

# %%
print(
    enriched
    .rows_at("unit")
    .select("item_index", "n_objects", "unit_sharpness", "unit_background_sharpness", "mean_object_sharpness")
    .head(5)
)

# %% [markdown]
# Of the three sharpness columns, two are direct pixel measurements and one is an aggregated summary. `unit_sharpness`
# and `unit_background_sharpness` were measured over literal regions of the image. In contrast, `mean_object_sharpness`
# is the average of the detection rows contained within the image.
#
# Consider rows 1 and 2. Image 1's `mean_object_sharpness` of 61.7 is the sharpness of a single detected object. Image
# 2's score of 33.1 averages five object measurements. Both report a single float, and only the companion `n_objects`
# column tells you whether the value is a lone measurement or an average.
#
# Averaging components is different from measuring the whole, so DataEval keeps measured and aggregated factors in
# separate columns. You never have to wonder which one you are holding.
#
# ```{note}
# The `agg` method has a deliberate restriction that prevents statistical distortion. It refuses to aggregate a factor
# that was inherited from above (for example, averaging an image's altitude across all detections within it). Doing so
# would weight each image by its detection count all over again, compounding the aggregation bias discussed in the
# "Analyze the same factor at two levels" section. To perform such an operation, pass `unique_by=` to name the entity
# that should be counted once. The restriction does not apply when you count rows with `pl.len()`, because that reads no
# column values.
# ```

# %% [markdown]
# ## Ask a bias question at each level
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
# As demonstrated in the "Choose your level with `at()`" section, evaluating at the image (`unit`) level yields an error
# because images lack a single class label. To work around this, specify an alternative conditioning axis. By supplying
# the aggregated `n_objects` factor to the `label=` parameter, you can measure what is associated with image density
# instead: "What factors correlate most strongly with a crowded image?"

# %%
per_image_balance = Balance(label="n_objects").evaluate(enriched.at("unit"))
print(per_image_balance.balance.sort("mi_value", descending=True).head(5))

# %% [markdown]
# The two evaluations produce different profiles, and both are correct.
#
# At the object level, `class_label` tops the table at 1.0 against itself, and the rest of the ranking is led by
# `storage` and `object_id`, identifiers recording which sortie an object was annotated in and which object it was, with
# `altitude` and `gimbal_pitch` in between. Which class you are looking at is largely a function of which flight you are
# looking at. `object_size` lands near the bottom of the table by comparison, at about a tenth of `storage`'s score. The
# intuitive guess is that a boat and a swimmer are told apart by how big the box is. This dataset does not report that.
#
# At the image level, the `n_objects` axis tops the table at 1.0 against itself, and `storage` leads the rest. Object
# crowding is a property of the flight sortie and not of the individual objects. At the detection level that signal is
# diluted across every object in the dataset.
#
# This shows what metadata levels are for: you run the same evaluator over the same dataset and isolate different
# populations. The image-level analysis is not possible until you choose the `unit` population and name a valid
# conditioning axis for it.

# %% [markdown]
# ## Subset populations using `where` and `having`
#
# Both {meth}`.Metadata.where` and {meth}`.Metadata.having` accept a Polars predicate and return a filtered `Metadata`
# instance containing a subset of the rows. Their mechanical difference lies in the hierarchical direction the filter
# propagates. To see it, apply the same predicate to both methods.
#
# In SeaDrone, class 1 represents a `swimmer`. However, "all swimmer detections" and "all images containing swimmers"
# define different populations.

# %%
swimmer_rows = metadata.where(pl.col("class_label") == 1, level="instance")
swimmer_images = metadata.having(pl.col("class_label") == 1, level="instance")

print("unfiltered:", metadata.level_counts)
print("where     :", swimmer_rows.level_counts)
print("having    :", swimmer_images.level_counts)

# %% [markdown]
# - `where` retains matching rows in isolation. The individual swimmer detections survive the filter, while the overall
#   image count stays untouched. The `where` method never filters upwards, so the parent images are preserved regardless
#   of whether their child detections were excluded.
# - `having` retains parent entities that contain a match. The filter propagates upwards, keeping only the images that
#   hold at least one swimmer. Once a parent image is retained, all of its child detections are retained as well, not
#   just the swimmers. This works as a standard "images containing class X" filter.
#
# Both behaviors follow one hierarchical rule:
#
# ```{important}
# A row is preserved only if every ancestor it possesses is also preserved.
# ```
#
# A filter at the image level propagates downward: if you keep only high-altitude images, you keep the detections
# contained in those frames, and no others.

# %%
high = metadata.where(pl.col("altitude") > 60, level="unit")
print("altitude > 60 m:", high.level_counts)

# %% [markdown]
# The dataset is untouched by a filter, so a filtered metadata no longer matches the dataset it was built from.
# [Keep metadata in sync with its dataset and cache it](./h2_sync_and_cache_metadata.py) shows how to realign the
# two and how to cache the metadata.

# %% [markdown]
# ## Conclusion
#
# You built a {class}`.Metadata` object over an aerial object detection dataset and read one factor at two levels,
# which showed that an image-level factor read from detection rows is weighted by the number of detections per image.
# You chose rows with {meth}`.Metadata.at` and columns with `inherited`, measured the objects, the whole frames and
# the background with {func}`.image_stats`, and aggregated detection values up to images with
# {meth}`.Metadata.agg`. You then ran {class}`.Balance` at both levels and kept subsets of the dataset with
# {meth}`.Metadata.where` and {meth}`.Metadata.having`.

# %% [markdown]
# ## What you learned
#
# - Metadata holds several levels at once, one kind of row per level, and `level_counts` tells you how many of each.
# - An image-level factor read from detection rows is weighted by detections per image. That weighting reshapes
#   correlations and diversity scores. It does not move bin edges, because DataEval cuts them at the factor's own level.
# - {meth}`.Metadata.at` chooses the rows and `inherited` chooses the columns. The two are independent.
# - {func}`.image_stats` measures intrinsic factors over up to three regions (object, whole image, and the background
#   behind the annotations), and {meth}`.Metadata.add_factors` places each at its own level from the result's
#   `source_index`.
# - {meth}`.Metadata.agg` moves values up only when you ask, and refuses the cases where the answer would be weighted
#   by fan-out. An averaged-up factor and a measured-whole factor are different columns on purpose.
# - {class}`.Balance` runs at either level, but only after you name an axis it can condition on: the class label at
#   `instance`, an aggregated factor such as `n_objects` at `unit`. The two answers describe different populations.
# - {meth}`.Metadata.where` keeps matching rows; {meth}`.Metadata.having` keeps the entities that have a match. Both
#   obey one rule: a row survives only if every ancestor it has survives.
#
# ## Next steps
#
# - [Metadata levels](../concepts/MetadataLevels.md) — The full level model, including the four-level graph used by
#   tracking datasets.
# - [Identify bias and correlations](./tt_identify_bias.py) — The bias evaluators used here, in depth.
# - [Keep metadata in sync with its dataset and cache it](./h2_sync_and_cache_metadata.py) — Realign a filtered
#   metadata with its dataset and cache the metadata so it is built once.
#
# ## On your own
#
# Try applying multi-level metadata analysis to your own object detection or tracking dataset:
#
# - Create a {class}`.Metadata` object and inspect `metadata.levels` and `metadata.level_counts` to see the hierarchy of
#   your data.
# - Compare an image-level factor at both the `unit` and `instance` levels using {meth}`.Metadata.at` to evaluate
#   potential aggregation bias.
# - Use {func}`.image_stats` with `per_image=True`, `per_target=True`, and `per_background=True` to compute regional
#   factors across your dataset.
# - Aggregate detection-level metrics up to the image level using {meth}`.Metadata.agg` and run {class}`.Balance` to
#   check for dataset correlations.
