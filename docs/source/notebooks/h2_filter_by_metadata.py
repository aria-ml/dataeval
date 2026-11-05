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
# # Filter a dataset by metadata factors
#
# {class}`.Filter` keeps the items of a dataset whose metadata factors match a predicate, such as the `altitude` of an
# image or the `object_size` of a detection. The result is a {class}`.View`, so nothing is copied. Factors belong to
# levels, which [Metadata levels](../concepts/MetadataLevels.md) explains.
#
# Use it to analyze a slice of a dataset chosen by how it was collected, or to remove detections that fail a quality
# bar. To filter by class label alone, see [Build dataset views with View and Operation](./h2_build_dataset_views.py).
#
# You need `dataeval` and `maite-datasets`, and about 1.1 GB of disk for the SeaDrone download.

# %% tags=["remove_cell"]
# Google Colab Only
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval maite-datasets
except Exception:
    pass

# %%
import warnings

import polars as pl
from maite_datasets.object_detection import SeaDrone

from dataeval import Metadata
from dataeval.config import set_seed
from dataeval.data import Filter, Limit, Shuffle, View

set_seed(0)  # For reproducibility

# %% tags=["hide-input"]
# A 200-image sample of SeaDrone stands in for your dataset. This guide does not use the binned factors, so skip the
# warnings about binning them.
warnings.filterwarnings("ignore", message=".*(was dropped|binned automatically).*")
dataset = View(SeaDrone(root="./data", image_set="val", download=True), [Shuffle(seed=0), Limit(200)])

# %% [markdown]
# ## Filter on an image-level factor
#
# SeaDrone records the `drone` and the `altitude` of each image. Pass `Filter` a predicate written as the `WHERE` clause
# of a SQL query, with the factor names that {attr}`.Metadata.dataframe` shows. Single quotes mark a text value.

# %%
print("all images:", Metadata(dataset).level_counts)

high = View(dataset, Filter("drone = 'mavic' AND altitude > 60"))
print("mavic above 60 m:", Metadata(high).level_counts)

# %% [markdown]
# `unit` counts images and `instance` counts the detections in them. The filter kept the matching images and every
# detection in them. A polars expression selects the same images, which suits a predicate built in code:

# %%
high_polars = View(dataset, Filter((pl.col("drone") == "mavic") & (pl.col("altitude") > 60)))

print("mavic above 60 m:", Metadata(high_polars).level_counts)

# %% [markdown]
# A missing value never matches a comparison. SeaDrone writes `-1` where the drone's telemetry was unavailable, which is
# not a null, so `altitude IS NULL` keeps nothing and `altitude >= 0` drops those images.

# %%
with_telemetry = View(dataset, Filter("altitude >= 0"))

print("images with telemetry:", len(with_telemetry), "of", len(dataset))

# %% [markdown]
# ## Filter on a detection-level factor
#
# `object_size` describes each detection, so the predicate is about detections, and `drop` chooses what happens to the
# ones that do not match:
#
# - `"both"` keeps the images with a matching detection and removes their other detections. This is the default.
# - `"items"` keeps those images whole.
# - `"detections"` keeps every image and removes the non-matching detections.

# %%
for drop in ("both", "items", "detections"):
    kept = View(dataset, Filter("object_size > 10000", drop=drop))
    print(f"drop={drop!r:13}", Metadata(kept).level_counts)

# %% [markdown]
# Only `"detections"` keeps all 200 images, so use it to clean the labels without shrinking the dataset.

# %% [markdown]
# ## Combine a factor with a class label
#
# Pass `classes` to add a class constraint to a factor predicate. A predicate that reads an image-level factor and a
# detection-level column is about detections, so an image with no swimmer has no row to match and is dropped.

# %%
swimmers_high = View(dataset, Filter("altitude > 60", classes=[1]))

print("swimmers above 60 m:", Metadata(swimmers_high).level_counts)

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert Metadata(dataset).level_counts == {"unit": 200, "instance": 1305}
assert Metadata(high).level_counts == {"unit": 15, "instance": 151}
assert high_polars.resolve_indices() == high.resolve_indices()
assert len(with_telemetry) == 142
sizes = {
    drop: Metadata(View(dataset, Filter("object_size > 10000", drop=drop))).level_counts
    for drop in ("both", "items", "detections")
}
assert sizes["both"] == {"unit": 117, "instance": 237}
assert sizes["items"] == {"unit": 117, "instance": 819}
assert sizes["detections"] == {"unit": 200, "instance": 237}
assert Metadata(swimmers_high).level_counts == {"unit": 16, "instance": 107}

# %% [markdown]
# ## Next steps
#
# - [Analyze a dataset across its levels](./tt_analyze_across_levels.py) — Learn how factors at the image and detection
#   levels differ, and filter metadata with `where` and `having`.
# - [Build dataset views with View and Operation](./h2_build_dataset_views.py) — Chain filters with limits, shuffles and
#   relabeling.
# - [Metadata levels](../concepts/MetadataLevels.md) — Explore dataset, datum, and factor levels in DataEval metadata.
