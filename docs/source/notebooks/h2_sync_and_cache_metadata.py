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
# # How to keep metadata in sync with its dataset and cache it
#
# Filtering {class}`.Metadata` with {meth}`.Metadata.where` or {meth}`.Metadata.having` removes rows from the metadata
# but nothing from its dataset, so anything computed from the dataset, embeddings above all, no longer
# lines up with it. {meth}`.Metadata.selected_items` returns the items that survived so you can build a matching
# {class}`.View`, and {meth}`.Metadata.save` caches the metadata so it is built once. See
# [Metadata levels](../concepts/MetadataLevels.md).
#
# Use it when you filter metadata and then pair it with embeddings or another evaluator, or when building the metadata
# is slow.
#
# You need `dataeval` and `maite-datasets`. This guide uses 200 SeaDrone images and about 1.1 GB of disk for the
# download.

# %% tags=["remove_cell"]
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
from dataeval.data import Indices, Limit, Shuffle, View
from dataeval.exceptions import MetadataFormatError

# %% tags=["hide-input"]
# This guide does not use the binned factors, so skip the warnings about binning them.
warnings.filterwarnings("ignore", message=".*(was dropped|binned automatically).*")

# %% [markdown]
# ## Filter the metadata
#
# Filter on an image-level factor. SeaDrone writes `-1` where the drone's telemetry was unavailable, so the filter on
# `altitude` also drops those images.

# %%
dataset = View(SeaDrone(root="./data", image_set="val", download=True), [Shuffle(seed=0), Limit(200)])
metadata = Metadata(dataset)
high = metadata.where(pl.col("altitude") > 60, level="unit")

print("before:", metadata.level_counts)
print("after :", high.level_counts)
print("is_filtered:", high.is_filtered)

# %% [markdown]
# The metadata holds fewer images, and the dataset still holds all 200. DataEval flags the filtered instance, and
# {class}`.Outliers`, {class}`.Coverage` and {class}`.Prioritize`, which pair metadata with embeddings, refuse it.

# %% [markdown]
# ## Match the dataset to the filter
#
# {meth}`.Metadata.selected_items` returns the indices of the items that survived. A {class}`.View` over them is a slice
# of the dataset that corresponds row for row with the filtered metadata.

# %%
items = high.selected_items()
matching_dataset = View(dataset, Indices(items.tolist()))

print("surviving items :", len(items))
print("matching dataset:", len(matching_dataset))

# %% [markdown]
# Any embeddings or metrics computed from `matching_dataset` now line up with `high`. This works because the filter cut
# along item boundaries. A detection-level filter keeps some of an image's detections, and a dataset can only return
# whole images, so `selected_items` raises instead of building a slice that cannot correspond:

# %%
swimmer_rows = metadata.where(pl.col("class_label") == 1, level="instance")
try:
    swimmer_rows.selected_items()
except ValueError as error:
    print("as expected:", str(error).split(".")[0])

# %% [markdown]
# ## Cache the metadata
#
# Metadata that carries measured factors, such as {func}`.image_stats` over every frame, takes minutes to build. Save
# the structure to disk and load it against the dataset later.

# %%
metadata.save("./data/seadrone-metadata.dem")
reloaded = Metadata.load("./data/seadrone-metadata.dem", dataset)

print("levels :", reloaded.levels)
print("counts :", reloaded.level_counts)

# %% [markdown]
# Both levels, the links between them and the factors survive. The file stores raw factor values and not the binning, so
# one file serves whatever `continuous_factor_bins` you want from it.
#
# ```{warning}
# The file is a cache and not an interchange format. It mirrors DataEval's internal layout, which may change between
# releases, and a newer installation may reject a file written by an older one. Catch {exc}`.MetadataFormatError` and
# rebuild. To keep metadata across an upgrade, write {attr}`.Metadata.dataframe` to a Parquet file.
# ```

# %%
try:
    metadata = Metadata.load("./data/seadrone-metadata.dem", dataset)
except MetadataFormatError:
    metadata = Metadata(dataset)
    metadata.save("./data/seadrone-metadata.dem")

print("ready:", metadata.level_counts)

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert high.is_filtered
assert high.level_counts["unit"] < metadata.level_counts["unit"]
assert len(items) == high.level_counts["unit"]
assert len(matching_dataset) == len(items)

assert reloaded.levels == metadata.levels
assert reloaded.level_counts == metadata.level_counts
assert reloaded.factor_names == metadata.factor_names
assert reloaded.rows_at("instance")["item_index"].to_list() == metadata.rows_at("instance")["item_index"].to_list()

# %% [markdown]
# ## Next steps
#
# - [Metadata levels](../concepts/MetadataLevels.md) — Explore dataset, datum, and factor levels in DataEval metadata.
# - [Analyze a dataset across its levels](./tt_analyze_across_levels.py) — Perform multi-level dataset analysis across
#   unit, instance, and background factors.
# - [How to build dataset views with View and Operation](./h2_build_dataset_views.py) — Construct and chain dataset view
#   transformations.
