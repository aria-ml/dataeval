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
# # Compare surround contrast, clipping and noise between two collections
#
# A statistic compares across collections only when both were measured the same way. This guide measures a reference
# collection and an operational one with one set of {func}`.image_stats` settings, checks the recorded settings, and
# compares surround contrast, clipping and noise.
#
# Use it when you compare footage from a new deployment with the footage a model was trained on.
#
# You need `dataeval` and `maite-datasets[hf]>=0.0.25`. The first run downloads the 1.7 GB validation videos of
# VisDrone2019-MOT.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval "maite-datasets[hf]>=0.0.25"
except Exception:
    pass

# %%
import warnings

import numpy as np
import polars as pl
from IPython.display import display
from maite_datasets.multiobject_tracking import VisDroneMOT

from dataeval import Metadata
from dataeval.config import set_max_processes
from dataeval.core import image_stats
from dataeval.data import EvenlySpaced, Indices, SequenceFrames, View
from dataeval.flags import ImageStats
from dataeval.utils.preprocessing import ChannelGroup

set_max_processes(4)
# Reading rows bins the numeric factors, which this guide does not use.
warnings.filterwarnings("ignore", message=".*binned automatically")

# %% [markdown]
# The videos are VisDrone2019-MOT ([Zhu et al., 2021](https://arxiv.org/abs/2001.06303)), shared under
# [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). The hidden cell stands in for your own
# collections: 60 evenly spaced frames of four daytime road videos are the reference, and 60 of the night video are the
# operational collection.

# %% tags=["hide-input"]
dataset = VisDroneMOT(root="./data", image_set="val", download=True)
videos = Metadata(dataset).rows_at("sequence")
day = videos.filter((pl.col("time_of_day") == "daytime") & (pl.col("scene_type") != "sporting event"))
night = videos.filter(pl.col("time_of_day") == "night")

collections = {
    "reference": SequenceFrames(View(dataset, Indices(day["item_index"].to_list())), EvenlySpaced(15)),
    "operational": SequenceFrames(View(dataset, Indices(night["item_index"].to_list())), EvenlySpaced(60)),
}
for name, frames in collections.items():
    print(f"{name:12} {len(frames)} frames")

# %% [markdown]
# ## Measure both collections with the same settings
#
# Put every setting that changes a value in one dictionary and pass it to both calls. Declare `value_range` even for
# 8-bit data, since an inferred range does not equal a declared one.

# %%
settings = {
    "stats": ImageStats.CONTEXT | ImageStats.PIXEL_SATURATED_FRACTION | ImageStats.NOISE,
    "value_range": (0, 255),
}
results = {name: image_stats(frames, **settings) for name, frames in collections.items()}
reference, operational = results["reference"], results["operational"]

print("settings match:", reference.get("measurement_specs", {}) == operational.get("measurement_specs", {}))
print(reference.get("measurement_specs", {})["target_surround_contrast"])

# %% [markdown]
# `measurement_specs` records how each statistic that depends on a setting was measured. Compare the specs before the
# values. The noise statistics record no spec although `normalize_pixel_values` changes them, so only the shared
# dictionary keeps them comparable.

# %% [markdown]
# ## Compare surround contrast on shared bin edges
#
# Hand each result to {meth}`.Metadata.add_factors` and keep the vehicle boxes. A box whose surround is too small to
# read is NaN, so check the share with a value.

# %%
metadata = {}
for name, frames in collections.items():
    metadata[name] = Metadata(frames)
    metadata[name].add_factors(results[name])

vehicles = [i for i, label in dataset.index2label.items() if label in ("car", "van", "truck", "bus")]
boxes = pl.concat(
    metadata[name]
    .rows_at("instance")
    .filter(pl.col("class_label").is_in(vehicles))
    .select(
        pl.lit(name).alias("collection"),
        pl.col("instance_target_surround_contrast").fill_nan(None).alias("contrast"),
        pl.col("instance_signal_clutter_ratio").fill_nan(None).alias("scr"),
    )
    for name in collections
)

display(
    boxes.group_by("collection", maintain_order=True).agg(
        pl.len().alias("vehicle boxes"),
        pl.col("contrast").is_not_null().mean().round(4).alias("share with a value"),
        pl.col("contrast").median().round(3).alias("median contrast"),
        pl.col("scr").median().round(3).alias("median scr"),
    )
)

# %% [markdown]
# Shares per bin compare only on the same edges. Declare them once, with open outer edges, and with 0 between boxes
# darker and brighter than their surround.

# %%
edges = {
    "contrast": [-np.inf, -0.4, -0.2, 0.0, 0.2, 0.4, np.inf],
    "scr": [-np.inf, -2.0, -1.0, 0.0, 1.0, 2.0, np.inf],
}
tables = {}
for column, cuts in edges.items():
    shares = {}
    for name in collections:
        values = boxes.filter(pl.col("collection") == name)[column].drop_nulls().to_numpy()
        shares[name] = np.round(np.histogram(values, cuts)[0] / len(values), 3)
    bins = [f"{low} to {high}" for low, high in zip(cuts[:-1], cuts[1:], strict=True)]
    tables[column] = pl.DataFrame({f"{column} bin": bins, **shares})
    display(tables[column])

# %% [markdown]
# All but one daytime box has a value. At night the median contrast is about 0.12 against 0.03 by day, and 18% of the
# boxes read above 0.4 against 6%. The signal-to-clutter ratio, which divides by the surround's spread, has a higher
# median at night too, about 0.34 against 0.17, but 16% of the daytime boxes lie beyond 2 in either direction against 7%
# at night. The night collection is one flight, and frames of one video are not independent, so this describes these
# frames and not night footage in general. A ratio compares stored intensities and does not establish that a vehicle is
# easier to see, as [The target-surround ratios](../concepts/ImageStatistics.md#the-target-surround-ratios) explains.
# The six bins are a display choice.

# %% [markdown]
# ## Compare clipping and noise per video
#
# Clipping and noise belong to the frame, so read them on frame rows, grouped by video.

# %%
per_video = pl.concat(
    metadata[name]
    .rows_at("unit")
    .group_by("source_id")
    .agg(
        pl.len().alias("frames"),
        (pl.col("unit_saturated_fraction") > 0.01).mean().alias("share over 1% clipped"),
        pl.col("unit_saturated_fraction").median().round(4).alias("median clipped"),
        pl.col("unit_noise_std").median().round(2).alias("median noise_std"),
    )
    .sort("source_id")
    .select(pl.lit(name).alias("collection"), pl.all())
    for name in collections
)
display(per_video)
for name, result in results.items():
    print(f"{name:12} clipping endpoint: {np.unique(np.asarray(result.get('details', {})['saturated_upper']))}")

# %% [markdown]
# Every frame of one daytime video has more than 1% of its values at the endpoint, 255, and every other video stays
# below 0.1%, so the reference's clipping is that one video's.
#
# The noise medians differ more among the daytime videos than between day and night. The videos show different scenes,
# and a noise reading compares only images of the same content
# ([Noise statistics](../concepts/ImageStatistics.md#noise-statistics)), so this does not show that the night camera is
# less noisy.

# %% [markdown]
# ## Declare the range of data beyond 8 bits
#
# Data in a wider type, such as a 12-bit sensor's output in `uint16`, has no range that DataEval can assume. The cell
# below rescales the operational frames to 12 bits to simulate one.

# %%
night_frames = [collections["operational"][i] for i in range(len(collections["operational"]))]
twelve_bit = [np.round(np.asarray(image, np.float64) * 4095 / 255).astype(np.uint16) for image, _, _ in night_frames]
twelve_bit_boxes = [np.asarray(target.boxes) for _, target, _ in night_frames]
ratios_and_clipping = (
    ImageStats.CONTEXT_TARGET_SURROUND_CONTRAST
    | ImageStats.CONTEXT_SIGNAL_CLUTTER_RATIO
    | ImageStats.PIXEL_SATURATED_FRACTION
)

undeclared = image_stats(twelve_bit, boxes=twelve_bit_boxes, stats=ratios_and_clipping)
print("context statuses:", sorted(set(np.asarray(undeclared.get("details", {})["context_status"]).tolist())))
print("clipping endpoint:", np.unique(np.asarray(undeclared.get("details", {})["saturated_upper"])))

# %% [markdown]
# Every box reads NaN with the status `unknown_range`. Declare the range on a {class}`.ChannelGroup`, which names the
# bands it covers.

# %%
declared = image_stats(
    twelve_bit,
    boxes=twelve_bit_boxes,
    stats={"rgb": ratios_and_clipping},
    channels={"rgb": ChannelGroup([0, 1, 2], value_range=(0, 4095))},
)
eight_bit = operational["stats"]
contrast_gap = np.nanmax(
    np.abs(np.asarray(declared["stats"]["rgb_target_surround_contrast"]) - eight_bit["target_surround_contrast"])
)
print(f"largest contrast difference from the 8-bit frames: {contrast_gap:.4f}")
print(
    "same clipped shares:",
    np.array_equal(declared["stats"]["rgb_saturated_fraction"], eight_bit["saturated_fraction"], equal_nan=True),
)
print("clipping endpoint:", np.unique(np.asarray(declared.get("details", {})["rgb_saturated_upper"])))
print("declared range:", declared.get("measurement_specs", {})["rgb_saturated_fraction"].declared_range)

# %% [markdown]
# Declared, the contrast differs from the 8-bit values by at most about 0.002, the rounding of the copy, the clipped
# shares match and the endpoint reads 4095. The spec records the declared range, so it does not equal the 8-bit spec
# although the values agree. Float data and thermal or multispectral bands need one too: see
# [Where the reference comes from](../concepts/ImageStatistics.md#where-the-reference-comes-from) and
# [Measure named groups of channels](./h2_measure_channel_groups.py).

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert {name: len(frames) for name, frames in collections.items()} == {"reference": 60, "operational": 60}
assert reference.get("measurement_specs", {}) == operational.get("measurement_specs", {})
assert "noise_std" not in reference.get("measurement_specs", {})
inferred = image_stats(collections["operational"], stats=ImageStats.PIXEL_SATURATED_FRACTION)
assert (
    inferred.get("measurement_specs", {})["saturated_fraction"]
    != operational.get("measurement_specs", {})["saturated_fraction"]
)
assert np.array_equal(inferred["stats"]["saturated_fraction"], eight_bit["saturated_fraction"], equal_nan=True)

summary = {
    row["collection"]: row
    for row in boxes
    .group_by("collection")
    .agg(
        pl.len().alias("n"),
        pl.col("contrast").null_count().alias("missing"),
        pl.col("contrast").median(),
        pl.col("scr").median(),
    )
    .iter_rows(named=True)
}
assert summary["reference"]["n"] == 1347
assert summary["operational"]["n"] == 939
assert summary["reference"]["missing"] == 1
assert summary["operational"]["missing"] == 0
assert 0.02 < summary["reference"]["contrast"] < 0.05
assert 0.10 < summary["operational"]["contrast"] < 0.15
assert 0.14 < summary["reference"]["scr"] < 0.20
assert 0.30 < summary["operational"]["scr"] < 0.38

contrast_shares, scr_shares = tables["contrast"], tables["scr"]
assert 0.14 < contrast_shares["operational"][-1] < 0.22
assert 0.03 < contrast_shares["reference"][-1] < 0.09
day_wide = scr_shares["reference"][0] + scr_shares["reference"][-1]
night_wide = scr_shares["operational"][0] + scr_shares["operational"][-1]
assert 0.12 < day_wide < 0.20
assert 0.04 < night_wide < 0.10

clipped = per_video.filter(pl.col("share over 1% clipped") > 0)
assert clipped["collection"].to_list() == ["reference"]
assert clipped["share over 1% clipped"].to_list() == [1.0]
frame_rows = pl.concat(
    metadata[name].rows_at("unit").select("source_id", "unit_saturated_fraction") for name in collections
)
assert frame_rows.filter(pl.col("source_id") != clipped["source_id"][0])["unit_saturated_fraction"].max() < 0.001
for result in results.values():
    assert np.unique(np.asarray(result.get("details", {})["saturated_upper"])).tolist() == [255]
day_noise = per_video.filter(pl.col("collection") == "reference")["median noise_std"]
night_noise = per_video.filter(pl.col("collection") == "operational")["median noise_std"][0]
assert day_noise.max() - day_noise.min() > day_noise.min() - night_noise > 0

statuses = np.asarray(undeclared.get("details", {})["context_status"])
assert set(statuses.tolist()) == {"not_applicable", "unknown_range"}
assert (statuses == "unknown_range").sum() == sum(len(b) for b in twelve_bit_boxes)
assert np.isnan(np.asarray(undeclared["stats"]["target_surround_contrast"])).all()
assert np.isnan(np.asarray(undeclared.get("details", {})["saturated_upper"])).all()
assert contrast_gap < 0.005
assert np.array_equal(declared["stats"]["rgb_saturated_fraction"], eight_bit["saturated_fraction"], equal_nan=True)
assert np.unique(np.asarray(declared.get("details", {})["rgb_saturated_upper"])).tolist() == [4095]
assert declared.get("measurement_specs", {})["rgb_saturated_fraction"].declared_range == (0, 4095)
assert (
    declared.get("measurement_specs", {})["rgb_saturated_fraction"]
    != operational.get("measurement_specs", {})["saturated_fraction"]
)

# %% [markdown]
# ## Next steps
#
# - [Image statistics](../concepts/ImageStatistics.md) — Understand what each statistic measures and which interval it
#   reads against.
# - [Compare objects to their surroundings](./tt_compare_objects_to_surroundings.py) — Learn to read box statistics
#   beside the statistics of the region around them.
# - [Control and reuse a factor's binning](./h2_control_factor_binning.py) — Fix bin edges once and reuse them across
#   collections.
