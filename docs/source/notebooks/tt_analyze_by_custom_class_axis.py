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
# (custom-class-axis)=
# # Analyze bias against a custom class axis
#
# This tutorial shows you how to condition DataEval's bias evaluators on an operational factor and not on the object
# classes. You work with SeaDrone, aerial imagery of people and boats in open water that three different drones
# captured, and you separate what belongs to the object class from what belongs to the platform.
#
# Estimated time to complete: 15 minutes
#
# Relevant ML stages: [Data Engineering](../getting-started/roles/ML_Lifecycle.md#data-engineering)
#
# Relevant personas: [Data Engineer](../getting-started/roles/data_engineer.md), [T&E Engineer](../getting-started/roles/te_engineer.md)
#
# ## What you will do
#
# - Load SeaDrone and read the class mix and the flight conditions of each drone
# - Declare the missing readings of altitude and camera pitch, choose the bins of the continuous factors, and check that
#   the bins matter
# - Pivot the class axis to the drone and evaluate `Balance` against it
# - Disentangle class-level findings from platform-level findings
# - Evaluate factors at the image level and not the detection level
# - Combine factors into a composite axis to find combinations that were never collected
# - Test the observed combinations with `Parity` and `Diversity`
# - Check which class axis a result used
# - Stratify dataset splits across drones
#
# ## What you will learn
#
# - How a custom class axis changes what the class-conditional evaluators report
# - How to tell whether a class or an operational factor drives an association
# - How the image-level and detection-level views weight an image
# - Which combinations of conditions the data never covers
#
# ## What you will need
#
# - A Python environment with `dataeval` and `maite-datasets` installed
# - No images: `Metadata` reads only the annotations and the metadata of each frame

# %% [markdown]
# ## Introduction
#
# DataEval evaluators, including {class}`.Balance`, {class}`.Diversity`, {class}`.Parity`, {class}`.Coverage`, and
# {func}`.split_dataset`, condition on dataset class labels by default. In many datasets, bias and underrepresentation
# are driven not by object classes, but by environmental or operational factors such as weather, sensor type, platform,
# time of day, or collection site.
#
# {meth}`.Metadata.classed_by` allows you to pivot the class axis to one or more
# metadata factors. The target factor's discrete levels become the new class labels,
# and all class-conditional evaluators condition on them instead. The dataset's
# original class labels are retained as a factor named `class`.
#
# You should use a custom class axis when:
#
# - You need to evaluate balance, diversity, or parity across environmental or operational conditions (such as weather,
#   site, or sensor platform).
# - A class and an environmental factor co-occur, and you need to determine which one drives an observed imbalance.
# - You want to identify uncollected or undersampled combinations of conditions across multiple metadata factors.
# - You need to perform bias analysis at the image or video level on multi-object datasets.
# - You want to stratify dataset splits across operational metadata rather than class labels.

# %% [markdown]
# ## Setup
#
# First, import the required libraries to set up the example.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval maite-datasets
except Exception:
    pass

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# %%
from typing import Literal

import numpy as np
import polars as pl
from maite_datasets.object_detection import SeaDrone

from dataeval import Metadata
from dataeval.bias import Balance, Diversity, Parity
from dataeval.data import split_dataset
from dataeval.types import Remap

# %% [markdown]
# ## Load the dataset
#
# Load the validation split of SeaDrone with `SeaDrone` from `maite_datasets`. It holds 1,547 frames of swimmers, boats,
# jet skis, life-saving appliances and buoys, and three drones captured them: the `mavic`, the `m210` and the `trinity`.
# Load it with `lazy=True` so that it reads the annotations and the metadata of each frame without the images.
#
# Build a {class}`.Metadata` with its defaults first, and read the flight conditions of each drone from its
# `dataframe`. A frame is a `unit` row, and a detection is an `instance` row.

# %%
dataset = SeaDrone("./data", image_set="val", download=True, lazy=True)
raw = Metadata(dataset)
rows = raw.dataframe
frames = rows.filter(pl.col("level") == "unit")

print(
    frames
    .group_by("drone")
    .agg(
        pl.len().alias("frames"),
        (pl.col("altitude") > 0).mean().round(2).alias("altitude recorded"),
        pl.col("altitude").filter(pl.col("altitude") > 0).median().round(0).alias("median altitude (m)"),
        (pl.col("gimbal_pitch") >= 0).mean().round(2).alias("pitch recorded"),
        pl.col("gimbal_pitch").filter(pl.col("gimbal_pitch") >= 0).median().round(0).alias("median pitch (deg)"),
    )
    .sort("frames", descending=True)
)

# %% [markdown]
# The drones fly different missions. The `mavic` flies at about 37 m, the `m210` at about 9 m, and the `trinity` at
# about 110 m. The camera points at the horizon at a low pitch and straight down at 90 degrees, and its pitch differs by
# drone as well. Two readings are patchy: the `m210` and the `trinity` record an altitude and a camera pitch on about a
# quarter of their frames, and a missing reading is stored as `-1`.
#
# The class mix differs by drone too. Each row below is one drone, and each column is the share of its detections that
# have a class:

# %%
detections = rows.filter(pl.col("level") == "instance").with_columns(
    pl.col("class_label").replace_strict(raw.index2label, return_dtype=pl.String).alias("class")
)
mix = detections.group_by("drone", "class").len().pivot(on="class", index="drone", values="len").fill_null(0)
totals = mix.select(pl.exclude("drone")).sum_horizontal()
print(mix.with_columns((pl.exclude("drone") * 100 / totals).round(0)))

# %% [markdown]
# Swimmers make up 93% of the detections of the `m210` and 46% of the detections of the `trinity`, and the `trinity`
# sees as many boats as swimmers. So a class and a drone co-occur, and a drone and an altitude co-occur. The question
# for a data engineer is which of the two, the class or the platform, an imbalance belongs to.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
by_drone_frames = {
    row["drone"]: row
    for row in frames
    .group_by("drone")
    .agg(
        pl.len().alias("frames"),
        (pl.col("altitude") > 0).mean().alias("recorded"),
        pl.col("altitude").filter(pl.col("altitude") > 0).median().alias("median"),
    )
    .iter_rows(named=True)
}
assert (len(frames), len(detections)) == (1547, 9630)
assert {name: row["frames"] for name, row in by_drone_frames.items()} == {"mavic": 937, "m210": 351, "trinity": 259}
assert by_drone_frames["mavic"]["recorded"] == 1.0
assert all(by_drone_frames[d]["recorded"] < 0.3 for d in ("m210", "trinity"))
assert (
    by_drone_frames["m210"]["median"]
    < 15
    < 30
    < by_drone_frames["mavic"]["median"]
    < 50
    < 100
    < by_drone_frames["trinity"]["median"]
)
shares = {row["drone"]: row for row in mix.iter_rows(named=True)}
swimmer_share = {
    drone: row["swimmer"] / sum(v for k, v in row.items() if k != "drone") for drone, row in shares.items()
}
boat_share = {drone: row["boat"] / sum(v for k, v in row.items() if k != "drone") for drone, row in shares.items()}
assert swimmer_share["m210"] > 0.9
assert 0.4 < swimmer_share["trinity"] < 0.5
assert boat_share["trinity"] > 0.4
assert boat_share["m210"] < 0.1

# %% [markdown]
# ## Declare the missing readings and choose the bins
#
# `Metadata` keeps four factors for this analysis: the `drone`, the `altitude`, the `gimbal_pitch` of the camera, and
# the `object_size` of each detection. {class}`.Balance`, {class}`.Diversity` and {class}`.Parity` do not read raw
# numbers. They read the levels that binning cuts a continuous factor into, so the bins decide what the evaluators can
# see. Three choices matter here, and the next section measures each of them:
#
# - A {class}`.Remap` correction turns every value below zero into a missing reading, so that `-1` stops counting as an
#   altitude or a pitch.
# - Altitude and pitch use bin edges that you choose, because the altitudes of the drones sit in known bands and the
#   camera is either pitched at an angle or pointed straight down.
# - Object size is skewed, with buoys far smaller than boats, so equal-width bins would put almost every detection in
#   the first bin. Leave it out of `continuous_factor_bins` and set `auto_bin_method="uniform_count"`, which gives every
#   bin the same number of detections.
#
# `partial_factors=True` keeps a frame with a missing reading and puts it in a `missing` level.

# %%
EXCLUDE = [
    "compass_heading", "frame", "gimbal_heading", "height", "object_id", "speed",
    "storage", "width", "xspeed", "yspeed", "zspeed",
]  # fmt: skip
CORRECTIONS = [Remap("altitude", {(None, 0): None}), Remap("gimbal_pitch", {(None, 0): None})]
ALTITUDE_EDGES = [0, 20, 60, 150, 300]
PITCH_EDGES = [0, 45, 80, 91]

metadata = Metadata(
    dataset,
    exclude=EXCLUDE,
    corrections=CORRECTIONS,
    continuous_factor_bins={"altitude": ALTITUDE_EDGES, "gimbal_pitch": PITCH_EDGES},
    auto_bin_method="uniform_count",
    partial_factors=True,
)

print(f"images:     {metadata.level_counts['unit']}")
print(f"detections: {metadata.level_counts['instance']}")
print(f"factors:    {list(metadata.factor_names)}")
print(f"class axis: {metadata.class_axis} ({metadata.class_axis_source})")

# %% [markdown]
# ### Check that the binning matters
#
# Measure how much the choices change what `Balance` reports. The `associations` function builds a `Metadata` with one
# binning and returns three mutual information scores, the altitude and the camera pitch against the drone and the
# object size against the class, together with the number of cells that a drone and altitude axis would have. Run it on
# six binnings:
#
# - The sentinel left in, and the sentinel declared missing, each with equal-width bins.
# - Your edges with an equal-width object size, and with an equal-count object size, which is the binning used above.
# - A coarser and a finer binning, to see whether the conclusions survive a reasonable change.

# %%
sizes = detections["object_size"].drop_nulls().to_numpy()
print(f"object size in pixels: median {np.median(sizes):,.0f}, largest {sizes.max():,.0f}")


def quantile_edges(count):
    """Edges that cut the object sizes into bins with the same number of detections."""
    edges = np.quantile(sizes, np.linspace(0, 1, count + 1))
    return [float(edge) for edge in np.unique([0, *edges[1:-1], edges[-1] + 1])]


def associations(bins, method: Literal["uniform_width", "uniform_count"] = "uniform_width", corrections=CORRECTIONS):
    """Score the altitude and pitch against the drone, and the object size against the class, under one binning."""
    candidate = Metadata(
        dataset,
        exclude=EXCLUDE,
        corrections=corrections,
        continuous_factor_bins=bins,
        auto_bin_method=method,
        partial_factors=True,
    )
    by_class = dict(Balance().evaluate(candidate).balance.iter_rows())
    by_platform = dict(Balance().evaluate(candidate.classed_by("drone")).balance.iter_rows())
    return {
        "altitude ~ drone": by_platform["altitude"],
        "pitch ~ drone": by_platform["gimbal_pitch"],
        "size ~ class": by_class["object_size"],
        "drone × altitude cells": len(candidate.at("unit").classed_by("drone", "altitude").index2label),
    }


equal_width = {"altitude": 4, "gimbal_pitch": 4, "object_size": 4}
chosen = {"altitude": ALTITUDE_EDGES, "gimbal_pitch": PITCH_EDGES}
binnings = {
    "Sentinel left in, equal-width bins": associations(equal_width, corrections=None),
    "Sentinel declared missing, equal-width bins": associations(equal_width),
    "Your edges, equal-width object size": associations(chosen | {"object_size": 4}),
    "Your edges, equal-count object size (used)": associations(chosen, "uniform_count"),
    "Coarser edges and bins": associations({
        "altitude": [0, 30, 100, 300],
        "gimbal_pitch": [0, 60, 91],
        "object_size": quantile_edges(3),
    }),
    "Finer edges and bins": associations({
        "altitude": [0, 10, 20, 40, 60, 100, 150, 300],
        "gimbal_pitch": [0, 30, 45, 60, 80, 91],
        "object_size": quantile_edges(8),
    }),
}
with pl.Config(tbl_rows=10, tbl_width_chars=140, fmt_str_lengths=50):
    print(
        pl.DataFrame([{"binning": name, **scores} for name, scores in binnings.items()]).with_columns(
            pl.col(pl.Float64).round(2)
        )
    )

# %% [markdown]
# The table shows why you choose the bins:
#
# - Leaving `-1` in puts the missing frames in the lowest altitude bin beside real low flights, and it stretches the
#   range of the factor. The association of altitude with the drone falls from about 0.62 to about 0.05, and the
#   association of pitch from about 0.61 to about 0.35, although nothing changed in the flights.
# - Equal-width bins cannot describe a skewed factor. The largest object is hundreds of times the median, so nearly
#   every detection lands in the first bin, and the association of object size with the class reads 0.00, where
#   equal-count bins give about 0.30. An association that is there goes unseen.
# - The number of bins changes the score and the number of cells. The association of object size with the class rises
#   from about 0.18 to about 0.26 as the equal-count bins go from 3 to 8. The drone and altitude axis grows from 9 to 16
#   cells as the altitude bands go from 3 to 7, which leaves more cells too thin for `Parity` to test.
#
# The altitude and pitch findings survive every sensible choice, at 0.61 to 0.65. The object size finding survives
# every equal-count binning and fails with equal-width ones. State the bins with a result, and check that the
# conclusion holds under a coarser and a finer binning, as above. [Control and reuse a factor's
# binning](./h2_control_factor_binning.py) shows how to declare and reuse bin edges across collections.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
scores = pl.DataFrame([{"binning": name, **row} for name, row in binnings.items()])
left_in, declared, wide, used, coarse, fine = (row for row in scores.iter_rows(named=True))
assert left_in["altitude ~ drone"] < 0.15
assert declared["altitude ~ drone"] > 0.5
assert left_in["pitch ~ drone"] < 0.45 < 0.55 < declared["pitch ~ drone"]
assert wide["size ~ class"] < 0.05
assert 0.25 < used["size ~ class"] < 0.35
assert sizes.max() > 100 * np.median(sizes)
assert all(
    0.55 < row["altitude ~ drone"] < 0.7 and 0.55 < row["pitch ~ drone"] < 0.7
    for row in (declared, wide, used, coarse, fine)
)
assert 0.1 < coarse["size ~ class"] < used["size ~ class"]
assert 0.2 < fine["size ~ class"] < 0.3
assert coarse["drone × altitude cells"] < used["drone × altitude cells"] < fine["drone × altitude cells"]
assert (used["drone × altitude cells"], fine["drone × altitude cells"], coarse["drone × altitude cells"]) == (11, 16, 9)
assert (
    used["altitude ~ drone"] == dict(Balance().evaluate(metadata.classed_by("drone")).balance.iter_rows())["altitude"]
)

# %% [markdown]
# ## Pivot to a single metadata factor
#
# Call {meth}`.Metadata.classed_by` to create a copy of the metadata configured with a new class axis. The original
# metadata object remains unchanged.

# %%
by_drone = metadata.classed_by("drone")

print(f"axis:     {by_drone.class_axis} ({by_drone.class_axis_source})")
print(f"groups:   {dict(by_drone.index2label)}")
print(f"original: {metadata.class_axis} ({metadata.class_axis_source})")

# %% [markdown]
# When you pivot to a factor:
#
# 1. `class_labels` contains the integer codes of the new axis (`drone`) rather than the original class labels.
# 2. `index2label` maps those codes to group names (`m210`, `mavic`, `trinity`).
# 3. The pivoted factor (`drone`) is removed from `factor_names` to prevent self-correlation, and the original label
#    set is added as a factor named `class`.

# %%
print(f"before: {list(metadata.factor_names)}")
print(f"after:  {list(by_drone.factor_names)}")

# %% [markdown]
# ## Evaluate balance across the custom axis
#
# Evaluate {class}`.Balance` first on the original class axis. Each row is a factor, and `mi_value` is its normalized
# mutual information with the class:

# %%
default = Balance().evaluate(metadata)
print(default.balance)

# %% [markdown]
# Seen from the class, only `object_size` stands out, at about 0.30. The `altitude` and the `gimbal_pitch` show almost
# nothing. Now evaluate {class}`.Balance` against the `drone` axis. Because the original class labels are now a factor
# named `class`, you can also measure how much the drone correlates with the object classes present.

# %%
result = Balance().evaluate(by_drone)
print(result.balance)

# %% [markdown]
# Seen from the drone, the picture reverses. The `altitude` and the `gimbal_pitch` show strong mutual information with
# the `drone`, at about 0.62 and 0.65, while the `class` and the `object_size` show very little. Each factor belongs to
# one axis:
#
# - Object size follows the class. Boats are far larger than swimmers in an image, whichever drone flew.
# - Altitude and camera pitch follow the drone. Each drone has its own mission.
#
# The class view could not see the platform effect, because every class occurs on at least two of the drones.

# %% [markdown]
# ## Disentangle class and metadata associations
#
# While overall balance scores indicate that an association exists, `classwise` metrics reveal which specific groups
# drive the relationship.
#
# Inspect the `altitude` rows from the default class-conditional evaluation:

# %%
print(default.classwise.filter(pl.col("factor_name") == "altitude"))

# %% [markdown]
# In the default class-conditional view, no class is flagged as imbalanced against altitude: each class is spread over
# the altitudes that the three drones flew. Now inspect the same factor under the `drone` class axis:

# %%
print(result.classwise.filter(pl.col("factor_name") == "altitude"))

# %% [markdown]
# Under the `drone` axis, every drone is flagged, and the `mavic` scores highest at about 0.81: within this dataset,
# the drone nearly determines the altitude band of a frame.
#
# The class-conditional view can only attribute variation to class labels. When a class co-occurs with an operational
# condition, an effect of that condition is reported as a class imbalance or is not seen at all. Pivoting to a metadata
# axis allows you to isolate whether an association originates from the class label or from the operational factor.
# The `object_size` rows show the opposite case, an association that does belong to the class:

# %%
print(default.classwise.filter(pl.col("factor_name") == "object_size"))

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert (metadata.level_counts["unit"], metadata.level_counts["instance"]) == (1547, 9630)
assert sorted(metadata.factor_names) == ["altitude", "drone", "gimbal_pitch", "object_size"]
assert by_drone.index2label == {0: "m210", 1: "mavic", 2: "trinity"}
assert "drone" not in by_drone.factor_names
assert "class" in by_drone.factor_names
by_class = dict(default.balance.iter_rows())
by_platform = dict(result.balance.iter_rows())
assert by_class["object_size"] > 0.25
assert by_class["altitude"] < 0.1
assert by_class["gimbal_pitch"] < 0.1
assert by_platform["altitude"] > 0.5
assert by_platform["gimbal_pitch"] > 0.5
assert by_platform["class"] < 0.15
assert by_platform["object_size"] < 0.1
assert not default.classwise.filter(pl.col("factor_name") == "altitude")["is_imbalanced"].any()
drone_altitude = {
    row["class_name"]: row for row in result.classwise.filter(pl.col("factor_name") == "altitude").iter_rows(named=True)
}
assert all(row["is_imbalanced"] for row in drone_altitude.values())
assert drone_altitude["mavic"]["mi_value"] > 0.7
size_rows = {
    row["class_name"]: row
    for row in default.classwise.filter(pl.col("factor_name") == "object_size").iter_rows(named=True)
}
assert size_rows["boat"]["is_imbalanced"]
assert size_rows["buoy"]["is_imbalanced"]

# %% [markdown]
# ## Evaluate metadata at the image level with at()
#
# In object detection datasets, class labels exist at the detection (`instance`) level, while factors like `drone` exist
# at the image (`unit`) level. Evaluating the default metadata at the unit level fails because an image containing
# multiple detections does not have a single class label:

# %%
try:
    Balance().evaluate(metadata.at("unit"))
except ValueError as error:
    print(f"class axis: {error}")

# %% [markdown]
# Because `drone` is defined for every image, pivoting to `drone` allows you to evaluate bias at the unit level
# directly:

# %%
by_drone_per_image = metadata.at("unit").classed_by("drone")
print(Balance().evaluate(by_drone_per_image).balance)

# %% [markdown]
# You can inspect {attr}`.ClassAxis.rows_per_group_entity` to verify the fan-out ratio between rows and entities:
#
# - In the detection view (`instance`), image-level factors are replicated across detections (fan-out > 1.0), weighting
#   crowded images more heavily.
# - In the image view (`unit`), each image is counted once (fan-out = 1.0).

# %%
print(
    f"detections view: {by_drone.class_labels.shape[0]} rows, fan-out {by_drone.class_axis_info.rows_per_group_entity:.2f}"
)
print(
    f"images view:     {by_drone_per_image.class_labels.shape[0]} rows, "
    f"fan-out {by_drone_per_image.class_axis_info.rows_per_group_entity:.2f}"
)

# %% [markdown]
# The two views weight an image differently, so their scores differ. The associations of `altitude` and `gimbal_pitch`
# with the drone are lower in the image view, at about 0.50 and 0.56, than in the detection view, at about 0.62 and
# 0.65.
#
# Note that `class` and `object_size` are absent when pivoting at the unit level, as multi-detection images have no
# single class label and no single object size. If you need a unit-level class summary, you can aggregate labels with
# {meth}`.Metadata.agg`.

# %%
print(f"detections view factors: {list(by_drone.factor_names)}")
print(f"images view factors:     {list(by_drone_per_image.factor_names)}")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
unit_balance = dict(Balance().evaluate(by_drone_per_image).balance.iter_rows())
assert unit_balance["altitude"] > 0.4
assert unit_balance["gimbal_pitch"] > 0.4
assert unit_balance["altitude"] < by_platform["altitude"]
assert unit_balance["gimbal_pitch"] < by_platform["gimbal_pitch"]
assert by_drone.class_labels.shape[0] == 9630
assert by_drone_per_image.class_labels.shape[0] == 1547
assert 6.2 < (by_drone.class_axis_info.rows_per_group_entity or 0) < 6.25
assert by_drone_per_image.class_axis_info.rows_per_group_entity == 1.0
assert "class" in by_drone.factor_names
assert "class" not in by_drone_per_image.factor_names

# %% [markdown]
# ## Create composite axes to identify missing combinations
#
# You can pass multiple factor names to {meth}`.Metadata.classed_by` to construct a composite class axis. The resulting
# groups represent all observed combinations, formatted as `factor1 × factor2`.
#
# By crossing factors, you can inspect which combinations exist in your data and which are missing:

# %%
cells = metadata.at("unit").classed_by("drone", "altitude")

print(f"axis: {cells.class_axis}")
counts = np.bincount(cells.class_labels, minlength=len(cells.index2label))
for code, name in sorted(cells.index2label.items(), key=lambda item: item[1]):
    print(f"  {name:24s} {counts[code]:4d} images")

bands = ["[0, 20)", "[20, 60)", "[60, 150)", "[150, 300)", "missing"]
possible = [f"{drone} × {band}" for drone in ("m210", "mavic", "trinity") for band in bands]
absent = [name for name in possible if name not in cells.index2label.values()]
print(f"\ncells present: {len(cells.index2label)} of {len(possible)} possible")
print(f"cells absent:  {absent}")

# %% [markdown]
# Only 11 of the 15 possible combinations are present. Nobody flew the `mavic` or the `m210` above 150 m, nobody flew
# the `trinity` below 20 m, and the `mavic` never lacks an altitude reading. This reveals specific operational gaps for
# future data collection:
#
# - The data gap is a specific factor combination (`mavic × high altitude`) rather than a missing class.
# - As long as the high-altitude cells hold only the `trinity`, model performance at high altitude cannot be separated
#   from model performance on that platform.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(cells.index2label) == 11
assert len(possible) == 15
assert sorted(absent) == ["m210 × [150, 300)", "mavic × [150, 300)", "mavic × missing", "trinity × [0, 20)"]
assert min(counts) >= 11

# %% [markdown]
# ## Test the observed combinations with Parity and Diversity
#
# The listing above is how you find absent combinations. {class}`.Parity` cannot: it tests the combinations that occur
# in the data, so a combination with no images has no cell to test. Parity answers a different question about the
# combinations that do exist. Once `drone` and `altitude` form the class axis, the remaining image-level factor is
# `gimbal_pitch`. Parity tests whether the camera pitch is spread across the cells as independence predicts, and which
# cells hold too few images to judge.
#
# The frames that lack an altitude or a pitch would form cells that no one flew, so keep the frames with both readings
# with {meth}`.Metadata.where`:

# %%
recorded = metadata.at("unit").where(pl.col("altitude").is_not_null() & pl.col("gimbal_pitch").is_not_null())
recorded_cells = recorded.classed_by("drone", "altitude")
parity = Parity().evaluate(recorded_cells)
print(parity.factors.select("factor_name", "status", "score", "p_value_adjusted", "has_insufficient_data"))
print(f"\ngrouping: {parity.group_source} ({parity.n_rows} rows, {parity.n_groups} groups, {parity.test} test)")
print(parity.cells["status"].value_counts().sort("status"))

findings = parity.cells.filter(pl.col("is_actionable")).sort("p_value_adjusted").head(8)
print(findings.select("level", "class_name", "status", "observed", "expected", "p_value_adjusted"))

# %% [markdown]
# Each row of {attr}`.ParityOutput.cells` is one pitch level in one drone and altitude cell, with its `status`:
#
# - The factor test is `associated`: the camera pitch is not independent of the drone and the altitude, with a score of
#   about 0.70 and an adjusted p-value that is effectively zero. The factor also carries `has_insufficient_data`,
#   because part of its table is too thin to count.
# - Several cells are `overrepresented` or `underrepresented`, and the strongest rows show how the missions differ. The
#   `m210` below 20 m flies with the camera pitched at 45 to 80 degrees far more often than independence predicts, and
#   the `mavic` at 60 to 150 m looks in that range more often too, while the `mavic` rarely points straight down.
# - Nine cells are `too_thin`: they hold too few images to judge, whatever their p-values. Collect more images there to
#   find out.
# - The images are the independent unit here, so `group_source` is `rows` and the ungrouped Pearson test runs. On a
#   detection-level view, Parity groups detections by image instead.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
# Lock the claims made in the prose above about the Parity cells.
assert parity.factors["status"].to_list() == ["associated"]
assert 0.6 < parity.factors["score"][0] < 0.8
assert parity.factors["p_value_adjusted"][0] < 1e-6
assert (parity.group_source, parity.test) == ("rows", "pearson")
status_counts = dict(parity.cells["status"].value_counts().iter_rows())
assert status_counts["too_thin"] == 9
assert status_counts["overrepresented"] >= 3
assert status_counts["underrepresented"] >= 8
cell_rows = {(row["class_name"], row["level"]): row for row in parity.cells.iter_rows(named=True)}
assert cell_rows[("m210 × [0, 20)", "[45, 80)")]["status"] == "overrepresented"
assert cell_rows[("mavic × [60, 150)", "[45, 80)")]["status"] == "overrepresented"
assert cell_rows[("mavic × [20, 60)", "[80, 91)")]["status"] == "underrepresented"

# %% [markdown]
# Similarly, you can evaluate {class}`.Diversity` across the factors of each drone:

# %%
diversity = Diversity().evaluate(by_drone)
print(diversity.classwise)

# %% [markdown]
# Diversity is low where a drone concentrates on one value. Each drone flies a narrow band of altitudes and camera
# pitches, so those factors are low for all three. The `class` factor is lowest for the `m210`, where swimmers are 93%
# of the detections, and the object size stays high for every drone, because each sees objects of all sizes.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
div = {(row["class_name"], row["factor_name"]): row for row in diversity.classwise.iter_rows(named=True)}
assert all(div[(d, f)]["is_low_diversity"] for d in ("m210", "mavic", "trinity") for f in ("altitude", "gimbal_pitch"))
assert (
    div[("m210", "class")]["diversity_value"]
    < 0.1
    < 0.3
    < min(div[("mavic", "class")]["diversity_value"], div[("trinity", "class")]["diversity_value"])
)
assert all(
    div[(d, "object_size")]["diversity_value"] > 0.5 and not div[(d, "object_size")]["is_low_diversity"]
    for d in ("m210", "mavic", "trinity")
)

# %% [markdown]
# ## Inspect class axis provenance
#
# Every class-conditional result includes a {class}`.ClassAxis` record detailing the active axis configuration:

# %%
axis = result.class_axis
assert axis
print(f"name:        {axis.name}")
print(f"source:      {axis.source}")
print(f"level:       {axis.level}")
print(f"groups:      {axis.groups}")
print(f"fan-out:     {axis.rows_per_group_entity:.2f} rows per {axis.level}")
print(f"vocabulary:  {axis.vocabulary}")

# %% [markdown]
# In automated evaluation pipelines, you should check `axis.source == "derived"` to confirm whether results were
# computed against a custom metadata axis rather than standard class labels. These parameters are also recorded in
# `result.meta().state`:

# %%
print({k: v for k, v in result.meta().state.items() if k.startswith("class_axis")})

# %% [markdown]
# ## Apply custom class axes to splitting and coverage
#
# Because `classed_by` updates both `class_labels` and `index2label`, any downstream function that operates on class
# labels uses the new axis automatically.
#
# For example, {func}`.split_dataset` will stratify folds across `drone` instead of object classes:

# %%
splits = split_dataset(by_drone_per_image, num_folds=2, stratify=True)
assert splits.class_axis
print(f"stratified on: {splits.class_axis.name} ({splits.class_axis.source})")

# %% [markdown]
# Custom class axes apply to other DataEval workflows:
#
# - {class}`.Coverage` computes per-group coverage across the custom axis.
# - {meth}`.OutliersOutput.aggregate_by_class` groups outlier metrics by the custom axis.
# - {class}`.Representation` requires ontology-aligned class labels and will reject a derived class axis.
#
# Custom class axes persist across view operations ({meth}`.Metadata.at`, {meth}`.Metadata.where`,
# {meth}`.Metadata.having`, and {meth}`.Metadata.agg`). However, like filters and view transformations, custom class
# axes are not serialized by {meth}`.Metadata.save`.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert (axis.name, axis.source, axis.level) == ("drone", "derived", "unit")
assert (splits.class_axis.name, splits.class_axis.source) == ("drone", "derived")

# %% [markdown]
# ## Conclusion
#
# You loaded SeaDrone, in which the class mix, the altitude and the camera pitch all differ by drone. You declared the
# missing readings, chose the bins, and measured how much the binning changes what the evaluators report. Then you
# pivoted the class axis to the drone. Against the class axis only the object size stood out, and against the drone axis
# the altitude and the camera pitch did: the class view could not see the platform effect. You evaluated the factors per
# image, listed the combinations of drone and altitude that the data never covers, tested the combinations that exist
# with `Parity`, checked which axis a result used, and stratified a split on it.

# %% [markdown]
# ## Next steps
#
# - [Control and reuse a factor's binning](./h2_control_factor_binning.py) — Declare explicit bin edges for
#   consistent factor levels across collections.
# - [Identify gaps in a labeled dataset](./tt_identify_dataset_gaps.py) — Find the classes and regions of the embedding
#   space that lack coverage.
# - [Dataset bias and coverage](../concepts/DatasetBias.md) — Learn about mutual information, parity, and diversity
#   metrics.

# %% [markdown]
# ## On your own
#
# To run the same analysis on your own dataset, build a `Metadata` for it that holds the environmental factors you
# suspect. Compare `Balance().evaluate(metadata)` with `Balance().evaluate(metadata.classed_by("<factor>"))` for each
# suspect factor. Use `metadata.at("unit")` when the factor is recorded per image and your dataset has several
# detections per image. Pass several factor names to `classed_by` to list the combinations that your data never covers.
# Declare a sentinel such as `-1` as missing with `Remap` before you bin a factor.
