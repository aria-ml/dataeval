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
# # How to detect drift in metadata factors

# %% [markdown]
# ## Problem statement
#
# Embedding-space drift detection will report that features drifted but cannot explain what
# changed. By analyzing metadata factors like altitude and camera angle, you will obtain
# checkable, human-readable drift sentences.
#
# In this guide, you will integrate {class}`.Metadata` directly into a drift detector to
# run a metadata drift pipeline end to end. You should then map the resulting p-values
# back to the underlying metadata to isolate and verify the root cause of the drift.

# %% [markdown]
# ### When to use
#
# You should use this approach when you want to:
#
# - Monitor data collection conditions rather than raw pixel changes.
# - Explain an embedding-space drift alarm by mapping it to recorded operational factors.
# - Screen a new collection against a reference before spending GPU hours embedding it.
# - Produce human-interpretable drift reports to justify findings.

# %% [markdown]
# ### What you will need
#
# 1. A dataset carrying metadata (this guide uses drone flight telemetry from SeaDrone).
# 1. A Python environment with the following packages installed:
#    - `dataeval`
#    - `maite-datasets`

# %% [markdown]
# ## Getting started

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # %pip install -q dataeval maite-datasets
except Exception:
    pass

# %%
from collections import Counter
from typing import Any

import numpy as np
import polars as pl
from maite_datasets.object_detection import SeaDrone

from dataeval import Metadata
from dataeval.data import Indices, View
from dataeval.shift import DriftUnivariate
from dataeval.types import ParseDateTime, Remap

# Every factor is worth reading; this guide's whole point is the named rows.
pl.Config.set_tbl_rows(20)

# %% [markdown]
# ## Load the dataset without decoding it
#
# You should pass `lazy=True` to return image handles that decode only when requested.
# This prevents unnecessary image loading, keeping data on disk and processing factors
# directly from memory. You can read more about this technique in the [lazy loading
# guide](./h2_lazy_load_images.py).

# %%
dataset = SeaDrone(root="./data", image_set="val", download=True, lazy=True)
datum_metadata: list[dict[str, Any]] = [dict(dataset[i][2]) for i in range(len(dataset))]

print(f"{len(dataset)} frames\n")
for key, value in sorted(datum_metadata[0].items()):
    print(f"  {key:16s} {value!r}")

# %% [markdown]
# This flight telemetry includes variables like altitude, gimbal orientation, and airframe type.
# You will use these operational conditions to phrase and detect dataset drift.

# %% [markdown]
# ## Split into two collection campaigns
#
# To establish a reference and operational pair, you will split the dataset using the
# `date_time` factor, separating the frames into two collection campaigns a year apart.

# %%
year_of = [m["date_time"][:4] for m in datum_metadata]
reference = View(dataset, Indices([i for i, y in enumerate(year_of) if y == "2020"]))
operational = View(dataset, Indices([i for i, y in enumerate(year_of) if y == "2021"]))

print(f"reference   (2020): {len(reference):4d} frames")
print(f"operational (2021): {len(operational):4d} frames")

# %% [markdown]
# ## Choose the level
#
# Flight telemetry is recorded once per frame. Because object detection datasets default
# to the instance level (one row per detection), you should set `view="unit"` to analyze
# metadata at the image level. This prevents crowded frames from skewing the results.
# See [Metadata levels](../concepts/MetadataLevels.md) for details on levels.

# %%
# `storage` names the folder each clip came from: it identifies the source rather than
# describing the flight, so it is excluded. Row identifiers need no manual exclusion - the
# datum's `id` is kept in DataEval's reserved `item_id` column instead of becoming a factor.
EXCLUDE = ["storage"]

print(Metadata(reference, view="unit", exclude=EXCLUDE).dropped_factors)

# %% [markdown]
# You should check {attr}`~.Metadata.dropped_factors` to see why certain factors are
# excluded. Here, {class}`.Metadata` drops three columns for two reasons:
#
# - `latitude` and `longitude` are **`mixed_types`**. Most rows hold a numeric coordinate,
#   but 25 hold the letter `'N'` or `'E'`. The column has no single type, so DataEval holds
#   it back instead of choosing a reading for you.
# - `date_time` is **`cardinality_over_budget`**. Every row holds a different timestamp, so
#   the column names its rows instead of grouping them. It is not numeric, so there is no
#   order to cut it into groups along.
#
# You do not have to accept either drop. {meth}`~.Metadata.repair` lets you declare the
# reading that turns such a column into a factor, and you will use it twice in this guide.
# You should handle the `mixed_types` pair now, before the comparison, for the reason below.

# %% [markdown]
# ## Make both campaigns agree on the factor set
#
# Whether a column is held back depends on the data you are reading, not on the extractor.
# Only the 2020 rows carry the `'N'` and `'E'` values, so the reference holds `latitude` and
# `longitude` back while the 2021 stream reads them as ordinary factors. One extractor then
# produces two different factor sets.
#
# You should fix this before you run the detector. A feature-wise detector compares feature
# 0 against feature 0, feature 1 against feature 1, and so on, so both sides must describe
# the same factors in the same order. Declaring the repair up front guarantees that: the
# reference reads those columns as numbers, and so does every stream you measure against it.

# %%
# The letters mean "no reading was taken", which is what `-1` means in the numeric telemetry
# columns of this dataset, so you can map them to the same value.
COORDINATES = [Remap("latitude", {"N": -1.0}), Remap("longitude", {"E": -1.0})]

print(Metadata(reference, view="unit", exclude=EXCLUDE).repair(COORDINATES).dropped_factors)

# %% [markdown]
# Only `date_time` is still dropped, and both campaigns now read the same fourteen factors.

# %% [markdown]
# ## Build the extractor and run the detector
#
# You will create a reusable {class}`.Metadata` extractor. The first call fits the reference
# and records the cuts and vocabularies, ensuring that subsequent transformations map the
# operational stream to the same bins. You should refer to the [factor binning guide]
# (./h2_control_factor_binning.py) if you need to declare the cuts explicitly.

# %%
extractor = Metadata(reference, view="unit", exclude=EXCLUDE).repair(COORDINATES)

detector = DriftUnivariate(method="ks", extractor=extractor).fit(reference)
result = detector.predict(operational)

feature_drift = np.asarray(result.details["feature_drift"])
p_values = np.asarray(result.details["p_vals"])
print(f"drifted: {result.drifted}   {feature_drift.sum()} of {len(feature_drift)} factors")

# %% [markdown]
# ## Put names on the result
#
# Because {class}`.Metadata` implements the {class}`~dataeval.protocols.NamedFeatureExtractor`
# protocol, you will map positional outputs (such as `p_vals`) directly to the names of the
# corresponding metadata factors.

# %%
# None when an extractor has no names to give (embeddings); Metadata always does.
factor_names = list(result.feature_names or [])

drift_table = pl.DataFrame({"factor": factor_names, "p_value": p_values, "drifted": feature_drift}).sort("p_value")
print(drift_table)

# %% [markdown]
# Fourteen named factors are reported. For example, `drone` does not register as drifted,
# but `gimbal_pitch` drifts significantly at `p ≈ 1e-15`. Unlike embedding-space detectors,
# you will see exactly which operational conditions changed.

# %% [markdown]
# ## Go back to the source
#
# Since a p-value only indicates a distribution change, you should inspect the underlying
# data to verify why a factor shifted. You can access raw and binned values directly from
# {attr}`~.Metadata.dataframe` to verify the source of the drift.

# %%
rows = extractor.dataframe.filter(pl.col("level") == "unit")
print(rows.select("item_index", "altitude", "altitude↕", "gimbal_pitch", "gimbal_pitch↕").head(5))


# %% [markdown]
# You should inspect `gimbal_pitch` to determine if the camera angle really shifted or if
# the telemetry logging changed.


# %%
def describe(key: str) -> None:
    """Compare a raw factor across the two campaigns, separating sentinels from real values."""
    print(f"{key}:")
    for year in ("2020", "2021"):
        values = np.array([m[key] for m, y in zip(datum_metadata, year_of, strict=True) if y == year])
        recorded = values[values != -1]
        print(
            f"  {year}  missing (-1): {100 * (values == -1).mean():4.1f}%"
            f"   median of the rest: {np.median(recorded):6.1f}"
        )


describe("gimbal_pitch")

# %% [markdown]
# The camera did not move. The median angle remains nearly identical, but a quarter of the
# 2021 frames failed to record telemetry and defaulted to `-1`. Since the binner treats
# `-1` as a valid numeric value, a spurious drift alarm is triggered. You should identify
# these telemetry logging failures rather than attributing the drift to camera movement.

# %% [markdown]
# ## Fix the data, not the threshold
#
# Rather than adjusting the p-value threshold, you should filter out the missing telemetry
# codes and re-evaluate the cleaned reference and operational datasets.

# %%
has_telemetry = [
    i for i, m in enumerate(datum_metadata) if m["altitude"] != -1 and m["gimbal_pitch"] != -1 and m["speed"] != -1
]
by_year = {y: [i for i in has_telemetry if year_of[i] == y] for y in ("2020", "2021")}
clean_reference = View(dataset, Indices(by_year["2020"]))
clean_operational = View(dataset, Indices(by_year["2021"]))

# Declare the same reading as before, so both passes measure the same fourteen factors.
clean_extractor = Metadata(clean_reference, view="unit", exclude=EXCLUDE).repair(COORDINATES)
clean_result = DriftUnivariate(method="ks", extractor=clean_extractor).fit(clean_reference).predict(clean_operational)

# Join on the factor name rather than lining the two results up by position, so the table
# stays correct even if a pass drops a factor the other kept.
before_df = pl.DataFrame({"factor": factor_names, "p_before": p_values, "before": feature_drift})
after_df = pl.DataFrame({
    "factor": list(clean_result.feature_names or []),
    "p_after": np.asarray(clean_result.details["p_vals"]),
    "after": np.asarray(clean_result.details["feature_drift"]),
})
comparison = before_df.join(after_df, on="factor", how="inner").with_columns(
    changed=pl.col("before") != pl.col("after")
)
print(comparison.sort("changed", descending=True))

# %% [markdown]
# After filtering, you will see four verdicts flip. `gimbal_pitch` no longer registers as
# drifted, which confirms the earlier result was a false alarm caused by missing data.
# `drone` now registers as drifted, which shows a real change in the airframe mix that the
# missing rows had hidden.

# %%
for label, indices in by_year.items():
    subset = [datum_metadata[i] for i in indices]
    altitudes = np.array([m["altitude"] for m in subset])
    resolutions = Counter("{}x{}".format(m["width"], m["height"]) for m in subset)  # noqa: UP032
    print(f"{label}  n={len(subset):3d}  drones={dict(Counter(m['drone'] for m in subset))}")
    print(f"        resolutions={dict(resolutions)}")
    print(f"        altitude p25={np.percentile(altitudes, 25):5.1f}  p75={np.percentile(altitudes, 75):5.1f}")

# %% [markdown]
# You should verify these findings directly against the collection context:
#
# - **`drone` drifts** because the drone types changed between campaigns.
# - **`width` and `height` stop drifting** because the surviving frames share the same resolution.
# - **`altitude` continues to drift** on its range and spread, rather than its median.
#
# You will use `item_index` from {attr}`~.Metadata.dataframe` to locate the frames behind any
# row, and you should use the [trace findings guide](./h2_trace_findings_to_source.py) to
# retrieve and view them.

# %% [markdown]
# ## Read the column that is still dropped
#
# One factor is still dropped: `date_time`, held back as `cardinality_over_budget`. You
# repaired the `mixed_types` pair at the start because the comparison needed them. You
# should check {attr}`~.Metadata.unusable` to see what is behind a drop and what you have to
# write a repair against.

# %%
review = Metadata(reference, view="unit", exclude=EXCLUDE)
for name, held in review.unusable.items():
    values = held.distinct.get("text", ())
    print(f"{name:10s} {held.reasons[0]:26s} repairable={held.repairable}")
    print(f"{'':10s} counts={dict(held.counts)}  e.g. {values[:2]}")

# %% [markdown]
# `counts` reports every row as text and no mixture, so nothing about these values
# disagrees. Every row just holds a different value, so the column names its rows instead of
# grouping them. It needs a vocabulary, and {class}`~dataeval.types.ParseDateTime` gives it
# one by reading each value as the period it falls in. You should read the distinct values
# first to pick the format: these are ISO 8601 with microseconds, which is the default, so
# you do not need to pass `format=`.
#
# You should choose the period carefully, because the campaigns are split by time. An
# absolute period such as a month or a day separates them completely, which only restates
# the split. A recurring position does not: every campaign has a 14:00, so
# `every="hour_of_day"` asks whether the flying moved to a different part of the day.

# %%
READINGS = [*COORDINATES, ParseDateTime("date_time", every="hour_of_day")]

repaired = Metadata(reference, view="unit", exclude=EXCLUDE).repair(READINGS)
print("still dropped:", dict(repaired.dropped_factors))
print("factors:      ", len(repaired.factor_names), "up from", len(factor_names))
print("date_time now:", sorted(set(repaired.rows_at("unit")["date_time"].to_list())))

# %% [markdown]
# All three columns are factors now, and `date_time` holds the hour of the day each frame
# was flown. A repair is a declaration, not a one-off edit to a dataframe. DataEval records
# it on the metadata, so you can read it back from {attr}`~.Metadata.repairs`, store it with
# {meth}`~.Metadata.save`, and apply it to next year's campaign without deciding it again.
# You can drop a repair with {meth}`~.Metadata.unrepair`.

# %%
time_of_day = DriftUnivariate(method="ks", extractor=repaired).fit(reference).predict(operational)
by_name = dict(zip(time_of_day.feature_names or [], np.asarray(time_of_day.details["p_vals"]), strict=True))
print(f"date_time (hour of day)  p = {by_name['date_time']:.3e}")

for year in ("2020", "2021"):
    hours = Counter(int(m["date_time"][11:13]) for m in datum_metadata if m["date_time"][:4] == year)
    print(f"  {year}  {dict(sorted(hours.items()))}")

# %% [markdown]
# The repaired factor reports a real change in collection conditions. The 2020 campaign flew
# between 12:00 and 15:00, while 2021 started as early as 10:00 and moved most of its flying
# an hour earlier. Time of day affects sun angle, sea state and thermal contrast, so this
# tells you something about the collection that the first pass could not report at all.

# %% [markdown]
# ## What you learned
#
# 1. **You will use {class}`.Metadata` as a {class}`~dataeval.protocols.FeatureExtractor`**.
#    Passing it as `extractor=` configures both the fit and predict stages of the detector.
# 1. **You should fit on the reference and transform on the operational data**. This ensures
#    both datasets are mapped to the same bins for a valid comparison.
# 1. **You should choose the level deliberately**. Setting `view="unit"` avoids replicating
#    telemetry factors across individual object detections.
# 1. **You will map results to factor names**. Reusing the fitted extractor lets you resolve
#    positional p-values to specific, named operational conditions.
# 1. **You will separate real drift from data artifacts**. Investigating the raw underlying
#    data is necessary to distinguish true changes from missing-data codes.
# 1. **You should check why a factor was dropped before you accept it**. A column held back
#    for mixing types or for naming its rows becomes a factor once you declare how to read
#    it with {meth}`~.Metadata.repair`.
# 1. **You should declare a repair before you compare two datasets**. Both sides must read
#    the same factors in the same order, and declaring the reading up front is what makes
#    them agree.
# 1. **You should pick a period that a time split cannot explain away**. An absolute period
#    separates two campaigns on its own; a recurring position such as `every="hour_of_day"`
#    stays comparable across them.

# %% [markdown]
# ## Next steps
#
# - [How to control factor binning](./h2_control_factor_binning.py) — Declare bin edges for
#   consistent mapping.
# - [How to trace findings](./h2_trace_findings_to_source.py) — Retrieve the specific frames
#   associated with a drift finding.
# - [How to lazy load images](./h2_lazy_load_images.py) — Speed up metadata extraction.
# - [Monitoring tutorial](./tt_monitor_shift.py) — Analyze embedding-space drift.
# - [Distribution shift](../concepts/DistributionShift.md) — Select drift detectors.
# - [Metadata levels](../concepts/MetadataLevels.md) — Understand metadata views and levels.
