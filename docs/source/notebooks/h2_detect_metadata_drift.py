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
# excluded. Here, {class}`.Metadata` automatically drops the unique row identifier
# `date_time` to prevent trivial, perfect drift signals.

# %% [markdown]
# ## Build the extractor and run the detector
#
# You will create a reusable {class}`.Metadata` extractor. The first call fits the reference
# and records the cuts and vocabularies, ensuring that subsequent transformations map the
# operational stream to the same bins. You should refer to the [factor binning guide]
# (./h2_control_factor_binning.py) if you need to declare the cuts explicitly.

# %%
extractor = Metadata(view="unit", exclude=EXCLUDE)

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

clean_extractor = Metadata(view="unit", exclude=EXCLUDE)
clean_result = DriftUnivariate(method="ks", extractor=clean_extractor).fit(clean_reference).predict(clean_operational)

comparison = pl.DataFrame({
    "factor": list(clean_result.feature_names or []),
    "p_before": p_values,
    "p_after": np.asarray(clean_result.details["p_vals"]),
    "before": feature_drift,
    "after": np.asarray(clean_result.details["feature_drift"]),
}).with_columns(changed=pl.col("before") != pl.col("after"))
print(comparison.sort("changed", descending=True))

# %% [markdown]
# After filtering, you will see four verdicts flip. Specifically, `gimbal_pitch` no longer
# registers as drifted, confirming the previous result was a false alarm caused by missing
# data. Conversely, `drone` now clearly registers as drifted, uncovering a genuine change
# in the airframe mix that was previously obscured.

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
