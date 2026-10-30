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
# # Detect drift in metadata factors
#
# Embedding-space drift detection reports that features drifted but not what changed. Metadata factors such as altitude
# and camera angle give checkable, human-readable drift statements: a {class}`.Metadata` extractor plugs into a drift
# detector, and the p-values map back to named factors. See [Distribution shift](../concepts/DistributionShift.md).
#
# Use it to:
#
# - monitor data collection conditions rather than raw pixel changes
# - explain an embedding-space drift alarm with the recorded operational factors
# - screen a new collection against a reference before spending GPU hours embedding it
#
# You need `dataeval`, `maite-datasets` and a dataset that carries metadata. This guide uses SeaDrone flight telemetry.

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
from dataeval.types import Remap

# Every factor is worth reading; this guide's whole point is the named rows.
pl.Config.set_tbl_rows(20)

# %% [markdown]
# ## Split the dataset into two campaigns
#
# Pass `lazy=True` so that images decode only when requested. See the
# [Defer image decoding](./h2_wrap_dataframe_dataset.py#defer-image-decoding) step of how to wrap a DataFrame as a
# dataset. The flight telemetry includes altitude, gimbal orientation and airframe type.

# %%
dataset = SeaDrone(root="./data", image_set="val", download=True, lazy=True)
datum_metadata: list[dict[str, Any]] = [dict(dataset[i][2]) for i in range(len(dataset))]

print(f"{len(dataset)} frames\n")
for key, value in sorted(datum_metadata[0].items()):
    print(f"  {key:16s} {value!r}")

# %% [markdown]
# Split on the `date_time` factor into a reference and an operational campaign a year apart.

# %%
year_of = [m["date_time"][:4] for m in datum_metadata]
reference = View(dataset, Indices([i for i, y in enumerate(year_of) if y == "2020"]))
operational = View(dataset, Indices([i for i, y in enumerate(year_of) if y == "2021"]))

print(f"reference   (2020): {len(reference):4d} frames")
print(f"operational (2021): {len(operational):4d} frames")

# %% [markdown]
# ## Make both campaigns agree on the factor set
#
# Flight telemetry is recorded once per frame, but object detection datasets default to one row per detection. Set
# `view="unit"` to analyze the factors at the image level. See [Metadata levels](../concepts/MetadataLevels.md).

# %%
# `storage` names the folder each clip came from: it identifies the source rather than
# describing the flight, so it is excluded. Row identifiers need no manual exclusion - the
# datum's `id` is kept in DataEval's reserved `item_id` column instead of becoming a factor.
EXCLUDE = ["storage"]

print(Metadata(reference, view="unit", exclude=EXCLUDE).dropped_factors)

# %% [markdown]
# {attr}`~.Metadata.dropped_factors` shows why {class}`.Metadata` holds a column back:
#
# - `latitude` and `longitude` are `mixed_types`: most rows hold a number, but 25 hold the letter `'N'` or `'E'`.
# - `date_time` is `cardinality_over_budget`: every row holds a different timestamp, so there is nothing to group.
#
# {meth}`~.Metadata.repair` declares the reading that turns such a column into a factor. Repair the `mixed_types` pair
# before you compare: only the 2020 rows carry the letters, so the reference would hold `latitude` and `longitude` back
# while the 2021 stream read them as factors, and a feature-wise detector needs the same factors on both sides, in the
# same order.

# %%
# The letters mean "no reading was taken", as `-1` does in this dataset's numeric columns.
# Latitude and longitude are known factors, checked as real coordinates, so a sentinel would
# pass as a real position: map both spellings to NaN, which DataEval reads as missing.
COORDINATES = [
    Remap("latitude", {"N": np.nan, -1: np.nan}),
    Remap("longitude", {"E": np.nan, -1: np.nan}),
]

print(Metadata(reference, view="unit", exclude=EXCLUDE).repair(COORDINATES).dropped_factors)

# %% [markdown]
# Only `date_time` is still dropped, and both campaigns now read the same fourteen factors. To repair a column like
# `date_time`, see [Identify bias and correlations](./tt_identify_bias.py).

# %% [markdown]
# ## Run the detector
#
# Build a reusable {class}`.Metadata` extractor. The first call fits the reference and records the cuts and
# vocabularies, so the operational stream maps to the same bins. To declare the cuts yourself, see
# [Control and reuse a factor's binning](./h2_control_factor_binning.py).

# %%
extractor = Metadata(reference, view="unit", exclude=EXCLUDE).repair(COORDINATES)

detector = DriftUnivariate(method="ks", extractor=extractor).fit(reference)
result = detector.predict(operational)

feature_drift = np.asarray(result.details["feature_drift"])
p_values = np.asarray(result.details["p_vals"])
print(f"drifted: {result.drifted}   {feature_drift.sum()} of {len(feature_drift)} factors")

# %% [markdown]
# {class}`.Metadata` implements the {class}`~dataeval.protocols.NamedFeatureExtractor` protocol, so you can map
# positional outputs such as `p_vals` to factor names.

# %%
factor_names = list(result.feature_names)

drift_table = pl.DataFrame({"factor": factor_names, "p_value": p_values, "drifted": feature_drift}).sort("p_value")
print(drift_table)

# %% [markdown]
# Fourteen named factors are reported. `drone` does not register as drifted, but `gimbal_pitch` drifts at `p ≈ 1e-15`.

# %% [markdown]
# ## Check a drifted factor against the source
#
# A p-value only says that a distribution changed. The raw and binned values are in {attr}`~.Metadata.dataframe`.

# %%
rows = extractor.dataframe.filter(pl.col("level") == "unit")
print(rows.select("item_index", "altitude", "altitude↕", "gimbal_pitch", "gimbal_pitch↕").head(5))


# %% [markdown]
# Check whether the camera angle really shifted or the telemetry logging changed.


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
# The camera did not move. The median angle is nearly identical, but a quarter of the 2021 frames failed to record
# telemetry and defaulted to `-1`. The binner treats `-1` as a valid number, so the alarm is spurious.

# %% [markdown]
# ## Fix the data, not the threshold
#
# Filter out the missing telemetry codes and rerun both passes.

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
    "factor": list(clean_result.feature_names),
    "p_after": np.asarray(clean_result.details["p_vals"]),
    "after": np.asarray(clean_result.details["feature_drift"]),
})
comparison = before_df.join(after_df, on="factor", how="inner").with_columns(
    changed=pl.col("before") != pl.col("after")
)
print(comparison.sort("changed", descending=True))

# %% [markdown]
# Four verdicts flip. `gimbal_pitch` no longer registers as drifted, so the earlier result was a false alarm caused by
# missing data. `drone` now registers as drifted, a real change in the airframe mix that the missing rows had hidden.
# `width` and `height` stop drifting.

# %%
for label, indices in by_year.items():
    subset = [datum_metadata[i] for i in indices]
    altitudes = np.array([m["altitude"] for m in subset])
    resolutions = Counter("{}x{}".format(m["width"], m["height"]) for m in subset)  # noqa: UP032
    print(f"{label}  n={len(subset):3d}  drones={dict(Counter(m['drone'] for m in subset))}")
    print(f"        resolutions={dict(resolutions)}")
    print(f"        altitude p25={np.percentile(altitudes, 25):5.1f}  p75={np.percentile(altitudes, 75):5.1f}")

# %% [markdown]
# The collection context confirms each finding: `drone` drifts because the drone types changed, `width` and `height`
# stop drifting because the surviving frames share one resolution, and `altitude` still drifts in range and spread more
# than in median. The `item_index` column of {attr}`~.Metadata.dataframe` locates the frames behind any row, and
# [Trace a finding back to its source](./h2_trace_findings_to_source.py) retrieves and shows them.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(factor_names) == 14
assert feature_drift.sum() == 11
drifted_before = dict(zip(factor_names, feature_drift, strict=True))
assert drifted_before["gimbal_pitch"]
assert not drifted_before["drone"]
assert p_values[factor_names.index("gimbal_pitch")] < 1e-6
missing_2021 = np.mean([m["gimbal_pitch"] == -1 for m, y in zip(datum_metadata, year_of, strict=True) if y == "2021"])
assert 0.2 < missing_2021 < 0.3
assert set(comparison.filter(pl.col("changed"))["factor"]) == {"drone", "gimbal_pitch", "height", "width"}

# %% [markdown]
# ## Next steps
#
# - [Distribution shift](../concepts/DistributionShift.md) — Understand covariate shift and drift detection across
#   datasets.
# - [Monitor shifts in operational data](./tt_monitor_shift.py) — Detect distribution shift between training and
#   operational data.
# - [Trace a finding back to its source](./h2_trace_findings_to_source.py) — Retrieve the specific frames
#   associated with a drift finding.
