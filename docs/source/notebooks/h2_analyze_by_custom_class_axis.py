# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dataeval
#     language: python
#     name: python3
# ---

# %% [markdown]
# (custom-class-axis)=
# # How to analyze bias against a custom class axis

# %% [markdown]
# ## Problem statement
#
# DataEval evaluators — including {class}`.Balance`, {class}`.Diversity`,
# {class}`.Parity`, {class}`.Coverage`, and {func}`.split_dataset` — condition
# on dataset class labels by default. In many datasets, bias and underrepresentation
# are driven not by object classes, but by environmental or operational factors
# such as weather, sensor type, time of day, or collection site.
#
# {meth}`.Metadata.classed_by` allows you to pivot the class axis to one or more
# metadata factors. The target factor's discrete levels become the new class labels,
# and all class-conditional evaluators condition on them instead. The dataset's
# original class labels are retained as a factor named `class`.
#
# This guide shows you how to:
#
# - Pivot the class axis to an environmental factor to detect metadata imbalance.
# - Disentangle class-level findings from correlated environmental conditions.
# - Evaluate factors at the unit (image) level rather than the instance (detection) level.
# - Combine multiple factors into a composite axis to identify uncollected data combinations.
# - Stratify dataset splits across metadata conditions.

# %% [markdown]
# ### When to use
#
# You should use a custom class axis when:
#
# - You need to evaluate balance, diversity, or parity across environmental or operational conditions (such as weather, site, or sensor platform).
# - A class and an environmental factor co-occur, and you need to determine which one drives an observed imbalance.
# - You want to identify uncollected or undersampled combinations of conditions across multiple metadata factors.
# - You need to perform bias analysis at the image or video level on multi-object datasets.
# - You want to stratify dataset splits across operational metadata rather than class labels.

# %% [markdown]
# ### What you will need
#
# 1. A Python environment with `dataeval` installed.
# 2. No external data; this guide creates a synthetic dataset in memory.

# %% [markdown]
# ## Getting started
#
# First, import the required libraries to set up the example.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval
except Exception:
    pass

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# %%
from dataclasses import dataclass

import numpy as np
import polars as pl

from dataeval import Metadata
from dataeval.bias import Balance, Diversity, Parity
from dataeval.data import split_dataset
from dataeval.protocols import DatasetMetadata, DatumMetadata

# %% [markdown]
# ## Create a dataset with correlated metadata
#
# You will create a synthetic object detection dataset of 60 images containing unit-level
# factors (`weather`, `time_of_day`, and `altitude_m`) and bounding box detections.
#
# Two correlations are intentionally structured into the dataset:
#
# - **Class and weather co-occurrence:** Foggy images contain mostly trucks, while clear images contain mostly people.
# - **Weather and altitude correlation:** Foggy images are captured at high altitude (300–400m), while clear and rainy images are captured at low altitude (50–200m).
#
# Because `truck` co-occurs with `foggy`, a standard class-conditional evaluation
# will confound altitude with class. By using a custom class axis, you will show
# that altitude is better explained by weather.
#
# If you evaluate only from a class-conditional perspective, you might focus data
# collection on gathering images of trucks in different weather conditions, which does
# not resolve the underlying dataset imbalance. Instead, you should identify the true
# confounding factor and collect data across a diverse set of weather conditions and
# altitudes.


# %%
@dataclass
class BoxTarget:
    """A minimal object detection target: boxes, labels, and scores."""

    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


WEATHER = ("clear", "rainy", "foggy")
TIME_OF_DAY = ("day", "night")
# Class mix per weather: clear is mostly people, foggy is mostly trucks.
CLASS_MIX = {
    "clear": [0.85, 0.10, 0.05],
    "rainy": [0.10, 0.80, 0.10],
    "foggy": [0.05, 0.15, 0.80],
}


class PatrolDataset:
    """A synthetic detection dataset whose class mix depends on the weather."""

    def __init__(self, images: int) -> None:
        rng = np.random.default_rng(0)
        self._weather = rng.choice(WEATHER, images)
        self._time = rng.choice(TIME_OF_DAY, images)
        # Altitude is correlated with weather, but is better explained by weather than by object class
        self._altitude = np.where(
            self._weather == "foggy",
            rng.uniform(300.0, 400.0, images),
            rng.uniform(50.0, 200.0, images),
        )
        self._counts = rng.integers(2, 5, images)
        # Drawn once in __init__ so every read of an item returns the same values
        self._labels = [
            rng.choice(3, count, p=CLASS_MIX[weather])
            for weather, count in zip(self._weather, self._counts, strict=True)
        ]
        self._areas = [rng.uniform(20.0, 200.0, count) for count in self._counts]
        self.metadata = DatasetMetadata(
            id="patrol-demo",
            index2label={0: "person", 1: "car", 2: "truck"},
        )

    def __len__(self) -> int:
        return len(self._counts)

    def __getitem__(self, index: int) -> tuple[np.ndarray, BoxTarget, DatumMetadata]:
        count = int(self._counts[index])
        target = BoxTarget(
            boxes=np.tile(np.array([[0.0, 0.0, 10.0, 10.0]]), (count, 1)),
            labels=self._labels[index],
            scores=np.ones(count),
        )
        # weather, time_of_day, and altitude_m describe the image; box_area describes each box
        datum_metadata: DatumMetadata = DatumMetadata(
            id=index,
            **{
                "weather": str(self._weather[index]),
                "time_of_day": str(self._time[index]),
                "altitude_m": float(self._altitude[index]),
                "box_area": self._areas[index].tolist(),
            },
        )
        return np.zeros((3, 32, 32), dtype=np.uint8), target, datum_metadata


dataset = PatrolDataset(60)
# Declare explicit bin counts for continuous factors to ensure stable comparisons
metadata = Metadata(
    dataset,
    exclude=["id"],
    continuous_factor_bins={"altitude_m": 3, "box_area": 4},
)

print(f"images:     {metadata.level_counts['unit']}")
print(f"detections: {metadata.level_counts['instance']}")
print(f"factors:    {list(metadata.factor_names)}")
print(f"class axis: {metadata.class_axis} ({metadata.class_axis_source})")

# %% [markdown]
# ## Pivot to a single metadata factor
#
# Call {meth}`.Metadata.classed_by` to create a copy of the metadata configured with a new class axis. The original metadata object remains unchanged.

# %%
by_weather = metadata.classed_by("weather")

print(f"axis:     {by_weather.class_axis} ({by_weather.class_axis_source})")
print(f"groups:   {dict(by_weather.index2label)}")
print(f"original: {metadata.class_axis} ({metadata.class_axis_source})")

# %% [markdown]
# When you pivot to a factor:
#
# 1. `class_labels` contains the integer codes of the new axis (`weather`) rather than the original class labels.
# 2. `index2label` maps those codes to group names (`clear`, `rainy`, `foggy`).
# 3. The pivoted factor (`weather`) is removed from `factor_names` to prevent self-correlation, and the original label set is added as a factor named `class`.

# %%
print(f"before: {list(metadata.factor_names)}")
print(f"after:  {list(by_weather.factor_names)}")

# %% [markdown]
# ## Evaluate balance across the custom axis
#
# You will now evaluate {class}`.Balance` against the `weather` axis. Because the original class labels are now a factor named `class`, you can measure how much weather correlates with the object classes present.

# %%
result = Balance().evaluate(by_weather)
print(result.balance)

# %% [markdown]
# In the output above, both `class` and `altitude_m` show strong mutual information with `weather`, indicating that weather is correlated with both the object class mix and flight altitude.
#
# Running {class}`.Balance` on the original metadata displays the same correlation from the class perspective:

# %%
print(Balance().evaluate(metadata).balance)

# %% [markdown]
# ## Disentangle class and metadata associations
#
# While overall balance scores indicate that an association exists, `classwise` metrics reveal which specific groups drive the relationship.
#
# Inspect the `altitude_m` rows from the default class-conditional evaluation:

# %%
default = Balance().evaluate(metadata)
print(default.classwise.filter(pl.col("factor_name") == "altitude_m"))

# %% [markdown]
# In the default class-conditional view, `truck` is flagged as imbalanced against altitude. Because trucks predominantly appear in foggy weather (which is flown at high altitude), the class-conditional view attributes the altitude imbalance directly to `truck`.
#
# Now inspect the same factor under the `weather` class axis:

# %%
print(result.classwise.filter(pl.col("factor_name") == "altitude_m"))

# %% [markdown]
# Under the `weather` axis, `foggy` scores `1.0`: within this dataset, foggy conditions deterministically identify high-altitude flights.
#
# The class-conditional view can only attribute variation to class labels. When a class co-occurs with an environmental condition, any effect caused by that condition is reported as class imbalance. Pivoting to a metadata axis allows you to isolate whether the imbalance originates from the class label or from the environmental factor.

# %% [markdown]
# ## Evaluate metadata at the image level with at()
#
# In object detection datasets, class labels exist at the detection (`instance`) level, while factors like `weather` exist at the image (`unit`) level. Evaluating the default metadata at the unit level fails because an image containing multiple detections does not have a single class label:

# %%
try:
    Balance().evaluate(metadata.at("unit"))
except ValueError as error:
    print(f"class axis: {error}")

# %% [markdown]
# Because `weather` is defined for every image, pivoting to `weather` allows you to evaluate bias at the unit level directly:

# %%
by_weather_per_image = metadata.at("unit").classed_by("weather")
print(Balance().evaluate(by_weather_per_image).balance)

# %% [markdown]
# You can inspect {attr}`.ClassAxis.rows_per_group_entity` to verify the fan-out ratio between rows and entities:
#
# - In the detection view (`instance`), image-level factors are replicated across detections (fan-out > 1.0), weighting crowded images more heavily.
# - In the image view (`unit`), each image is counted exactly once (fan-out = 1.0).

# %%
print(
    f"detections view: {by_weather.class_labels.shape[0]} rows, "
    f"fan-out {by_weather.class_axis_info.rows_per_group_entity:.2f}"
)
print(
    f"images view:     {by_weather_per_image.class_labels.shape[0]} rows, "
    f"fan-out {by_weather_per_image.class_axis_info.rows_per_group_entity:.2f}"
)

# %% [markdown]
# Note that `class` is absent when pivoting at the unit level, as multi-detection images have no single class label. If you need a unit-level class summary, you can aggregate labels with {meth}`.Metadata.agg`.

# %%
print(f"detections view factors: {list(by_weather.factor_names)}")
print(f"images view factors:     {list(by_weather_per_image.factor_names)}")

# %% [markdown]
# ## Create composite axes to identify missing combinations
#
# You can pass multiple factor names to {meth}`.Metadata.classed_by` to construct a composite class axis. The resulting groups represent all observed combinations, formatted as `factor1 × factor2`.
#
# By crossing factors, you can inspect which combinations exist in your data and which are missing:

# %%
cells = metadata.at("unit").classed_by("weather", "altitude_m")

print(f"axis: {cells.class_axis}")
counts = np.bincount(cells.class_labels, minlength=len(cells.index2label))
for code, name in sorted(cells.index2label.items(), key=lambda item: item[1]):
    print(f"  {name:32s} {counts[code]:3d} images")
print(f"\ncells present: {len(cells.index2label)} of {3 * 3} possible")

# %% [markdown]
# ## Detect undersampled combinations with Parity and Diversity
#
# In this dataset, only 5 of 9 possible weather and altitude combinations are present: there is no low-altitude fog and no high-altitude clear weather.
#
# This reveals specific operational gaps for future data collection:
#
# - The data gap is a specific factor combination (`foggy × low altitude`) rather than a missing class.
# - As long as these cells remain empty, model performance under foggy conditions cannot be separated from model performance at high altitude.
#
# You should use {class}`.Parity` to detect factor combinations with insufficient sample sizes:

# %%
parity = Parity().evaluate(cells)
print(parity.factors)
print(f"\nthin cells: {parity.insufficient_data}")

# %% [markdown]
# Similarly, you can evaluate {class}`.Diversity` across categorical factor combinations such as `weather` and `time_of_day`:

# %%
by_conditions = metadata.classed_by("weather", "time_of_day")
print(Diversity().evaluate(by_conditions).classwise.head(6))

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
# In automated evaluation pipelines, you should check `axis.source == "derived"` to confirm whether results were computed against a custom metadata axis rather than standard class labels. These parameters are also recorded in `result.meta().state`:

# %%
print({k: v for k, v in result.meta().state.items() if k.startswith("class_axis")})

# %% [markdown]
# ## Apply custom class axes to splitting and coverage
#
# Because `classed_by` updates both `class_labels` and `index2label`, any downstream function that operates on class labels uses the new axis automatically.
#
# For example, {func}`.split_dataset` will stratify folds across `weather` instead of object classes:

# %%
splits = split_dataset(by_weather_per_image, num_folds=2, stratify=True)
assert splits.class_axis
print(f"stratified on: {splits.class_axis.name} ({splits.class_axis.source})")

# %% [markdown]
# Custom class axes apply to other DataEval workflows:
#
# - {class}`.Coverage` computes per-group coverage across the custom axis.
# - {meth}`.OutliersOutput.aggregate_by_class` groups outlier metrics by the custom axis.
# - {class}`.Representation` requires ontology-aligned class labels and will reject a derived class axis.
#
# Custom class axes persist across view operations ({meth}`.Metadata.at`, {meth}`.Metadata.where`, {meth}`.Metadata.having`, and {meth}`.Metadata.agg`). However, like filters and view transformations, custom class axes are not serialized by {meth}`.Metadata.save`.

# %% [markdown]
# ## Next steps
#
# - [How to reason about factor binning across levels](./h2_bin_factors_by_level.py) — Understand how factor levels influence binning and fan-out.
# - [How to control factor binning](./h2_control_factor_binning.py) — Declare explicit bin edges for consistent factor levels across collections.
# - [How to detect undersampling](./h2_detect_undersampling.py) — Measure coverage across custom groups and classes.
# - [Dataset Bias and Coverage](../concepts/DatasetBias.md) — Learn about mutual information, parity, and diversity metrics.
