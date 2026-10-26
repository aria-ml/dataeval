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
# This tutorial shows you how to condition DataEval's bias evaluators on an environmental factor and not on the object
# classes. You create a dataset in which class and weather are correlated, then separate the two.
#
# Estimated time to complete: 15 minutes
#
# Relevant ML stages: [Data Engineering](../getting-started/roles/ML_Lifecycle.md#data-engineering)
#
# Relevant personas: [Data Engineer](../getting-started/roles/data_engineer.md), [T&E Engineer](../getting-started/roles/te_engineer.md)
#
# ## What you will do
#
# - Create a synthetic dataset in which class, weather and altitude are correlated
# - Pivot the class axis to an environmental factor and evaluate `Balance` against it
# - Disentangle class-level findings from correlated environmental conditions
# - Evaluate factors at the image level and not the detection level
# - Combine factors into a composite axis to find combinations that were never collected
# - Test the observed combinations with `Parity` and `Diversity`
# - Check which class axis a result used
# - Stratify dataset splits across metadata conditions
#
# ## What you will learn
#
# - How a custom class axis changes what the class-conditional evaluators report
# - How to tell whether a class or an environmental factor drives an imbalance
# - How the image-level and detection-level views weight an image
# - Which combinations of conditions the data never covers
#
# ## What you will need
#
# - A Python environment with `dataeval` installed
# - No external data: this tutorial creates a synthetic dataset in memory
#

# %% [markdown]
# ## Introduction
#
# DataEval evaluators, including {class}`.Balance`, {class}`.Diversity`, {class}`.Parity`, {class}`.Coverage`, and
# {func}`.split_dataset`, condition on dataset class labels by default. In many datasets, bias and underrepresentation
# are driven not by object classes, but by environmental or operational factors such as weather, sensor type, time of
# day, or collection site.
#
# {meth}`.Metadata.classed_by` allows you to pivot the class axis to one or more
# metadata factors. The target factor's discrete levels become the new class labels,
# and all class-conditional evaluators condition on them instead. The dataset's
# original class labels are retained as a factor named `class`.
#
# You should use a custom class axis when:
#
# - You need to evaluate balance, diversity, or parity across environmental or operational conditions (such as weather, site, or sensor platform).
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
# - Class and weather co-occur. Foggy images contain mostly trucks, while clear images contain mostly people.
# - Weather and altitude are correlated. Foggy images are captured at high altitude (300–400m), while clear and rainy
#   images are captured at low altitude (50–200m).
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
# Declare explicit bin counts for continuous factors to ensure stable comparisons. No exclusion of
# the per-item `id` is needed: it names the datum rather than describing it, so DataEval keeps it
# out of the factor space on its own (it becomes the reserved `item_id` column, not a factor).
metadata = Metadata(
    dataset,
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
# - In the image view (`unit`), each image is counted once (fan-out = 1.0).

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
# ## Test the observed combinations with Parity and Diversity
#
# In this dataset, only 5 of 9 possible weather and altitude combinations are present: there is no low-altitude fog and no high-altitude clear weather.
#
# This reveals specific operational gaps for future data collection:
#
# - The data gap is a specific factor combination (`foggy × low altitude`) rather than a missing class.
# - As long as these cells remain empty, model performance under foggy conditions cannot be separated from model performance at high altitude.
#
# The listing above is how you find absent combinations. {class}`.Parity` cannot: it tests the combinations that occur in the data, so a combination with no images has no cell to test. Parity answers a different question about the combinations that do exist. Once `weather` and `altitude_m` form the class axis, the remaining image-level factor is `time_of_day`. Parity tests whether it is spread across the five present cells as independence predicts, and which of those cells hold too few images to judge:

# %%
parity = Parity().evaluate(cells)
print(parity.factors.select("factor_name", "status", "score", "p_value_adjusted", "has_insufficient_data"))
print(f"\ngrouping: {parity.group_source} ({parity.n_rows} rows, {parity.n_groups} groups, {parity.test} test)")

findings = parity.cells.filter(pl.col("is_actionable") | (pl.col("status") == "too_thin"))
print(f"\nactionable cells: {parity.cells['is_actionable'].sum()}")
print(findings.select("level", "class_name", "status", "observed", "expected", "p_value_adjusted"))

# %% [markdown]
# Each row of {attr}`.ParityOutput.cells` is one `time_of_day` level in one weather and altitude cell, with its `status`:
#
# - The factor test is `inconclusive`: it finds no association between `time_of_day` and the cells (adjusted p-value of 0.63, score 0.0), but part of its table is too thin to count that as a negative result.
# - Four cells are `too_thin`: day and night in the two mid-altitude cells, which hold only 5 and 7 images. They are not findings, however small their p-values. Collect more images there to find out.
# - No cell is `underrepresented` or `overrepresented`, so no cell is actionable. The other six cells are `no_finding`.
# - The images are the independent unit here, so `group_source` is `rows` and the ungrouped Pearson test runs. On a detection-level view, Parity groups detections by image instead.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
# Lock the claims made in the prose above about the Parity cells.
assert parity.factors["status"].to_list() == ["inconclusive"]
assert parity.cells.filter(pl.col("status") == "too_thin").height == 4
assert parity.cells.filter(pl.col("status") == "no_finding").height == 6
assert not parity.cells["is_actionable"].any()
assert (parity.group_source, parity.test) == ("rows", "pearson")

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
# ## Conclusion
#
# You created a dataset in which class, weather and altitude are correlated, and pivoted the class axis to weather. Against the weather
# axis, you saw that the altitude imbalance the class view attributes to trucks follows the weather. You evaluated the factors per image,
# listed the combinations of weather and altitude that the data never covers, tested the combinations that exist with `Parity`, checked
# which axis a result used, and stratified a split on it.
#

# %% [markdown]
# ## Next steps
#
# - [How to control and reuse a factor's binning](./h2_control_factor_binning.py) — Declare explicit bin edges for consistent factor levels across collections.
# - [How to detect undersampled data subsets](./h2_detect_undersampling.py) — Measure coverage across custom groups and
#   classes.
# - [Dataset Bias and Coverage](../concepts/DatasetBias.md) — Learn about mutual information, parity, and diversity metrics.

# %% [markdown]
# ## On your own
#
# To run the same analysis on your own dataset, build a `Metadata` for it that holds the environmental factors you suspect. Compare
# `Balance().evaluate(metadata)` with `Balance().evaluate(metadata.classed_by("<factor>"))` for each suspect factor. Use
# `metadata.at("unit")` when the factor is recorded per image and your dataset has several detections per image. Pass several factor
# names to `classed_by` to list the combinations that your data never covers.
