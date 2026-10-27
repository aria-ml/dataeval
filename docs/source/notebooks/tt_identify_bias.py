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
#     display_name: dataeval-prototype (3.12.12)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Identify bias and correlations
#
# This guide provides a beginner friendly introduction to dataset bias, including
# [balance](../concepts/DatasetBias.md#measuring-bias-mutual-information-and-evenness) and
# [diversity](../concepts/DatasetBias.md#measuring-bias-mutual-information-and-evenness).
#
# Estimated time to complete: 15 minutes
#
# Relevant ML stages: [Data Engineering](../getting-started/roles/ML_Lifecycle.md#data-engineering)
#
# Relevant personas: [Data Engineer](../getting-started/roles/data_engineer.md), [T&E Engineer](../getting-started/roles/te_engineer.md)

# %% [markdown]
# ## What you will do
#
# - Use DataEval to identify bias and correlations in the
#   [SeaDronesSee dataset](https://seadronessee.cs.uni-tuebingen.de/)
# - Check which columns DataEval held back, declare how to read them and declare the bins for every factor
# - Analyze the results using plots and tables

# %% [markdown]
# ## What you will learn
#
# - You will see how to identify bias and correlations present in a dataset.
# - You will understand the potential impact on your data and ways to mitigate them.
# - You will learn to repair a column DataEval held back, rather than accepting the drop.

# %% [markdown]
# ## What you will need
#
# - Basic familiarity with Python
# - Basic understanding of your dataset structure, including but not limited to its
#   [metadata](../concepts/ActingOnResults.md#diagnosing-findings-with-metadata)
# - An environment with DataEval installed

# %% [markdown]
# ## Introduction
#
# Identifying any biases or correlations present in a dataset is essential to accurately interpreting your model's
# performance and its ability to generalize to new data. A common cause of poor generalization is shortcut learning,
# where a model uses secondary or background information to make predictions. Dataset sampling biases enable or
# exacerbate it.
#
# ### Bias and correlations
#
# Understanding biases or correlations present in your dataset is an important part of creating meaningful data splits.
# Bias in data can lead to misleading conclusions and poor model performance on operational data. There are many
# different [types of bias](https://arxiv.org/abs/1908.09635). Some biases occur during data collection, some during
# dataset development, some during model development, and some are a result of the user.
#
# Not all forms of bias directly affect the dataset. To address the biases that do, you have to make a few assumptions:
#
# 1. All desired classes are present.
# 1. All available metadata is provided.
# 1. The metadata has been recorded correctly.
#
# If any of these assumptions is violated, the analysis will not be accurate. When you use your own data, verify these
# assumptions.
#
# This guide focuses on identifying the bias that you can find when developing a dataset. It does not eliminate all
# bias.
#
# ### DataEval metrics
#
# DataEval has two dedicated classes for identifying and understanding the bias or correlations that may be present in a
# dataset: {class}`.Balance` and {class}`.Diversity`.
#
# The `Balance` evaluator measures correlational relationships between metadata factors and classes by calculating the
# mutual information between the metadata factors and the labels.
#
# The `Diversity` evaluator measures the evenness or uniformity of the sampling of metadata factors over a dataset using
# the inverse Simpson index or Shannon index.
#
# These techniques help you minimize things like shortcut learning and leakage between training and testing sets when
# you split the data for your projects.

# %% [markdown]
# ## Setup
#
# You will begin by importing the libraries that this guide uses.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval dataeval-plots[plotly] maite-datasets
except Exception:
    pass

# %%
import dataeval_plots as dep
import numpy as np
import plotly.io as pio
import polars as pl
from maite_datasets.object_detection import SeaDrone

from dataeval import Metadata
from dataeval.bias import Balance, Diversity
from dataeval.protocols import CodedMetadataLike
from dataeval.types import ParseDateTime, Remap

# Show every row of the comparison tables below rather than polars' default window - the
# rows this guide reasons about would otherwise be the ones elided.
pl.Config.set_tbl_rows(20)

# Use plotly to render plots
dep.set_default_backend("plotly")

# Use the notebook renderer so JS is embedded
pio.renderers.default = "notebook"

# %% [markdown]
# ## Load the data
#
# You are going to work with the SeaDronesSee object detection dataset, a UAV dataset aimed at helping develop systems
# for search and rescue using UAVs in maritime scenarios. It was used for a computer vision competition in 2023. This
# tutorial uses it because it has multiple classes and imagery collected from multiple UAVs, and because it contains
# metadata.
#
# If this data is already on your computer you can change the file location from `"./data"` to wherever the data is
# stored. If not, make sure you set the `root` path to be wherever you would like the dataset to be downloaded.
#
# To keep this tutorial fast on most computers, you are going to analyze only the validation dataset, which is a little
# over 1500 images.

# %%
# Download the validation dataset and verify the size of the loaded dataset
ds = SeaDrone(root="./data", download=True, image_set="val", lazy=True)
len(ds)

# %% [markdown]
# Before you move on, verify that the code cell above printed 1547 for the size of the dataset.
#
# This confirms that everything works as the tutorial needs.
#
# :::{note}
# If it did not, make sure that you have an up-to-date version of maite-datasets (it must be `>=0.0.17`), and if you are
# up to date and still have issues, see the [contributing guide for bug reports](../getting-started/contributing.md).
# :::

# %% [markdown]
# ## Prepare the metadata
#
# {class}`.Balance` and {class}`.Diversity` read the factors of a {class}`.Metadata` object. By default, `Metadata` bins
# the continuous factors on its own and sets aside the columns it cannot read as factors. Check what it did before you
# analyze, because a bad bin or a dropped column changes every result below.
#
# Start with the metadata of the first item. This dataset has 19 categories for each image, and it records a missing
# reading as `-1`. The `id` category names the item and does not describe it, so DataEval keeps it out of the factor
# space.

# %%
ds[0][2]

# %%
# Extract the metadata from the dataset
metadata = Metadata(ds)

# %% [markdown]
# :::{note}
# `Metadata` cannot process nested lists, so it ignores any category that is a list of lists.
# :::

# %% [markdown]
# ### Check what DataEval held back
#
# {attr}`~.Metadata.unusable` reports each column that DataEval set aside, the reason, whether you can repair it and the
# values you write the repair against.

# %%
for name, held in metadata.unusable.items():
    print(f"{name:10s} {held.reasons[0]:24s} repairable={held.repairable}  counts={dict(held.counts)}")
    for kind, values in held.distinct.items():
        print(f"{'':10s}   {kind}: {values[:3]}")

# %% [markdown]
# `latitude` and `longitude` are held back as `mixed_types`. Each holds 25 rows of the string "N" or "E" among its
# numbers, and a `-1` where a reading is missing. `date_time` is held back because nearly every row holds a different
# timestamp, so the column names its rows and does not group them. The first value in its sample is an empty string, a
# row with no timestamp.

# %% [markdown]
# ### Declare the repairs and bins
#
# A repair declares what the values of a column are, and a bin declares how they are grouped. DataEval records both on
# the metadata, so you can read them back, store them with {meth}`~.Metadata.save` and apply them to the next dataset.
# See [Binning](../concepts/Binning.md) and
# [How to control and reuse a factor's binning](./h2_control_factor_binning.py) for how to choose and reuse them.
#
# The repairs below do three things. {class}`~dataeval.types.Remap` marks the missing coordinates as NaN, which DataEval
# bins in a group of its own. `latitude` and `longitude` are known factors that DataEval checks as real coordinates, so
# a `-1` would pass as a position. The same record moves the `-1` of `xspeed` and `yspeed` to `-99`, clear of the real
# speeds, because the `-1` sits inside their range and no cut could separate it, and sets the empty `date_time` strings
# to `-1`. {class}`~dataeval.types.ParseDateTime` then reads the timestamps as the hour of the day, which suits a
# dataset collected over a couple of days.

# %%
# Exclude object_id - it is purely unique, so it cannot carry bias
metadata.exclude = ["object_id"]

metadata = metadata.repair([
    # Mark the missing coordinates, spelled as a letter or as -1, as missing
    Remap("latitude", {"N": np.nan, -1: np.nan}),
    Remap("longitude", {"E": np.nan, -1: np.nan}),
    # Give date_time a vocabulary: the hour of the day each frame was collected in
    Remap("date_time", {"": -1}),
    ParseDateTime("date_time", every="hour_of_day"),
    # Move the missing-reading code clear of the real speed range
    Remap("xspeed", {-1: -99}),
    Remap("yspeed", {-1: -99}),
])

print("still dropped:", dict(metadata.dropped_factors))

# %% [markdown]
# Then declare the bins. Each cut follows the range of its factor: 45 degrees for `compass_heading` and
# `gimbal_heading`, 15 degrees for `gimbal_pitch`, steps of 26 for `altitude`, and percentiles for `frame` (every 20%)
# and `object_size` (every 5%). A `-1` gets a bin of its own, and so does the `-99` of `xspeed` and `yspeed`. `latitude`
# and `longitude` need one range for the real coordinates, and the missing readings form a group of their own.

# %%
metadata.continuous_factor_bins = {
    "compass_heading": [-1, 0, 45, 90, 135, 180, 225, 270, 315, 360],
    "gimbal_heading": [-1, 0, 45, 90, 135, 180, 225, 270, 315, 360],
    "gimbal_pitch": [-1, 0, 15, 30, 45, 60, 75, 90],
    "altitude": [-1, 0, 26, 52, 78, 104, 130, 156, 182, 208, 234, 260],
    "frame": np.quantile(metadata.rows_at("unit")["frame"], np.linspace(0, 1, 6)).tolist(),
    "speed": [-1, 0, 3, 15],
    # The -99 sentinel gets the first bin; the real readings keep their own cuts
    "xspeed": [-99, -15, -5, 0, 5, 15],
    "yspeed": [-99, -15, -5, 0, 5, 15],
    "zspeed": [-5, -0.0001, 0.0001, 5],
    # The real coordinates; missing readings are grouped separately
    "latitude": [0, 90],
    "longitude": [0, 90],
    "object_size": np.quantile(metadata.rows_at("instance")["object_size"], np.linspace(0, 1, 21)).tolist(),
}

# %% [markdown]
# Check the result before you analyze. Every image-level factor should group its rows into a sensible number of bins.
# `object_size` is recorded for each detection, so it is not in this table.

# %%
repaired = metadata.at("unit")
print("          Name  - Raw Unique - Bin Unique")
for name in repaired.factor_names:
    raw_values = repaired.rows_at("unit")[name]
    binned = repaired.rows_at("unit")[f"{name}\u2195" if name in metadata.continuous_factor_bins else f"{name}#"]
    print(f"{name:>15} - {raw_values.n_unique():^10} - {binned.n_unique():^10}")

# %% [markdown]
# DataEval warns that some declared cuts hold no rows: `altitude` fills 9 of its 11 bins and `compass_heading` 8 of 9.
# The empty bins are the sentinel ranges where this dataset records nothing. That is a finding about the data and not an
# error, so leave them alone.

# %% [markdown]
# ## Assess dataset balance

# %% [markdown]
# The {class}`.Balance` class measures correlational relationships between metadata factors and classes in a dataset. It
# analyzes the metadata factors against both the classes and other factors to identify relationships.
#
# You can retrieve the results with the `balance` and `factors` attributes of the output.

# %%
bal = Balance().evaluate(metadata)

# %% [markdown]
# You can see the information that `Balance` provides in a heat map.

# %%
dep.plot(bal, figsize=(10, 10))

# %% [markdown]
# The heatmap has one large block. The flight telemetry (`gimbal_pitch`, `gimbal_heading`, `compass_heading`,
# `altitude`, `latitude`, `longitude` and `speed`) is correlated with almost everything, at 0.79 to 1.00 within the
# block, and highest wherever `latitude` or `longitude` is one of the pair. Nothing else comes close, and `object_size`
# is correlated with nothing.
#
# Do not read that block as physics. Those columns share their missing rows: the same 454 frames, 29% of the dataset,
# record no altitude, no gimbal pitch and no coordinates, and `compass_heading`, `gimbal_heading` and `speed` are
# unrecorded on almost the same rows. A factor that is unrecorded on the same rows as another tracks that factor, so
# `altitude` at 0.99 with `latitude` mostly says the telemetry failed together, not that altitude predicts position. You
# can see this because the missing readings have bins of their own. Left as ordinary numbers, they would have been mixed
# in with real readings and this structure would not have been visible.
#
# The pair `latitude` and `longitude` sits at 1.00 for a related reason. Each has one group holding every real
# coordinate and one holding the missing readings, so both columns carry the same information: whether a position was
# recorded.
#
# The pair `drone` and `storage` is also at 1.00, and `storage` and `date_time` are at 0.99. The `storage` factor names
# the folder a clip came from, so it stands in for the collection session: which airframe flew, and when.
#
# The two correlations against `class_label` matter most, and they are the two largest in that row:
#
# - `storage` at 0.34. The session an image came from predicts what is in it, so the sessions were not balanced across
#   classes. That is a collection problem rather than a property of the imagery.
# - `object_size` at 0.29. This is bounding box bias: a model can learn the class from the size of the object. It
#   carries more weight than the number alone suggests, because `object_size` is the one factor correlated with nothing
#   else, so it has no other factor to borrow the signal from.
#
# Every other factor falls to 0.14 or below.
#
# Next, check whether these biases hold across all classes or are concentrated in a few.

# %%
dep.plot(bal, plot_classwise=True, figsize=(12, 5))

# %% [markdown]
# The classwise heatmap splits each of those findings by class.
#
# The bounding box bias sits mainly in `boat` (0.33), `buoy` (0.28) and `swimmer` (0.25). Those are the classes whose
# objects appear at a narrow range of sizes.
#
# The session bias sits mainly in `life_saving_appliances` (0.37) and `buoy` (0.36), which are the two flagged as
# imbalanced. Those classes were captured in a subset of the sessions rather than across all of them, so a model can
# pick them out from whatever the session has in common (the airframe, the resolution, the light) instead of from the
# object.
#
# The `date_time` factor, which you repaired into the hour of the day, tells the same story more narrowly. It reaches
# 0.28 on `life_saving_appliances` and falls to 0.12 or below on every other class. That class was largely collected at
# particular times, so a model could guess it from brightness or contrast rather than from the object itself.
#
# To fix the bounding box bias, collect more images at different distances from the object, so that each object appears
# at a range of sizes. To fix the session bias, collect every class across every session rather than concentrating a
# class in a few, and collect `life_saving_appliances` and `jetski` at varying times of day.
#
# Next, assess whether there is any additional bias by analyzing the dataset's diversity.

# %% [markdown]
# ## Assess dataset diversity

# %% [markdown]
# The {class}`.Diversity` evaluator measures the evenness or uniformity of the sampling of metadata factors over a
# dataset. Values near 1 indicate uniform sampling, while values near 0 indicate imbalanced sampling, e.g. all values
# taking a single value. For more information, see the [Diversity](../concepts/DatasetBias.md) concept page.
#
# You can retrieve the results with the `diversity_index` attribute of the output.

# %%
div = Diversity().evaluate(metadata)

# %% [markdown]
# It is often easiest to see the differences between factors in a bar chart of the factor-class analysis.

# %%
dep.plot(div, figsize=(10, 6))

# %% [markdown]
# In the results above, many factors have values over 0.5, which indicates a small potential for bias, and `speed` and
# `object_size` have values near 1, which means that there is relatively little or no sampling bias in these factors.
#
# The factors of most interest have values between 0.4 and 0.1, because this region represents skewed value
# distributions for the factor.
#
# The following factors fall in this range:
#
# - `class_label`
# - `altitude`
# - `height`
# - `width`
#
# These factors contain sampling bias, which means that one value occurs significantly more often than the others. For
# instance, `class_label` shows that there is unevenness in the number of data points per class.
#
# Whether you need to do anything about the low values in `height` and `width` depends on many things, including how you
# perform your pre-processing steps before the model sees the image and whether the less common image sizes contain only
# some classes instead of all classes. The low value in `altitude` explains a little of why there is a large correlation
# between `class_label` and `object_size`: there is little variety in altitude.
#
# Diversity also analyzes metadata factors by individual class, to assess the uniformity of metadata factors within a
# class. This can help identify specific class biases. Set the `plot_classwise` parameter to `True` to visualize the
# classwise results.

# %%
dep.plot(div, plot_classwise=True, figsize=(12, 5))

# %% [markdown]
# These results extend the results above to each class.
#
# Look for large variances in a factor across the classes. For example, `drone` has values ranging from 0.55 to 0.10,
# which means that most of the images of `buoy` and `life_saving_appliances` were taken with a specific drone, while
# `swimmer` was spread over the different types of drone. This can result in subtle differences in the images that a
# model can pick up, which is an opportunity for shortcut learning.
#
# The other factors with concerning diversity values are `object_size`, `altitude` and `gimbal_pitch`. Overall,
# `object_size` is near 1, but the classwise plot shows it low, from 0.21 to 0.41, for every class except `swimmer`.
# The `altitude` factor was discussed above. The `gimbal_pitch` factor has a low 0.17 for `life_saving_appliances`,
# which means that the images show this class from a near constant perspective.
#
# Because many of the other factors have a large amount of missing metadata, it is hard to say how much their different
# aspects could contribute to bias and shortcut learning.

# %% [markdown]
# ## Evaluate a derived factor
#
# Every correction above was a declaration: a record that DataEval applies, stores and replays for you. That covers most
# of what a dataset needs, but not everything. A factor derived from two other columns has no record that describes it.
#
# For those cases, DataEval accepts anything that satisfies the {class}`.CodedMetadataLike` protocol. You
# can bin the data yourself and pass the result to the same evaluators. You should use this only when a
# repair cannot express what you need, because a declaration reapplies itself to the next dataset and a
# hand-built array does not.

# %% [markdown]
# The shell needs five members: the binned factor array, the class labels, the label lookup, the factor names, and which
# factors are binned. The example below builds a factor from two others, the ratio of an object's size to the altitude
# it was seen from, and passes it to the same evaluator you used above.


# %%
class AdjustedMetadata(CodedMetadataLike):
    def __init__(self, factors, labels, index2label, names, binned):
        self._factors = factors
        self._labels = labels
        self._index2label = index2label
        self._names = names
        self._binned = binned

    @property
    def factor_names(self):
        return self._names

    @property
    def factor_data(self):
        return self._factors

    @property
    def class_labels(self):
        return self._labels

    @property
    def index2label(self):
        return self._index2label

    @property
    def is_binned(self):
        return self._binned


# One row per detection, which is the level `object_size` is recorded at.
rows = metadata.rows_at("instance")

# A derived factor: apparent size per meter of altitude, cut into 5 groups. Rows with no altitude
# reading (-1) take a bin of their own rather than a ratio that would not mean anything.
ratio = np.where(rows["altitude"].to_numpy() > 0, rows["object_size"].to_numpy() / rows["altitude"].to_numpy(), -1.0)
edges = np.quantile(ratio[ratio > 0], np.linspace(0, 1, 5))
derived = np.where(ratio < 0, 0, np.digitize(ratio, edges))

custom_metadata = AdjustedMetadata(
    factors=np.column_stack([derived, rows["object_size\u2195"].to_numpy()]),
    labels=rows["class_label"].to_numpy().squeeze(),
    index2label=ds.index2label,
    names=["size_per_meter", "object_size"],
    binned=[True, True],
)

Balance().evaluate(custom_metadata)

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
# Verify the dataset and the metadata repairs that the text describes
assert len(ds) == 1547
fresh = Metadata(ds)
assert {name: held.reasons[0] for name, held in fresh.unusable.items()} == {
    "latitude": "mixed_types",
    "longitude": "mixed_types",
    "date_time": "cardinality_over_budget",
}
assert dict(fresh.unusable["latitude"].counts) == {"text": 25, "numeric": 1522}
assert not metadata.dropped_factors

# Verify the telemetry gaps: the same 454 frames, 29% of the dataset, have no altitude, gimbal pitch or coordinates
units = metadata.rows_at("unit")
missing = units["altitude"] == -1
assert missing.sum() == 454
assert round(100 * missing.sum() / units.height) == 29
assert ((units["gimbal_pitch"] == -1) == missing).all()
assert units["latitude"].is_nan().filter(missing).all()
assert units["longitude"].is_nan().filter(missing).all()

# Verify the bins that the text counts
assert repaired.rows_at("unit")["altitude\u2195"].n_unique() == 9
assert repaired.rows_at("unit")["compass_heading\u2195"].n_unique() == 8

# Verify the balance findings: the two factors that predict the class, and the classes that carry them
vs_class = {
    str(n): v for n, v in zip(bal.balance["factor_name"], bal.balance["mi_value"], strict=True) if n != "class_label"
}
ranked = sorted(vs_class, key=lambda n: -vs_class[n])
assert ranked[:2] == ["storage", "object_size"]
assert abs(vs_class["storage"] - 0.34) < 0.02
assert abs(vs_class["object_size"] - 0.29) < 0.02
assert vs_class[ranked[2]] < 0.15

by_class = {(str(c), str(f)): (v, flag) for c, f, v, flag in bal.classwise.iter_rows()}
size = sorted((c for c, f in by_class if f == "object_size"), key=lambda c: -by_class[(c, "object_size")][0])
assert size[:3] == ["boat", "buoy", "swimmer"]
assert {c for (c, f), (_, flag) in by_class.items() if f == "storage" and flag} == {"life_saving_appliances", "buoy"}
assert abs(by_class[("life_saving_appliances", "storage")][0] - 0.37) < 0.02
assert abs(by_class[("buoy", "storage")][0] - 0.36) < 0.02
assert abs(by_class[("life_saving_appliances", "date_time")][0] - 0.28) < 0.02
assert max(v for (c, f), (v, _) in by_class.items() if f == "date_time" and c != "life_saving_appliances") < 0.13

# Verify the diversity findings
diversity = {str(n): v for n, v in zip(div.factors["factor_name"], div.factors["diversity_value"], strict=True)}
assert {n for n, v in diversity.items() if 0.1 <= v <= 0.4} == {"class_label", "altitude", "height", "width"}
assert diversity["speed"] > 0.85
assert diversity["object_size"] > 0.9
by_div = {(str(c), str(f)): v for c, f, v, _ in div.classwise.iter_rows()}
drone = [v for (c, f), v in by_div.items() if f == "drone"]
assert abs(max(drone) - 0.55) < 0.02
assert abs(min(drone) - 0.10) < 0.02
assert abs(by_div[("life_saving_appliances", "gimbal_pitch")] - 0.17) < 0.02
size_diversity = {c: v for (c, f), v in by_div.items() if f == "object_size"}
assert all(0.2 < v < 0.45 for c, v in size_diversity.items() if c != "swimmer")
assert size_diversity["swimmer"] > 0.7

# %% [markdown]
# ## Conclusion
#
# You checked which columns DataEval held back, declared how to read them and how to bin every factor, and analyzed the
# SeaDronesSee dataset with {class}`.Balance` and {class}`.Diversity`. You also evaluated a factor that you derived
# yourself. The dataset shows bias that a model can learn as a shortcut, so a model trained on it could underperform on
# operational data unless the operational data shows the same biases. DataEval cannot tell you why a factor is
# imbalanced. It points to the categories that you need to check.

# %% [markdown]
# ## What you learned
#
# You found three things worth acting on:
#
# - `storage` is the strongest predictor of class, and it stands in for the collection session. This is session bias.
#   `life_saving_appliances` and `buoy` were captured in a subset of the sessions.
# - `object_size` is the second strongest, and it is the only factor that borrows no signal from any other. This is
#   bounding box bias, and it makes `object_size` the cleanest shortcut in this dataset.
# - 29% of frames record no altitude, gimbal pitch or position, and they are the same frames each time. Any conclusion
#   drawn from those factors describes the recorded subset only.

# %% [markdown]
# ## Next steps
#
# - [Dataset bias and coverage](../concepts/DatasetBias.md) — Learn about normalized mutual information and diversity
#   indices for measuring dataset bias.
# - [How to add intrinsic factors to Metadata](./h2_add_intrinsic_factors.py) — Measure brightness, sharpness and other
#   image statistics and add them to the metadata.
# - [How to measure label independence](./h2_measure_label_independence.py) — Compare label distributions between two
#   datasets to assess class representation.

# %% [markdown]
# ## On your own
#
# Once you are familiar with DataEval and dataset analysis, run this analysis on your own dataset. Analyze all of your
# data and not only the training set.
