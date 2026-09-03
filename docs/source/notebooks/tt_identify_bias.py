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
# This guide provides a beginner friendly introduction to dataset bias, including [balance](../concepts/DatasetBias.md#measuring-bias-normalized-mutual-information)
# and [diversity](../concepts/DatasetBias.md#measuring-bias-normalized-mutual-information).
#
# Estimated time to complete: 15 minutes
#
# Relevant ML stages: [Data Engineering](../getting-started/roles/ML_Lifecycle.md#data-engineering)
#
# Relevant personas: [Data Engineer](../getting-started/roles/data_engineer.md), [T&E Engineer](../getting-started/roles/te_engineer.md)

# %% [markdown]
# ## What you'll do
#
# - Use DataEval to identify bias and correlations in the [SeaDronesSee dataset](https://seadronessee.cs.uni-tuebingen.de/)
# - Check what the default binning produced, and see which columns DataEval could not read
# - Declare how those columns are read, and declare the bins for every factor
# - Analyze the results using plots and tables

# %% [markdown]
# ## What you'll learn
#
# - You will see how to identify bias and correlations present in a dataset.
# - You will understand the potential impact on your data and ways to mitigate them.
# - You will learn to repair a column DataEval held back, rather than accepting the drop.

# %% [markdown]
# ## What you'll need
#
# - Basic familiarity with Python
# - Basic understanding of your dataset structure, including but not limited to its
#   [metadata](../concepts/ActingOnResults.md#diagnosing-findings-with-metadata)
# - An environment with DataEval installed

# %% [markdown]
# ## Introduction
#
# Identifying any biases or correlations present in a dataset is essential to accurately interpreting your model's
# performance and its ability to generalize to new data. A common cause of poor generalization is shortcut learning —
# where a model uses secondary or background information to make predictions — which is enabled or exacerbated by dataset
# sampling biases.
#
# ### Bias and correlations
#
# Understanding biases or correlations present in your dataset is a key component to creating meaningful data splits. Bias
# in data can lead to misleading conclusions and poor model performance on operational data. There are many different
# [types of bias](https://arxiv.org/abs/1908.09635). A few of these biases occur during data collection, others occur
# during dataset development, others occur during model development, while others are a result of the user.
#
# Not all forms of bias directly affect the dataset and in order to address the biases that do, you have to make a few
# assumptions:
#
# 1. All desired classes are present.
# 1. All available metadata is provided.
# 1. The metadata has been recorded correctly.
#
# If any of the above assumptions are violated, then the analysis will not be accurate. When using your own data, you
# should verify the above assumptions.
#
# This guide does not focus on eliminating all bias, rather it focuses on identifying the bias that can be found when
# developing a dataset.
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
# These techniques help ensure that when you split the data for your projects, you minimize things like shortcut learning
# and leakage between training and testing sets.

# %% [markdown]
# ## Importing the necessary libraries
#
# You'll begin by importing the necessary libraries to walk through this guide.

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
from IPython.display import display
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
# You are going to work with the SeaDronesSee object detection dataset. This dataset is a UAV dataset aimed at helping develop
# systems for Search and Rescue using UAVs in maritime scenarios. It was used for a computer vision competition in 2023.
# This dataset was chosen because it has multiple classes, imagery collected from multiple UAVs, and the dataset contains metadata.
#
# If this data is already on your computer you can change the file location from `"./data"` to wherever the data is
# stored. If not, make sure you set the `root` path to be wherever you would like the dataset to be downloaded.
#
# For the sake of ensuring that this tutorial runs quickly on most computers, you are going to analyze only the validation
# dataset, which is a little over 1500 images.

# %%
# Download the validation dataset and verify the size of the loaded dataset
ds = SeaDrone(root="./data", download=True, image_set="val", lazy=True)
len(ds)

# %% [markdown]
# Before moving on, verify that the above code cell printed out 1547 for the size of the dataset.
#
# This ensures that everything is working as needed for the tutorial.
#
# :::{note}
# If it didn't, make sure that you have an up-to-date version of maite-datasets (_needs to be >=0.0.17_ )
# and if you are all up-to-date and still having issues see the [contributing guide for bug reports](../getting-started/contributing.md).
# :::

# %% [markdown]
# ## Structure the metadata
#
# This guide focuses on evaluating the labels and metadata of the dataset rather than the images themselves. As each dataset
# has its own image and metadata formats, you will need to understand how your particular metadata is structured.
#
# Start by taking a look at the metadata structure of the dataset by grabbing the first item from the dataset and selecting
# just the metadata.

# %%
ds[0][2]

# %% [markdown]
# The metadata in the dataset is provided as a dictionary entry for each datum, such that the aggregated data is a
# collection of _N_ metadata dictionaries each with a nested list of _M_ objects in the image.
#
# This dataset has 19 metadata categories, and from the *object_id* category highlights that this image has 1 object in it.
# From the multiple -1 values, it appears that not every image has a value for every metadata category, which may or may not point towards a bias.
#
# Now we'll extract out the metadata for the entire dataset.
#
# To do this, we need to first determine if we need to subset our metadata
# categories by either selecting the factors to include or selecting the factors to exclude (whatever is a easier list to compile).
# To start we will leave in all of the metadata categories for the bias analysis.
# The one category that never needs a decision is _id_: it names the datum rather than describing it, so
# DataEval keeps it out of the factor space on its own (it becomes the reserved `item_id` column instead of a factor).
#
# Next, we need to determine if and how we want to bin any continuous data.
# Because we have multiple categories with -1 values, we'll let DataEval handle the binning with it's auto_bin_method.
# We can always go back and adjust if needed.

# %%
# Extract the metadata from the dataset
metadata = Metadata(ds)

# %% [markdown]
# :::{note}
# `Metadata` is unable to process nested lists - any category that is a list of lists will be ignored.
# :::

# %% [markdown]
# ### Check what DataEval did on its own
#
# Before you configure anything, you should look at what the defaults produced. Most of the metadata is
# image specific rather than object specific, so you will create an image-level metadata instance and
# check its bins. You will also check *object_id* and *object_size*, which are recorded per detection.
# To grab the binned data for those two, the binned version of a continuous column is the category name
# followed by a `↕`.

# %%
# Create the image-level metadata instance
image_metadata = metadata.at("unit")

# Check the binning
factors = image_metadata.factor_names
data = image_metadata.factor_data
print("          Name  - Raw Unique - Bin Unique - Bin Unique Counts")
for i, col in enumerate(data.T):
    unique, counts = np.unique(col, return_counts=True)
    raw_values = image_metadata.dataframe.select(factors[i])
    print(f"{factors[i]:>15} - {raw_values.n_unique():^10} - {len(unique):^10} - {len(np.unique(counts)):>5}")

# Check the target-level metadata - object_id and object_size
target_metadata = metadata.rows_at("instance")
raw_obj_id = target_metadata["object_id"]
bin_obj_id = target_metadata["object_id↕"]
print(
    f"      object_id - {raw_obj_id.n_unique():^10} - {bin_obj_id.n_unique():^10} - {bin_obj_id.value_counts()['count'].n_unique():>5}"
)
raw_obj_size = target_metadata["object_size"]
bin_obj_size = target_metadata["object_size↕"]
print(
    f"    object_size - {raw_obj_size.n_unique():^10} - {bin_obj_size.n_unique():^10} - {bin_obj_size.value_counts()['count'].n_unique():>5}"
)


# %% [markdown]
# *object_id* is a purely unique value, so it cannot contain any bias - we will exclude it. The _id_ needed no
# checking at all: as noted above, DataEval keeps item identifiers out of the factor space itself.
#
# Now to understand the binning. We would like to see close to identical numbers in our print statement above, sets that have very different numbers
# such as _frame_ with 824 and 20 inform us that the auto binning didn't do a good job. And we really need to bin the data ourselves.
# The categories which appear to be fine are _drone_, _height_, and _width_. _Storage_ appears to be categorical based on when we inspected the metadata above, so it's close enough.
#
# So let's get into creating good bins for our data. First lets view the dataframe statistics for the data to get an idea of our value ranges.

# %%
raw_data = image_metadata.rows_at("unit").select(factors)
r_desc = raw_data.describe()
os_desc = raw_obj_size.describe()
combined = r_desc.join(os_desc, on="statistic", how="full", coalesce=True).rename({"value": "object_size"})
display(combined)

# %% [markdown]
# You should read this table for the value ranges you will cut into bins.
#
# _Altitude_, *compass_heading*, *gimbal_heading*, and *gimbal_pitch* have a significant number of -1
# with a scattering of a larger range of values, so you will have to check how sparse those other values are.
#
# _Speed_, _xspeed_, _yspeed_, _zspeed_ are all close to 0, so you can separate out the -1 values and
# then break the rest into slow and fast, positive and negative groups.
#
# _Drone_, _width_, _height_ and _storage_ are fine with the default settings, and you will drop *object_id*.
#
# That leaves _frame_ and *object_size*. For _frame_, you will use the percentiles from above, every 20%,
# which gives 5 bins. You will do the same for *object_size* every 5%, which gives 20 bins.
#
# Three columns are missing from the table entirely: _latitude_, _longitude_ and *date_time*. They are not
# factors, so `factors` never named them and `describe` had nothing to report. DataEval held them back
# instead of guessing at them, and you will get all three back further down.

# %%
# Inspecting the columns that are still factors
raw_data.select([
    "altitude",
    "compass_heading",
    "gimbal_heading",
    "gimbal_pitch",
    "frame",
    "speed",
    "xspeed",
    "yspeed",
    "zspeed",
])

# %% [markdown]
# Those are the values behind the ranges above. To see the three columns that are missing, you should
# check {attr}`~.Metadata.unusable`, which says what is behind each drop and what you have to write a
# repair against.

# %%
for name, held in metadata.unusable.items():
    print(f"{name:10s} {held.reasons[0]:24s} repairable={held.repairable}  counts={dict(held.counts)}")
    for kind, values in held.distinct.items():
        print(f"{'':10s}   {kind}: {values[:3]}")

# %% [markdown]
# Each column reports why it was dropped, whether you can repair it, and the values you have to write
# that repair against.
#
# _Latitude_ and _longitude_ are held back as `mixed_types`. Each holds 25 rows of the string "N" or
# "E" alongside its numeric readings, plus a `-1` among those numbers, which is how this dataset
# records a missing reading. A column with no single type is not a factor, so DataEval sets it aside
# instead of choosing one of the two readings for you.
#
# *Date_time* is held back for a different reason. Nothing about its values disagrees - `counts`
# reports all 1547 rows as text and no mixture. Nearly every row holds a different timestamp, so the
# column names its rows instead of grouping them. It needs a vocabulary rather than a type.
#
# Note the first value in its sample: an empty string, from the rows that recorded no timestamp at
# all. You will have to say what those mean, because no reading of a timestamp covers them. The
# sample is capped, which `sampled` reports, but it always shows the values the full list would show
# first, so a value like this one cannot hide behind the cap.

# %% [markdown]
# ### Declare how the held-back columns are read
#
# You do not have to accept a drop, and you do not have to edit the dataframe yourself to undo one.
# You will use {meth}`~.Metadata.repair` to declare the reading that turns a held-back column into a
# factor. Two kinds of record cover everything this dataset needs:
#
# - {class}`~dataeval.types.Remap` replaces named values. You will use it to replace the sentinels: "N"
#   and "E" become `-2`, which keeps the two kinds of missing reading, the string and the numeric `-1`,
#   separate from each other and from a real coordinate.
# - {class}`~dataeval.types.ParseDateTime` reads text as a time. With `every="hour_of_day"` it labels
#   each row with the hour it was collected in, which gives *date_time* the vocabulary it needs. This
#   dataset was collected over a couple of days, so time of day is the most useful period to group by.
#
# You should remap the empty *date_time* strings to `-1` first, matching how the rest of this dataset
# records a missing reading. `ParseDateTime` then leaves that number alone and reads only the timestamps.
#
# You can use the same approach for the `-1` in _xspeed_ and _yspeed_. Those sit in the middle of the
# real speed range, so no cut separates them from a genuine slow negative reading. Remapping them to
# `-99` moves them clear, and a bin edge below it gives them a group of their own.
#
# A repair is a declaration, not a one-off edit to a dataframe. DataEval records it on the metadata, so
# you can read it back from {attr}`~.Metadata.repairs`, store it with {meth}`~.Metadata.save`, and apply
# it to the next dataset without deciding it again.
#
# :::{note}
# Declare these on the original `metadata` variable, not on the `image_metadata` view you have been
# reading. The view was for looking; this is the instance the analysis runs on.
# :::

# %%
# Exclude object_id - it is purely unique, so it cannot carry bias
metadata.exclude = ["object_id"]

metadata.repair([
    # Replace the string sentinels, keeping them distinct from the numeric -1
    Remap("latitude", {"N": -2.0}),
    Remap("longitude", {"E": -2.0}),
    # Give date_time a vocabulary: the hour of the day each frame was collected in
    Remap("date_time", {"": -1}),
    ParseDateTime("date_time", every="hour_of_day"),
    # Move the missing-reading code clear of the real speed range
    Remap("xspeed", {-1: -99}),
    Remap("yspeed", {-1: -99}),
])

print("still dropped:", dict(metadata.dropped_factors))

# %% [markdown]
# ### Declare the bins
#
# Nothing is held back now. A repair says what the values *are*; the bins say how they are *grouped*, so
# they are a separate decision and you can now make it for every factor in one place - including the
# three columns DataEval could not read before.
#
# The ranges you read off the statistics table earlier are what the cuts are chosen from:
#
# - *compass_heading* and *gimbal_heading* are in degrees over [0,360], so 9 bins - every 45 degrees,
#   plus the -1s.
# - *gimbal_pitch* is similar but over [0,90], so 7 bins - every 15 degrees, plus the -1s.
# - _Altitude_ has a decent spread and maxes out at ~260, so 11 bins - multiples of 26, plus the -1s.
# - _Frame_ and *object_size* use their own percentiles, every 20% and every 5%.
# - _Latitude_ and _longitude_ need three groups each: the two sentinels and the real coordinates.
# - _Xspeed_ and _yspeed_ get a bin below `-99` to hold the missing readings you moved there.

# %%
metadata.continuous_factor_bins = {
    "compass_heading": [-1, 0, 45, 90, 135, 180, 225, 270, 315, 360],
    "gimbal_heading": [-1, 0, 45, 90, 135, 180, 225, 270, 315, 360],
    "gimbal_pitch": [-1, 0, 15, 30, 45, 60, 75, 90],
    "altitude": [-1, 0, 26, 52, 78, 104, 130, 156, 182, 208, 234, 260],
    "frame": np.quantile(raw_data["frame"], np.linspace(0, 1, 6)).tolist(),
    "speed": [-1, 0, 3, 15],
    # The -99 sentinel gets the first bin; the real readings keep the cuts chosen above
    "xspeed": [-99, -15, -5, 0, 5, 15],
    "yspeed": [-99, -15, -5, 0, 5, 15],
    "zspeed": [-5, -0.0001, 0.0001, 5],
    # Two sentinels and the real coordinates: three groups each
    "latitude": [-2, -1, 0, 90],
    "longitude": [-2, -1, 0, 90],
    "object_size": np.quantile(raw_obj_size, np.linspace(0, 1, 21)).tolist(),
}

# %% [markdown]
# ### Check the repaired and binned result
#
# You should check the binning once more before you analyze anything, to confirm that every image-level
# factor groups its rows sensibly. *Object_size* is recorded per
# detection, so it is not in this table; you checked it separately above.
#
# DataEval warns that some declared cuts hold no rows - `altitude` fills 9 of its 11 bins, and
# `compass_heading` 8 of 9. That is why the counts below are lower than the bins you declared. The
# empty bins are the sentinel ranges where this dataset happens to record nothing, which is a fact
# about the data rather than an error, so you should read them as a finding and leave them alone.

# %%
repaired = metadata.at("unit")
print("          Name  - Raw Unique - Bin Unique")
for name in repaired.factor_names:
    raw_values = repaired.rows_at("unit")[name]
    binned = repaired.rows_at("unit")[f"{name}\u2195" if name in metadata.continuous_factor_bins else f"{name}#"]
    print(f"{name:>15} - {raw_values.n_unique():^10} - {binned.n_unique():^10}")

# %% [markdown]
# Now that the metadata is ready to go, you can begin analyzing the dataset for bias!

# %% [markdown]
# ## Assess dataset balance

# %% [markdown]
# The {class}`.Balance` class measures correlational relationships between metadata factors and classes in a dataset. It
# analyzes the metadata factors against both the classes and other factors to identify relationships.
#
# The results can be retrieved using the _balance_ and _factors_ attributes of the output.

# %%
bal = Balance().evaluate(metadata)

# %% [markdown]
# The information provided by `Balance` may be visually understood with a heat map.

# %%
dep.plot(bal, figsize=(10, 10))

# %% [markdown]
# The heatmap has one large block in it. The flight telemetry - *gimbal_pitch*, *gimbal_heading*,
# *compass_heading*, _altitude_, _latitude_, _longitude_ and _speed_ - is correlated with almost
# everything, at 0.94 to 0.99 within the block. Nothing else comes close, and *object_size* is
# correlated with nothing at all.
#
# You should not read that block as physics. Those columns share their **missing rows**: the same 454
# frames, 29% of the dataset, record no altitude, no gimbal pitch and no coordinates, and *compass_heading*,
# *gimbal_heading* and _speed_ are unrecorded on almost exactly the same rows. A factor that is
# "unrecorded here, unrecorded there" tracks every other factor with the same gaps, so `altitude` at 0.99
# with `latitude` mostly says the telemetry failed together, not that altitude predicts position. You can
# see this because you gave the sentinels their own bins; left as ordinary numbers they would have been
# mixed in with real readings and this structure would not have been visible.
#
# _Latitude_ and _longitude_ sit at 1.00 for a related reason. You cut each into three groups - two kinds
# of missing reading and one group holding every real coordinate - so both columns now carry the same
# information, which is whether a position was recorded and how it failed. They are one column twice.
#
# _Drone_ and *storage* are also at 1.00, and *storage* and *date_time* at 0.99. *Storage* names the folder
# a clip came from, so it is standing in for the collection session: which airframe flew, and when.
#
# The two correlations against *class_label* are the ones that matter, and they are the two largest in
# that row:
#
# - **_Storage_ at 0.34.** Which session an image came from predicts what is in it, so the sessions were
#   not balanced across classes. That is a collection problem rather than a property of the imagery.
# - **_Object_size_ at 0.29.** This is bounding box bias - a model can learn the class from the size of the
#   object. It carries more weight than the number alone suggests, because *object_size* is the one factor
#   correlated with nothing else: it has no other factor to borrow the signal from.
#
# Every other factor falls to 0.14 or below.
#
# Let's investigate this further to see if these biases hold across all classes or are concentrated in a few.

# %%
dep.plot(bal, plot_classwise=True, figsize=(12, 5))

# %% [markdown]
# The classwise heatmap splits each of those findings by class.
#
# The bounding box bias sits mainly in _boat_ (0.33), _buoy_ (0.28) and _swimmer_ (0.25). Those are the
# classes whose objects appear at a narrow range of sizes.
#
# The session bias sits mainly in *life_saving_appliances* (0.37) and _buoy_ (0.36), which are the two
# flagged as imbalanced. Those classes were captured in a subset of the sessions rather than across all
# of them, so a model can pick them out from whatever the session has in common - the airframe, the
# resolution, the light - instead of from the object.
#
# *Date_time*, which you repaired into the hour of the day, tells the same story more narrowly: it reaches
# 0.28 on *life_saving_appliances* and falls to 0.12 or below on everything else. That class was largely
# collected at particular times, so a model could guess it from brightness or contrast rather than from
# the object itself.
#
# To fix the bounding box bias, collect more images at different distances from the object, so that each
# object appears at a range of sizes.
# To fix the session bias, collect every class across every session rather than concentrating a class in a
# few, and collect *life_saving_appliances* and _jetski_ at varying times of day.
#
# Next, let's assess if there is any additional bias by analyzing the datasets diversity.

# %% [markdown]
# ## Assess dataset diversity

# %% [markdown]
# The {class}`.Diversity` evaluator measures the evenness or uniformity of the sampling of metadata factors over a
# dataset. Values near 1 indicate uniform sampling, while values near 0 indicate imbalanced sampling, e.g. all values
# taking a single value. For more information see the [Diversity](../concepts/DatasetBias.md) concept page.
#
# The results can be retrieved using the _diversity_index_ attribute of the output.

# %%
div = Diversity().evaluate(metadata)

# %% [markdown]
# It's often easiest to see the differences between the different factors when visualizing them using a bar chart
# to show the factor-class analysis.

# %%
dep.plot(div, figsize=(10, 6))

# %% [markdown]
# In the results above, there are many factors that have values over 0.5 indicating a small potential for bias,
# and _speed_ and *object_size* have values near 1, meaning that there is relatively little or no sampling bias in these factors.
#
# The categories of most interest are those that are between 0.4 and 0.1 because this region represents skewed value
# distributions for the factor.
#
# The following factors fall into this category:
#
# - *class_label*
# - _altitude_
# - _height_
# - _width_
# - _xspeed_
# - _latitude_ and _longitude_
#
# The last three appear because of how you repaired them. Each is mostly one value with a small sentinel
# group beside it, which is a skewed distribution, so the evaluator reports it as one.
#
# These factors contain sampling bias which means that there is significantly more of one value in that category than others.
# For instance, the *class_label* factor highlights that there is unevenness in the number of data points per class.
#
# Whether you need to do anything about the low values in _height_ and _width_, depend on many factors including
# how you perform your pre-processing steps before the model sees the image and whether or not the less common image sizes
# contain only a select number of classes instead of all classes.
# The low value in _altitude_ explains a little why there is a large correlation between *class_label* and *object_size* -
# there isn't very much variety in altitude.
#
# The diversity function also analyzes metadata factors by individual classes to assess
# uniformity of metadata factors within a class. This can help identify specific class biases which may occur.
# You can visualize the classwise results by setting the `plot_classwise` parameter to True.

# %%
dep.plot(div, plot_classwise=True, figsize=(12, 5))

# %% [markdown]
# These results expand the above results on a classwise basis.
#
# Things to look for here are large variances for a given factor across the different classes. For example, _drone_ has
# values ranging from 0.55 to 0.10, which means that most of the images of _buoy_ and *life_saving_appliances* were
# taken with a specific drone, while _swimmer_ was spread out over the different types of drone. This can result in subtle
# differences in the images that a model can pick up leading to an opportunity for shortcut learning.
#
# The other categories with concerning diversity values are *object_size* and _altitude_ (which were discussed above), and
# *gimbal_pitch* with that low 0.17 value for *life_saving_appliances* meaning that the images have a near constant perspective of this class.
#
# Due to the large number of missing metadata in many of the other categories, its hard to say how much the different
# aspects of the categories could be contributing to bias and shortcut learning.

# %% [markdown]
# ## Conclusion

# %% [markdown]
# Having analyzed the dataset for bias with multiple metrics, the conclusion is that this dataset has bias. Training a
# model on this dataset has the potential to learn shortcuts and underperform on operational data if the biases are not
# representative of biases in the operational dataset. It also means that a model trained on this dataset, isn't going
# to generalize very well.
#
# You found three things worth acting on:
#
# - **Session bias.** *Storage* is the strongest predictor of class, and it stands in for the collection
#   session. *Life_saving_appliances* and _buoy_ were captured in a subset of the sessions.
# - **Bounding box bias.** *Object_size* is the second strongest, and it is the only factor that borrows
#   no signal from any other, which makes it the cleanest shortcut on offer.
# - **Missing telemetry.** 29% of frames record no altitude, gimbal pitch or position, and they are the
#   same frames each time. Any conclusion drawn from those factors describes the recorded subset only.
#
# The metadata categories identified by the `Balance` and `Diversity` evaluators contain issues such as imbalanced classes
# and imbalanced parameters per class. DataEval isn't able to tell you exactly why they are imbalanced, but it highlights
# the categories that you need to check.
#
# As you can see, the DataEval methods are here to help you gain a deep understanding of your dataset and all of its
# strengths and limitations. It is designed to help you create representative and reliable datasets.
#
# Good luck with your data!

# %% [markdown]
# ## When a declaration is not enough
#
# Every correction above was a declaration - a record that DataEval applies, stores and replays for you.
# That covers most of what a dataset needs, but not everything. A factor derived from two other columns
# has no record that describes it.
#
# For those cases, DataEval accepts anything that satisfies the {class}`.CodedMetadataLike` protocol. You
# can bin the data yourself and pass the result to the same evaluators. You should use this only when a
# repair cannot express what you need, because a declaration reapplies itself to the next dataset and a
# hand-built array does not.

# %% [markdown]
# The shell needs five members: the binned factor array, the class labels, the label lookup, the factor
# names, and which factors are binned. The example below builds a factor from two others - the ratio of an
# object's size to the altitude it was seen from - and passes it to the same evaluator you used above.


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

# A derived factor: apparent size per metre of altitude, cut into 5 groups. Rows with no altitude
# reading (-1) take a bin of their own rather than a ratio that would not mean anything.
ratio = np.where(rows["altitude"].to_numpy() > 0, rows["object_size"].to_numpy() / rows["altitude"].to_numpy(), -1.0)
edges = np.quantile(ratio[ratio > 0], np.linspace(0, 1, 5))
derived = np.where(ratio < 0, 0, np.digitize(ratio, edges))

custom_metadata = AdjustedMetadata(
    factors=np.column_stack([derived, rows["object_size\u2195"].to_numpy()]),
    labels=rows["class_label"].to_numpy().squeeze(),
    index2label=ds.index2label,
    names=["size_per_metre", "object_size"],
    binned=[True, True],
)

Balance().evaluate(custom_metadata)

# %% [markdown]
# ## Next steps
#
# - [Dataset bias concepts](../concepts/DatasetBias.md) — Learn about normalized mutual information and diversity indices for measuring dataset bias.
# - [Clean dataset tutorial](./tt_clean_dataset.py) — Identify duplicate, corrupt, and low-quality samples to prepare a dataset for model training.
# - [Assess data space tutorial](./tt_assess_data_space.py) — Identify coverage gaps, undersampled clusters, and outliers in dataset representations.
# - [Monitor shift tutorial](./tt_monitor_shift.py) — Monitor feature distributions over time to detect operational data drift.
# - [Add intrinsic factors](./h2_add_intrinsic_factors.py) — Apply statistical image metrics as intrinsic factors in dataset metadata.
# - [Measure label independence](./h2_measure_label_independence.py) — Compare label distributions between two datasets to assess class representation.
# - [Detect undersampling](./h2_detect_undersampling.py) — Identify undersampled subsets and rare feature combinations in dataset metadata.

# %% [markdown]
# ## On your own
#
# Once you are familiar with DataEval and dataset analysis, you will want to run this analysis on your own dataset. When
# you do, make sure that you analyze all of your data and not just the training set.
