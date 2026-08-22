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
# - Analyze the results using plots and tables

# %% [markdown]
# ## What you'll learn
#
# - You will see how to identify bias and correlations present in a dataset.
# - You will understand the potential impact on your data and ways to mitigate them.

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
# This dataset has 20 metadata categories, and from the *object_id* category highlights that this image has 1 object in it.
# From the multiple -1 values, it appears that not every image has a value for every metadata category, which may or may not point towards a bias.
#
# Now we'll extract out the metadata for the entire dataset.
#
# To do this, we need to first determine if we need to subset our metadata
# categories by either selecting the factors to include or selecting the factors to exclude (whatever is a easier list to compile).
# To start we will leave in all of the 20 metadata categories for the bias analysis.
# We could pull out _id_ and *image_id* now, but let's double check them first to make sure there are no duplicates.
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
# One other thing to note, most of the metadata is image specific not object specific, so we are going to create an image metadata class and then check it's bins.
# We'll also double check specifically the object_id and object_size metadata. To grab the binned data for the two object metadata categories, the binned version of continuous columns is the category name followed by a `↕`.

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
# We were right in that both _id_ and *image_id* are purely unique values; *object_id* is also a purely unique value. We will exclude them since they will not contain any bias.
#
# Now to understand the binning. We would like to see close to identical numbers in our print statement above, sets that have very different numbers
# such as _latitude_ with 478 and 15 inform us that the auto binning didn't do a good job. And we really need to bin the data ourselves.
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
# Ah! Look closely at _latitude_ and _longitude_ - these columns are likely a mix of numerics and strings forcing everything to be a string.
# We'll fix that in a minute, but first let's continue analyzing the rest of the columns.
#
# _Altitude_, *compass_heading*, *gimbal_heading*, and *gimbal_pitch* appear to have a significant number of -1 with a scattering of a larger range of values,
# we'll have to investigate how sparse those other values are.
#
# _Speed_, _xspeed_, _yspeed_, _zspeed_ are all pretty close to 0, so we should be able to separate out the -1 values and
# then break it down into slow and fast, positive and negative groups.
#
# As mentioned above, _drone_, _width_, _height_ and _storage_ are just fine with the default settings, and we're going to drop _id_ and *image_id*.
#
# That leaves *date_time*, _frame_ and *object_size*. *Date_time* has several easy built in ways to bin it - month, day, year, time of day - so we'll just need to choose one.
# From reading about the dataset, we learn that it was collected over just a couple of days so time of day is probably the most helpful way to bin, so we'll bin according to the hour.
# For _frame_, we'll choose something simple based on the percentiles from above, let's say every 20% which gives us 5 bins.
# We'll also do something similar for *object_size* using the percentiles but we'll bin based on every 5% which gives us 20 bins.
#
# To address the potential issues with _latitude_ and _longitude_, and to get a better understanding of some of the other categories,
# let's inspect some of the actual values.

# %%
# Inspecting the desired columns
raw_data.select([
    "latitude",
    "longitude",
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
# Ah, yes! Just what we suspected _latitude_ and _longitude_ are a mix of numerics and strings.
# From the column statistics above, it appears our strings are "N" and "E", respectively.
# And from inspecting the data here, it appears that most of the numbers are very close to each other,
# so let's try binning them by creating a -2 value for the strings and then rounding all of the actual numbers to their integers and see how many bins that gets us.
#
# It appears that *compass_heading* and *gimbal_heading* are in degrees and in the range [0,360] so we can bin into 9 bins - every 45 degrees plus the -1s.
# It appears the *gimbal_pitch* is similar to the headings and in the range [0,90], so we can bin this into 7 bins - every 15 degrees plus the -1s.
#
# _Altitude_ appears to have a decent spread and maxes out at ~260, so lets do 11 bins - multiples of 26 plus the -1s.
#
# Okay, this brings us to a slight complication, DataEval's Metadata class currently doesn't handle complicated binning strategies
# like we want to do with our strings so we will have to post-process bin those columns.
# For everything else, we can go ahead and create our bins.
#
# :::{note}
# We want to process the original metadata variable, not the image_metadata that we were looking at.
# :::

# %%
# Add date_time to the excluded columns and we'll create a new hour column in its place
metadata.exclude = ["id", "image_id", "object_id", "date_time"]
metadata.continuous_factor_bins = {
    "compass_heading": [-1, 0, 45, 90, 135, 180, 225, 270, 315, 360],
    "gimbal_heading": [-1, 0, 45, 90, 135, 180, 225, 270, 315, 360],
    "gimbal_pitch": [-1, 0, 15, 30, 45, 60, 75, 90],
    "altitude": [-1, 0, 26, 52, 78, 104, 130, 156, 182, 208, 234, 260],
    "frame": np.quantile(raw_data["frame"], np.linspace(0, 1, 6)).tolist(),
    "speed": [-1, 0, 3, 15],
    "xspeed": [-15, -5, 0, 5, 15],
    "yspeed": [-15, -5, 0, 5, 15],
    "zspeed": [-5, -0.0001, 0.0001, 5],
    "object_size": np.quantile(raw_obj_size, np.linspace(0, 1, 21)).tolist(),
}

# %% [markdown]
# Now for the post-processing. The binned version of categorical columns is the category name followed by a `#`. For example, the binned version of _latitude_ is _latitude#_.

# %%
df = metadata.dataframe

# Post-process latitude
# Fix the strings
df = df.with_columns(
    pl
    .when(pl.col("latitude") == "N")
    .then(pl.lit("-2"))
    .otherwise(pl.col("latitude"))
    .cast(pl.Float64)
    .cast(pl.Int64)
    .alias("latitude")
)
# Bin the data
df = df.with_columns((pl.col("latitude").rank("dense") - 1).cast(pl.Int64).alias("latitude#"))

# Post-process longitude
# Fix the strings
df = df.with_columns(
    pl
    .when(pl.col("longitude") == "E")
    .then(pl.lit("-2"))
    .otherwise(pl.col("longitude"))
    .cast(pl.Float64)
    .cast(pl.Int64)
    .alias("longitude")
)
# Bin the data
df = df.with_columns((pl.col("longitude").rank("dense") - 1).cast(pl.Int64).alias("longitude#"))

# Post-process date_time
# Fix the strings
df = df.with_columns(
    pl
    .when(pl.col("date_time") == "")
    .then(pl.lit(-1))
    .otherwise(pl.col("date_time").str.to_datetime(strict=False).dt.hour())
    .fill_null(-1)
    .alias("hour")
)
# Bin the data
df = df.with_columns((pl.col("hour").rank("dense") - 1).cast(pl.Int64).alias("hour#"))

# Post-process xspeed and yspeed to account for -1 values
new_max_x = pl.col("xspeed↕").max() + 1
df = df.with_columns(pl.when(pl.col("xspeed") == -1).then(new_max_x).otherwise(pl.col("xspeed↕")).alias("xspeed↕"))
new_max_y = pl.col("yspeed↕").max() + 1
df = df.with_columns(pl.when(pl.col("yspeed") == -1).then(new_max_y).otherwise(pl.col("yspeed↕")).alias("yspeed↕"))


# %% [markdown]
# Now that we have fixed the data, we need to get the data back into a state to pass to our bias functions. We'll create a minimal shell using the {class}`.CodedMetadataLike` protocol.


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


corrected_metadata = AdjustedMetadata(
    factors=df
    .filter(df["level"] == "instance")
    .select([  # Only the binned data is needed
        "altitude↕",
        "compass_heading↕",
        "drone#",
        "frame↕",
        "gimbal_heading↕",
        "gimbal_pitch↕",
        "height#",
        "hour#",
        "latitude#",
        "longitude#",
        "object_size↕",
        "speed↕",
        "storage#",
        "width#",
        "xspeed↕",
        "yspeed↕",
        "zspeed↕",
    ])
    .to_numpy(),
    labels=df.filter(df["level"] == "instance").select("class_label").to_numpy().squeeze(),
    index2label=ds.index2label,
    names=[
        "altitude",
        "compass_heading",
        "drone",
        "frame",
        "gimbal_heading",
        "gimbal_pitch",
        "height",
        "hour",
        "latitude",
        "longitude",
        "object_size",
        "speed",
        "storage",
        "width",
        "xspeed",
        "yspeed",
        "zspeed",
    ],
    binned=[
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
    ],
)

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
bal = Balance().evaluate(corrected_metadata)

# %% [markdown]
# The information provided by `Balance` may be visually understood with a heat map.

# %%
dep.plot(bal, figsize=(10, 10))

# %% [markdown]
# The heatmap shows that *storage* is highly correlated with many of the other factors, while *object_size* is only correlated with few other categories.
# The greatest correlations are between things that one might expect to be correlated _latitude_ and _longitude_,
# _drone_ and _storage_, the headings - *compass_heading* and *gimbal_heading*, and the size of the image - _height_ and _width_.
#
# However, the most important correlation to note is the correlation between *object_size* and *class_label*.
# This tells us that this dataset doesn't do a good job of ensuring that each of the class objects is seen at different distances.
# Thus, this dataset has bounding box bias - a model can learn class just by the size of the object - which will definitely lead to some shortcut learning and poor generalization.
#
# Let's investigate this further to see if this bias holds across all classes or is concentrated in a few classes.

# %%
dep.plot(bal, plot_classwise=True, figsize=(12, 5))

# %% [markdown]
# The classwise heatmap shows that the main culprits of our bounding box bias are the classes - boat, buoy, and swimmer.
#
# The heatmap also shows us that there is some correlation with _hour_ and _storage_.
# The correlation with _storage_ tells us that they probably staged the data collection to ensure that they had different setup variations during data collection.
# The correlation with _hour_ tells us that the model would be able to guess if it should look for the *life_saving_appliances* class based on internal aspects of the image such as brightness or contrast instead of the object itself.
#
# To fix the bounding box bias more images at different distances from the object, so that each object appears at different sizes, will need to be collected.
# To fix the time of day bias more images of the *life_saving_appliances* class and probably the *jetski* class should be collected at varying times of day.
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
div = Diversity().evaluate(corrected_metadata)

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
# The metadata categories identified by the `Balance` and `Diversity` evaluators contain issues such as imbalanced classes
# and imbalanced parameters per class. DataEval isn't able to tell you exactly why they are imbalanced, but it highlights
# the categories that you need to check.
#
# As you can see, the DataEval methods are here to help you gain a deep understanding of your dataset and all of its
# strengths and limitations. It is designed to help you create representative and reliable datasets.
#
# Good luck with your data!

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
