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
# # How to add intrinsic factors to Metadata

# %% [markdown]
# ## Problem statement
#
# When performing analysis on datasets,
# [metadata](../concepts/DatasetBias.md#measuring-bias-normalized-mutual-information) may
# sometimes be sparse or unavailable. Adding metadata to a dataset for analysis
# may be necessary at times, and can come in the forms of calculated intrinsic
# values or additional information originally unavailable on the source dataset.
#
# This guide will show you how to add in the calculated statistics from DataEval's
# {func}`.compute_stats` function to the metadata for bias analysis.

# %% [markdown]
# ### When to use
#
# Adding metadata factors should be done when little or no metadata is available on the dataset, or to gain insights
# specific to metadata of interest that is not present natively in the dataset metadata.

# %% [markdown]
# ### What you will need
#
# 1. A dataset to analyze
# 1. A Python environment with the following packages installed:
#    - `dataeval`
#    - `dataeval-plots[plotly]`
#    - `maite-datasets`

# %% [markdown]
# ## Getting started
#
# First import the required libraries needed to set up the example.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval dataeval-plots[plotly] maite-datasets
except Exception:
    pass

# %%
import dataeval_plots as dep
import plotly.io as pio
import polars as pl
from maite_datasets.image_classification import CIFAR10

from dataeval import Metadata
from dataeval.bias import Balance, Diversity
from dataeval.core import compute_stats
from dataeval.flags import ImageStats
from dataeval.selection import Limit, Select, Shuffle

_ = pl.Config.set_tbl_rows(-1)
# Use plotly to render plots
dep.set_default_backend("plotly")
dep.set_default_backend("matplotlib")  # LOL BUMP BUMP

# Use the notebook renderer so JS is embedded
pio.renderers.default = "notebook"

# %% [markdown]
# ## Load the dataset
#
# Begin by loading in the CIFAR-10 dataset.
#
# The CIFAR-10 dataset contains 60,000 images - 50,000 in the train set and 10,000 in the test set. We will use a shuffled
# sample of 20,000 images from both sets.

# %%
# Load in the CIFAR10 dataset and limit to 20,000 images with random shuffling
cifar10 = Select(CIFAR10("data", image_set="base", download=True), [Limit(20000), Shuffle(seed=0)])
print(cifar10)

# %% [markdown]
# ## Inspect the metadata
#
# You can begin by inspecting the available factor names in the dataset.

# %%
metadata = Metadata(cifar10)
print(f"Factor names: {metadata.factor_names}")

# %% [markdown]
# A quick check of the {func}`.balance` of the single factor will show no mutual information between the classes and the
# `batch_num` which indicates the on-disk binary file the image was extracted from.

# %%
# Balance at index 0 is always class
Balance().evaluate(metadata).balance[2]

# %% [markdown]
# ## Add image statistics to the metadata
#
# In order to perform additional bias analysis on the dataset when no meaningful metadata are provided, you will augment
# the metadata with statistics of the images using the {func}`.compute_stats` function.
#
# Begin by running `compute_stats` for the `PIXEL` and `VISUAL` image stats for the dataset and adding the stats factors
# to the `Metadata`.

# %%
# Calculate pixel and visual statistics
calc_results = compute_stats(cifar10, stats=ImageStats.PIXEL | ImageStats.VISUAL)

# Append the factors to the metadata
metadata.add_factors(calc_results["stats"])

# %% [markdown]
# Next you will add the `compute_stats` output to the metadata as factors, and exclude factors that are uniform or without
# significance.
#
# Additionally, you will specify a binning strategy for continuous statistical factors, which are, for our purposes,
# continuous. For this example, bin everything into 10 uniform-width bins.

# %%
# Exclude the id and batch_num as it is not a relevant factor for bias analysis
metadata.exclude = ["id", "batch_num"]

# Provide binning for the continuous statistical factors using 5 uniform-width bins for each factor
keys = ("mean", "std", "var", "skew", "kurtosis", "entropy", "brightness", "darkness", "sharpness", "contrast", "zeros")
metadata.continuous_factor_bins = dict.fromkeys(keys, 5)

# %% [markdown]
# ## Perform bias analysis
#
# Now you can run the bias analysis evaluators {class}`.Balance` and {class}`.Diversity` on the dataset metadata augmented
# with intrinsic statistical factors.

# %%
balance_output = Balance().evaluate(metadata)

# %%
dep.plot(balance_output)

# %% [markdown]
# Notice the very high mutual information between the variance and standard deviation of image intensities, which is
# expected. Mean image intensity correlates with brightness, darkness, and contrast. However, none of the intrinsic
# factors correlate strongly with class label.

# %%
dep.plot(balance_output, plot_classwise=True)

# %% [markdown]
# Classwise balance also indicates minimal correlation of image statistics and individual classes. Uniform mutual
# information between individual classes and all class labels indicates balanced class representation in the subsampled
# dataset.

# %%
diversity_output = Diversity().evaluate(metadata)
dep.plot(diversity_output)

# %% [markdown]
# The diversity index also indicates uniform sampling of classes within the dataset. The apparently low diversity of
# kurtosis across the dataset may indicate an inadequate binning strategy (for metric computation) given that the other
# statistical moments appear to be more evenly distributed. Further investigation and iteration could be done to assess
# sensitivity to binning strategy.
#
# You can now augment your datasets with additional metadata information, either from additional sources or using
# `dataeval` statistical functions for insights into your data.

# %% [markdown]
# ## Related concepts
#
# - [Dataset Bias and Coverage](../concepts/DatasetBias.md)
# - [Acting on Results](../concepts/ActingOnResults.md)
#
# ## See also
#
# ### How-to guides
#
# - [How to specify custom statistics on object detection datasets](./h2_custom_image_stats_object_detection.py)
# - [How to configure global hardware configuration defaults in DataEval](../notebooks/h2_configure_hardware_settings.py)
