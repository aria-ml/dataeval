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
# # How to visualize cleaning issues

# %% [markdown]
# ## Problem statement
#
# Exploratory data analysis (EDA) can be overwhelming. There are so many things to check. Duplicates in your dataset,
# bad/corrupted images in the set, blurred or bright/dark images, the list goes on.
#
# DataEval created a [data cleaning](../concepts/DataIntegrity.md) class to assist you with your EDA so you can start
# training your models on high quality data.

# %% [markdown]
# ### When to use
#
# The cleaning class should be used during the initial EDA process or if you are trying to verify that you have the right
# data in your dataset.

# %% [markdown]
# ### What you will need
#
# 1. A dataset to analyze
# 1. A Python environment with the following packages installed:
#    - `dataeval`
#    - `maite-datasets`

# %% [markdown]
# ## Getting started
#
# Let's import the required libraries needed to set up a minimal working example

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval maite-datasets
except Exception:
    pass

# %%
import polars as pl
from maite_datasets.image_classification import CIFAR10

from dataeval import Metadata
from dataeval.config import set_max_processes
from dataeval.quality import Outliers

set_max_processes(4)
_ = pl.Config.set_tbl_rows(-1)

# %% [markdown]
# ## Loading in the data
#
# We are going to start by loading in the CIFAR-10 dataset.
#
# The CIFAR-10 dataset contains 60,000 images - 50,000 in the train set and 10,000 in the test set. For the purposes of
# this demonstration, we are just going to use the test set.

# %%
# Load in the CIFAR10 dataset
testing_dataset = CIFAR10("./data", image_set="test", download=True)

# Create the metadata for the dataset
metadata = Metadata(testing_dataset)

# %% [markdown]
# ## Cleaning the dataset
#
# Now we can begin finding those images which are significantly different from the rest of the data.

# %%
# Initialize the Outliers class
outliers = Outliers()

# Evaluate the data
results = outliers.evaluate(testing_dataset)

# Also evaluate the data classwise
results_classwise = results.classwise(metadata)

# %% [markdown]
# The results are a dictionary with the keys being the image that has an issue in one of the listed properties below:
#
# - Brightness
# - Blurriness
# - Missing
# - Zero
# - Width
# - Height
# - Size
# - Aspect Ratio
# - Channels
# - Depth

# %%
print(f"Total number of images with an issue: {len(results.aggregate_by_item())}")

# %%
print(f"Total number of images with an issue (classwise): {len(results_classwise.aggregate_by_item())}")

# %%
# View issues by metric
results.aggregate_by_metric()

# %%
# View issues by metric (classwise)
results_classwise.aggregate_by_metric()

# %%
# View issues by class
results.aggregate_by_class(metadata)

# %%
# View issues by class (classwise)
results_classwise.aggregate_by_class(metadata)

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
print(results.shape[0])
assert results.shape[0] == 404

# %% [markdown]
# ## Next steps
#
# - [Data Integrity](../concepts/DataIntegrity.md) — Learn about data corruption, missing values, and dataset quality issues.
# - [Clustering](../concepts/Clustering.md) — Explore clustering concepts for grouping similar data samples.
# - [Acting on Results](../concepts/ActingOnResults.md) — Learn strategies to address dataset imbalance, coverage gaps, and bias.
# - [Dataset Bias and Coverage](../concepts/DatasetBias.md) — Understand how class imbalance and missing data impact model reliability.
# - [Introduction to data cleaning](./tt_clean_dataset.py) — Clean datasets by detecting corrupted images, outliers, and duplicates.
# - [Detecting common augmentations as duplicates](./tt_augmentation_duplicates.py) — Identify augmented or transformed duplicate images in a dataset.
# - [How to specify custom statistics on object detection datasets](./h2_custom_image_stats_object_detection.py) — Configure custom image statistics for object detection dataset analysis.
# - [How to run clustering analysis](./h2_cluster_analysis.py) — Cluster feature embeddings to inspect data grouping.
# - [How to add intrinsic factors to Metadata](./h2_add_intrinsic_factors.py) — Extract and attach intrinsic image factors to dataset metadata.
