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
# # How to identify duplicates

# %% [markdown]
# ## Problem statement
#
# One of the first steps in Exploratory Data Analysis (EDA) is to check for duplicates. Duplicates add no new information
# and can distort model training by over-emphasizing features that in appear in the duplicates.
#
# DataEval provides a Duplicates class to assist you in removing duplicates so you can start training your models on high
# quality data.

# %% [markdown]
# ### When to use
#
# The Duplicates class should be used if you need to find duplicate images in your dataset.

# %% [markdown]
# ### What you will need
#
# 1. A python envornment with following packages installed:
#    - dataeval
#    - maite-datasets
# 1. A dataset to analyze

# %% [markdown]
# ## Getting started
#
# Let's import the required libraries needed to set up a minimal working example

# %% tags=["remove_cell"]
# Google Colab Only
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval maite-datasets
except Exception:
    pass

# %%
import numpy as np
from IPython.display import display
from maite_datasets.image_classification import MNIST

from dataeval import Metadata
from dataeval.config import set_max_processes
from dataeval.quality import Duplicates
from dataeval.selection import Indices, Select

set_max_processes(4)

# %% [markdown]
# ## Loading in the data
#
# Load the MNIST data and create the dataset.
#
# The MNIST dataset contains 70,000 images - 60,000 in the train set and 10,000 in the test set. For the purposes of this
# demonstration, we are just going to use the test set.

# %%
# Load in the mnist dataset
testing_dataset = MNIST(root="./data/", image_set="test", download=True)

# Get the labels
labels = Metadata(testing_dataset).class_labels

# %% [markdown]
# Because the MNIST dataset does not contain any exact duplicates we are going to adjust the dataset to include some.

# %%
# Creating some indices to duplicate
print("Exact duplicates")
duplicates = {}
for i in [1, 2, 5, 9]:
    matching_indices = np.where(labels == i)[0]
    print(f"\t{i} - ({matching_indices[23]}, {matching_indices[78]})")
    duplicates[int(matching_indices[78])] = int(matching_indices[23])

# %%
# Create a subset with the identified duplicate indices swapped
indices_with_duplicates = [duplicates.get(i, i) for i in range(len(testing_dataset))]
duplicates_ds = Select(testing_dataset, Indices(indices_with_duplicates))

# %% [markdown]
# ## Finding the duplicates
#
# Now we are asking our Duplicates class to find the needle in the haystack. There are only 4 exact duplicates.

# %%
# Initialize the Duplicates class to begin to identify duplicate images.
identifyDuplicates = Duplicates()

# Evaluate the data
results = identifyDuplicates.evaluate(duplicates_ds)

# %% [markdown]
# The results can be viewd as a DataFrame with exact and near groups enumerated.

# %%
display(results)

# %% [markdown]
# The `Duplicates` class was able to find all 4 exact duplicates out of the 10,000 samples.
#
# It also found several sets of images that are very closely related to each other, and since we are using hand written
# digits we would expect it to find some images that were nearly identical.

# %% tags=["remove_cell"]
### TEST ASSERTION CELL ###
assert results.exact
assert len(results.exact) == len(duplicates)
for k, v in duplicates.items():
    assert [v, k] in results.exact

# %% [markdown]
# ## Related concepts
#
# - [Data Integrity](../concepts/DataIntegrity.md)
# - [Clustering](../concepts/Clustering.md)
# - [Acting on Results](../concepts/ActingOnResults.md)
# - [Dataset Bias and Coverage](../concepts/DatasetBias.md)
#
# ## See also
#
# ### How-to guides
#
# - [How to specify custom statistics on object detection datasets](./h2_custom_image_stats_object_detection.py)
# - [How to visualize cleaning issues](./h2_visualize_cleaning_issues.py)
# - [How to run clustering analysis](./h2_cluster_analysis.py)
# - [How to encode images with ONNX models](./h2_encode_with_onnx.py)
# - [How to add intrinsic factors to Metadata](./h2_add_intrinsic_factors.py)
#
# ### Tutorials
#
# - [Introduction to data cleaning](./tt_clean_dataset.py)
# - [Detecting common augmentations as duplicates](./tt_augmentation_duplicates.py)
