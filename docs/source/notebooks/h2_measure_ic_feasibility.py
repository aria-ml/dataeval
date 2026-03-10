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
# # How to determine image classification feasibility

# %% [markdown]
# ## Problem statement
#
# For classification machine learning tasks, there is an _inherent difficulty_ associated with signal to noise ratio in
# the images. One way of quantifying this difficulty is the
# [Bayes Error Rate](../concepts/PerformanceLimits.md#bayes-error-rate--the-irreducible-floor), or irreducable error. This
# metric tells you if it would be _feasible_ to use a given feature set to predict a target variable.
#
# DataEval has introduced a method of calculating this error rate that uses image embeddings.

# %% [markdown]
# ### When to use
#
# The `BER` metric should be used when you would like to measure the feasibility of a machine learning task. For example,
# if you have an operational accuracy requirement of 80%, and would like to know if this is feasibly achievable given the
# imagery. A low feasibility score will tell you that the problem you are trying to score cannot be solved with the
# existing data at the accuracy you desire. This in turn implies that your question does not follow a learnable pattern or
# that your data is noisy.

# %% [markdown]
# ### What you will need
#
# 1. A set of image {term}`embeddings <Embeddings>` and their corresponding class labels. This requires training an
#    autoencoder to compress the images.
# 1. A Python environment with the following packages installed:
#    - `dataeval`

# %% [markdown]
# ### Getting started
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
from maite_datasets.image_classification import MNIST

from dataeval import Embeddings, Metadata
from dataeval.config import set_seed
from dataeval.core import ber_mst
from dataeval.extractors import FlattenExtractor
from dataeval.selection import ClassBalance, ClassFilter, Limit, Select

set_seed(42)  # For reproducibility

# %% [markdown]
# ## Loading in data
#
# While you can use your own dataset, for this example we imported the `MNIST` dataset and will use it going forward. It
# was imported from the DataEval utils package.
#
# To highlight the effects of modifying the dataset on its Bayes Error Rate, we will only include a subset of 6,000 images
# and their labels for digits 1, 4, and 9

# %%
# Configure the dataset transforms
transforms = [
    lambda x: x / 255.0,  # scale to [0, 1]
    lambda x: x.astype(np.float32),  # convert to float32
]

# Load the train set of the MNIST dataset and apply transforms
train_ds = MNIST(root="./data/", image_set="train", transforms=transforms, download=True)

# Get the indices of the first 2000 samples for labels 1, 4, and 9
train_ds = Select(train_ds, selections=[Limit(6000), ClassFilter((1, 4, 9)), ClassBalance("interclass")])

# Split out the embeddings and labels
extractor = FlattenExtractor()
embeddings = Embeddings(train_ds, extractor=extractor, batch_size=64)
labels = Metadata(train_ds).class_labels

print(train_ds)

# %%
print("Number of training samples: ", len(embeddings))
print("Image shape:", embeddings.shape)
print("Label counts: ", np.unique(labels, return_counts=True))

# %% [markdown]
# We have taken a subset of the data that is only the digits 1, 4, and 9. The BER estimate requires 1 dimension, that's
# why we have flattened images. This is ok since MNIST images are small, in practice we would need to do some dimension
# reduction (autoencoder) here.
#
# We now have 6,000 flattened images of size 784. Next we can move on to evaluation of the dataset.

# %% [markdown]
# ## Evaluation
#
# Suppose we would like to build a classifier that differentiates between the handwritten digits 1, 4, and 9 with
# predetermined accuracy requirement of 99%.
#
# We will use BER to check the feasibility of the task. As the images are small, we can simple use the flattened raw pixel
# intensities to calculate BER (no embedding necessary). _Note_: This will not be the case in general.

# %%
# Evaluate the BER metric for the MNIST data with digits 1, 4, 9.
# One minus the value of this metric gives our estimate of the upper bound on accuracy.
ber_result = ber_mst(embeddings, labels)

# %%
print("The bayes error rate estimation:", ber_result["upper_bound"])

# %% tags=["remove_cell"]
### TEST ASSERTION CELL ###
assert 0.93 < 1 - ber_result["upper_bound"] < 0.96

# %% [markdown]
# The estimate of the maximum achievable accuracy is one minus the BER estimate.

# %%
print("The maximum achievable accuracy:", 1 - ber_result["upper_bound"])

# %% [markdown]
# ### Initial results
#
# The maximum achievable accuracy on a dataset of 1, 4, and 9 is about 94%. This _does not_ meet our requirement of 99%
# accuracy!

# %% [markdown]
# ## Modify dataset classification
#
# To address insufficient accuracy, lets modify the dataset to classify an image as "1" or "Not a 1". By combining
# classes, we can hopefully achieve the desired level of attainable accuracy.

# %%
# Creates a binary mask where current label == 1 that can be used as the new labels
labels_merged = labels == 1
print("New label counts:", np.unique(labels_merged, return_counts=True))

# %%
# Evaluate the BER metric for the MNIST data with updated labels
new_result = ber_mst(embeddings, labels_merged)

# %%
print("The bayes error rate estimation:", new_result["upper_bound"])

# %% tags=["remove_cell"]
### TEST ASSERTION CELL ###
assert 0.99 < 1 - new_result["upper_bound"] < 0.995

# %% [markdown]
# The estimate of the maximum achievable accuracy is one minus the BER estimate.

# %%
print("The maximum achievable accuracy:", 1 - new_result["upper_bound"])

# %% [markdown]
# ### Modified results
#
# The maximum achievable accuracy on a dataset of 1 and not 1 (4, 9) is about 99%. This _does_ meet our accuracy
# requirement.
#
# By using BER to check for feasibility early on, we were able to reformulate the problem such that it is feasible under
# our specifications

# %% [markdown]
# ## Related concepts
#
# - [Performance Limits](../concepts/PerformanceLimits.md)
# - [Embeddings](../concepts/Embeddings.md)
# - [Dataset Bias and Coverage](../concepts/DatasetBias.md)
#
# ## See also
#
# ### How-to guides
#
# - [How to measure dataset sufficiency for image classification](./h2_measure_ic_sufficiency.py)
# - [How to detect undersampled data subsets](./h2_detect_undersampling.py)
# - [How to configure global hardware configuration defaults in DataEval](../notebooks/h2_configure_hardware_settings.py)
