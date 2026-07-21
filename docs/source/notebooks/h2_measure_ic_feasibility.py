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
# The `ber_mst` or `ber_knn` functions should be used when you would like to measure the feasibility of a machine learning
# task. For example, if you have an operational accuracy requirement of 80%, and would like to know if this is feasibly
# achievable given the imagery. A low feasibility score will tell you that the problem you are trying to score cannot be
# solved with the existing data at the accuracy you desire. This in turn implies that your question does not follow a
# learnable pattern or that your data is noisy.

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
from dataeval.data import ClassBalance, ClassFilter, View
from dataeval.extractors import FlattenExtractor

set_seed(42)  # For reproducibility

# %% [markdown]
# ## Loading in data
#
# We use the `MNIST` handwritten-digit dataset (imported from `maite-datasets`). Suppose we need a classifier that
# reaches **99% accuracy**. We will use BER to check whether that target is _feasible_ — first for the full ten-digit
# problem, then for progressively narrower versions of the task.
#
# BER is estimated from a one-dimensional feature per image. MNIST images are small, so we flatten the raw pixels
# directly (no embedding necessary); in practice you would first reduce dimensionality with an autoencoder.

# %%
# Configure the dataset transforms
transforms = [
    lambda x: x / 255.0,  # scale to [0, 1]
    lambda x: x.astype(np.float32),  # convert to float32
]

# Load the MNIST train set and a flattening feature extractor
mnist = MNIST(root="./data/", image_set="train", transforms=transforms, download=True)
extractor = FlattenExtractor()

# %% [markdown]
# ## 1. All ten digits
#
# First we measure the ceiling for the full task: telling all ten digits apart. We take a class-balanced subset of 6,000
# images (600 per digit) so that no class dominates the estimate.

# %%
digits_all = View(mnist, operations=[ClassBalance("interclass", num_samples=6000)])
embeddings_all = Embeddings(digits_all, extractor=extractor, batch_size=64)
labels_all = Metadata(digits_all).class_labels
print("Label counts:", np.unique(labels_all, return_counts=True))

ber_all = ber_mst(embeddings_all, labels_all)
print("Maximum achievable accuracy (10 digits):", 1 - ber_all["upper_bound"])

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert 0.92 < 1 - ber_all["upper_bound"] < 0.95

# %% [markdown]
# The ceiling for the ten-digit task is only about **94%** — well short of our 99% requirement. On raw pixels several
# digits are easily confused (4/9, 3/5/8), so no classifier can reliably do better on this data.

# %% [markdown]
# ## 2. Narrow the task to three digits (1, 4, 9)
#
# If the application only needs to distinguish a few digits, the task is easier. We keep just 1, 4, and 9 and, with the
# same 6,000-image budget concentrated on fewer classes, balance to 2,000 per class.

# %%
digits_149 = View(mnist, operations=[ClassFilter((1, 4, 9)), ClassBalance("interclass", num_samples=6000)])
embeddings_149 = Embeddings(digits_149, extractor=extractor, batch_size=64)
labels_149 = Metadata(digits_149).class_labels
print("Label counts:", np.unique(labels_149, return_counts=True))

ber_149 = ber_mst(embeddings_149, labels_149)
print("Maximum achievable accuracy (1, 4, 9):", 1 - ber_149["upper_bound"])

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert 0.96 < 1 - ber_149["upper_bound"] < 0.98

# %% [markdown]
# Narrowing from ten classes to three raises the ceiling to about **97%** — a concrete demonstration that feasibility is
# a property of the _task_, not just the data. But 97% still does not meet our 99% requirement.

# %% [markdown]
# ## 3. Reformulate as a binary problem (1 vs. not-1)
#
# Suppose the real question is only "is this a 1?". We collapse the three-digit labels into a binary target and re-check
# feasibility on the same images.

# %%
labels_binary = labels_149 == 1
print("Label counts:", np.unique(labels_binary, return_counts=True))

ber_binary = ber_mst(embeddings_149, labels_binary)
print("Maximum achievable accuracy (1 vs. not-1):", 1 - ber_binary["upper_bound"])

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert 0.99 < 1 - ber_binary["upper_bound"] < 0.998

# %% [markdown]
# The binary formulation reaches about **99.5%**, which _does_ meet our requirement.

# %% [markdown]
# ## Summary
#
# BER lets us check feasibility _before_ investing in modeling. On raw-pixel MNIST:
#
# | task | classes | max achievable accuracy |
# | --- | --- | --- |
# | all digits | 10 | ~94% |
# | 1, 4, 9 | 3 | ~97% |
# | 1 vs. not-1 | 2 | ~99.5% |
#
# Only the binary formulation clears a 99% requirement. When a task is infeasible at your target accuracy, BER can guide
# you to reformulate it: narrowing the class set (or merging classes) raises the achievable ceiling.

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
# - [How to configure global DataEval defaults](../notebooks/h2_configure_defaults.py)
#
# ### Tutorials
#
# - [Introduction to data cleaning](./tt_clean_dataset.py)
