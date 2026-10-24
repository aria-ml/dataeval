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
# # How to measure train and test dataset divergence

# %% [markdown]
# ## Problem statement
#
# When evaluating new testing data, or comparing two datasets, we often want to have a quantitative way of comparing and
# evaluating shifts in covariates. HP [divergence](../concepts/Divergence.md) is a nonparametric divergence metric which
# gives the distance between two datasets. A divergence of 0 means that the two datasets are approximately identically
# distributed. A divergence of 1 means the two datasets are completely separable.

# %% [markdown]
# ### When to use
#
# The `divergence_fnn` or `divergence_mst` functions should be used when you would like to know how far two datasets are
# diverged for one another. For example, if you would like to measure
# [operational drift](../concepts/DistributionShift.md#drift-detection).

# %% [markdown]
# ### What you will need
#
# 1. A Python environment with the following packages installed:
#    - dataeval
#    - maite-datasets
# 1. A set of image embeddings for each dataset. This guide builds them from the pixels of MNIST images.

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
from maite_datasets.image_classification import MNIST

from dataeval import Embeddings
from dataeval.core import divergence_fnn
from dataeval.data import Limit, View
from dataeval.extractors import FlattenExtractor

# %% [markdown]
# ## Loading in data
#
# Load the MNIST data and create the training dataset. For the purposes of this example, we will use subsets of the
# training (4000) data.
#
# The embeddings here are the flattened pixels of each image, made by a {class}`.FlattenExtractor`. Flattened pixels are
# a baseline, because distances between raw pixels follow brightness and alignment more than content. They work in this
# example because MNIST images are small and share one shape. For photographs, use an extractor that wraps a model, as
# described in [How to create embeddings](./h2_create_embeddings.py).

# %%
# Load in the training mnist dataset and use the first 4000
train_ds = MNIST(root="./data/", image_set="train", download=True)
train_view = View(train_ds, Limit(4000))

# Create the extractor. It holds no state, so one instance serves every dataset below
extractor = FlattenExtractor()

# Extract the embeddings of the 4000 images
embeddings = np.asarray(Embeddings(train_view, extractor=extractor, batch_size=64))

# %%
print("Number of samples: ", len(embeddings))
print("Image shape:", embeddings[0].shape)

# %% [markdown]
# ## Calculate initial divergence
#
# Let's calculate the divergence using nearest neighbor disagreements between the first 2000 images and the second 2000
# images from this sample.

# %%
data_a = embeddings[:2000]
data_b = embeddings[2000:]

# %%
div = divergence_fnn(data_a, data_b)
print(div)

# %% [markdown]
# The two halves are samples of the same dataset, so this value is the reference for the next comparison. It is above 0.
# The nearest neighbor estimator can overestimate divergence when the samples are sparse relative to the dimensionality,
# as 2000 samples of 784 pixels are. HP divergence has no significance threshold, so
# [read a divergence against a reference](../concepts/Divergence.md#limitations) like this one.

# %% [markdown]
# ## Loading in corrupted data
#
# Now let's load in a corrupted mnist dataset.

# %%
corrupted_ds = MNIST(root="./data", image_set="train", corruption="translate", download=True)
corrupted_view = View(corrupted_ds, Limit(2000))

# Reuse the extractor from above
corrupted_emb = np.asarray(Embeddings(corrupted_view, extractor=extractor, batch_size=64))

# %%
print("Number of corrupted samples: ", len(corrupted_emb))
print("Corrupted image shape:", corrupted_emb[0].shape)

# %% [markdown]
# ## Calculate corrupted divergence
#
# Now lets calculate the divergence between this corrupted dataset and the original images

# %%
div = divergence_fnn(data_a, corrupted_emb)
print(div)

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert div["divergence"] > 0.95

# %% [markdown]
# We conclude that the translated MNIST images differ from the original images far more than two samples of the original
# images differ from each other.

# %% [markdown]
# ## Next steps
#
# ### Concepts
#
# - [Embeddings](../concepts/Embeddings.md) — Learn how feature embeddings represent dataset samples in low-dimensional space.
# - [Distribution Shift](../concepts/DistributionShift.md) — Understand covariate shift and drift detection across datasets.
# - [Clustering](../concepts/Clustering.md) — Explore cluster structure and density analysis in feature space.
#
# ### Tutorials
#
# - [Monitor shifts in operational data](./tt_monitor_shift.py) — Detect distribution shift between training and operational data.
# - [Identify out-of-distribution samples](./tt_identify_ood_samples.py) — Find anomalous or out-of-distribution samples using feature embeddings.
# - [Assess an unlabeled data space](./tt_assess_data_space.py) — Analyze unlabeled image collections using clustering and divergence metrics.
#
# ### How-to guides
#
# - [How to create embeddings](./h2_create_embeddings.py) — Create image embeddings with PyTorch, ONNX or a model-free extractor.
# - [How to detect undersampled data subsets](./h2_detect_undersampling.py) — Identify sparse or underrepresented regions in dataset feature space.
# - [How to run clustering analysis](./h2_cluster_analysis.py) — Perform k-means or MST clustering on extracted embeddings.
