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
#
# When you evaluate new test data or compare two datasets, you often want a quantitative measure of the shift in
# covariates. HP [divergence](../concepts/Divergence.md) is a nonparametric measure of the distance between two
# datasets: 0 means they are approximately identically distributed, and 1 means they are completely separable.
#
# Use {func}`.divergence_fnn` or {func}`.divergence_mst` when you want to know how far apart two datasets are, for
# example to measure [operational drift](../concepts/DistributionShift.md#drift-detection).
#
# You need `dataeval`, `maite-datasets` and a set of image embeddings for each dataset. This guide builds them from the
# pixels of MNIST images.

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
# ## Load the data
#
# Load the MNIST data and create the training dataset. For the purposes of this example, you will use subsets of the
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

# %% [markdown]
# ## Calculate the initial divergence
#
# Calculate the divergence using nearest neighbor disagreements between the first 2000 images and the second 2000
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
# ## Load the corrupted data
#
# Load a corrupted MNIST dataset.

# %%
corrupted_ds = MNIST(root="./data", image_set="train", corruption="translate", download=True)
corrupted_view = View(corrupted_ds, Limit(2000))

# Reuse the extractor from above
corrupted_emb = np.asarray(Embeddings(corrupted_view, extractor=extractor, batch_size=64))

# %% [markdown]
# ## Calculate the corrupted divergence
#
# Calculate the divergence between this corrupted dataset and the original images.

# %%
div = divergence_fnn(data_a, corrupted_emb)
print(div)

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert embeddings.shape == (4000, 784)
assert corrupted_emb.shape == (2000, 784)
assert div["divergence"] > 0.95

# %% [markdown]
# The comparison shows that the translated MNIST images differ from the original images far more than two samples of the
# original images differ from each other.

# %% [markdown]
# ## Next steps
#
# - [Embeddings](../concepts/Embeddings.md) — Learn how feature embeddings represent dataset samples in low-dimensional
#   space.
# - [Distribution Shift](../concepts/DistributionShift.md) — Understand covariate shift and drift detection across
#   datasets.
# - [Monitor shifts in operational data](./tt_monitor_shift.py) — Detect distribution shift between training and
#   operational data.
