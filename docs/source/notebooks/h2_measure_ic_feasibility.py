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
# # How to determine image classification feasibility
#
# A classification requirement is reachable only if the classes are separable in your features. The Bayes error rate
# (BER) measures how much they overlap, and {func}`.ber_mst` estimates it from image embeddings. See
# [Performance Limits](../concepts/PerformanceLimits.md#bayes-error-rate--the-irreducible-floor).
#
# Use it to check an accuracy requirement against the imagery before you invest in modeling. If the estimated lower end
# of the achievable accuracy falls short of the requirement, the estimate does not show that the requirement is
# reachable. It does not prove it unreachable either, because BER bounds hold only in the limit of unlimited data.
#
# You need `dataeval` and `maite-datasets`. This guide flattens MNIST pixels into embeddings. For larger images, reduce
# the dimensionality first, for example with an autoencoder.

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
from dataeval.data import ClassBalance, Filter, View
from dataeval.extractors import FlattenExtractor

set_seed(42)  # For reproducibility

# %% [markdown]
# ## Load the data
#
# Load the MNIST training set. Suppose you need a classifier that reaches 99% accuracy. BER can check whether that
# target is feasible, first for the full ten-digit problem and then for narrower versions of the task. MNIST images are
# small, so a {class}`.FlattenExtractor` turns the raw pixels directly into embeddings.

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
# ## Estimate all ten digits
#
# Estimate the lower end of the achievable accuracy for the full task: telling all ten digits apart. Take a
# class-balanced subset of 6,000 images (600 per digit) so that no class dominates the estimate.

# %%
digits_all = View(mnist, operations=[ClassBalance("interclass", num_samples=6000)])
embeddings_all = Embeddings(digits_all, extractor=extractor, batch_size=64)
labels_all = Metadata(digits_all).class_labels
print("Label counts:", np.unique(labels_all, return_counts=True))

ber_all = ber_mst(embeddings_all, labels_all)
print("Estimated lower end of the achievable (Bayes) accuracy (10 digits):", 1 - ber_all["upper_bound"])

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert 0.92 < 1 - ber_all["upper_bound"] < 0.95

# %% [markdown]
# The estimated lower end of the achievable accuracy for the ten-digit task is about 94%, well short of the 99%
# requirement, so this estimate does not show that 99% is reachable. On raw pixels several digits are easily confused
# (4/9, 3/5/8).

# %% [markdown]
# ## Narrow the task to three digits (1, 4, 9)
#
# If the application only needs to distinguish a few digits, the task is easier. Keep just 1, 4 and 9 and, with the same
# 6,000-image budget concentrated on fewer classes, balance to 2,000 per class.

# %%
digits_149 = View(mnist, operations=[Filter(classes=(1, 4, 9)), ClassBalance("interclass", num_samples=6000)])
embeddings_149 = Embeddings(digits_149, extractor=extractor, batch_size=64)
labels_149 = Metadata(digits_149).class_labels
print("Label counts:", np.unique(labels_149, return_counts=True))

ber_149 = ber_mst(embeddings_149, labels_149)
print("Estimated lower end of the achievable (Bayes) accuracy (1, 4, 9):", 1 - ber_149["upper_bound"])

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert 0.96 < 1 - ber_149["upper_bound"] < 0.98

# %% [markdown]
# Narrowing from ten classes to three raises the estimate to about 97%, which shows that feasibility is a property of
# the task and not only of the data. But 97% still falls short of the 99% requirement.

# %% [markdown]
# ## Reformulate as a binary problem (1 vs. not-1)
#
# Suppose the real question is only "is this a 1?". Collapse the three-digit labels into a binary target and check
# feasibility again on the same images.

# %%
labels_binary = labels_149 == 1
print("Label counts:", np.unique(labels_binary, return_counts=True))

ber_binary = ber_mst(embeddings_149, labels_binary)
print("Estimated lower end of the achievable (Bayes) accuracy (1 vs. not-1):", 1 - ber_binary["upper_bound"])

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert 0.99 < 1 - ber_binary["upper_bound"] < 0.998

# %% [markdown]
# The estimate for the binary formulation reaches about 99.5%, which clears the requirement. When the estimate falls
# short of your target accuracy, narrowing the class set or merging classes raises the estimate, so reformulating the
# task can bring the requirement within reach.

# %% [markdown]
# ## Next steps
#
# - [Performance Limits](../concepts/PerformanceLimits.md) — Understand Bayes Error Rate and theoretical performance ceilings for classification tasks.
# - [Assess whether a classification requirement is reachable](./tt_assess_ic_feasibility.py) — Judge whether classification requirements are reachable, with the evidence behind each answer.
# - [How to measure dataset sufficiency for image classification](./h2_measure_ic_sufficiency.py) — Project model performance curves as a function of dataset size.
