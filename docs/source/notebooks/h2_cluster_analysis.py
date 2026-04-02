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
# # How to run clustering analysis

# %% [markdown]
# ## Problem statement
#
# Data does not typically come labeled and labeling/verifying labels is a time and resource intensive process. Exploratory
# data analysis (EDA) can often be enhanced by splitting data into similar groups.
#
# [Clustering](../concepts/Clustering.md) is a method which groups data in the format of (samples, features). This can be
# used with images or image embeddings as long as the arrays are flattened to only contain 2 dimensions.
#
# The `cluster` function utilizes a clustering algorithm based on the HDBSCAN algorithm. The `Outliers` and `Duplicates`
# detectors can then analyze the cluster results to identify outliers and duplicates.

# %% [markdown]
# ### When to use
#
# The clustering workflow can be used during the EDA process to perform the following:
#
# - group a dataset into clusters
# - verify labeling as a quality control
# - identify [outliers](../concepts/DataIntegrity.md#outlier-detection-image-statistics-and-embeddings) in your dataset
#   using the `Outliers` detector
# - identify duplicates in your dataset using the `Duplicates` detector

# %% [markdown]
# ### What you will need
#
# 1. A 2 dimensional dataset (samples, features)
# 1. A Python environment with the following packages installed:
#    - `dataeval`
#    - `matplotlib`
#
# This could be a set of flattened images or image embeddings. We recommend using image embeddings (with the feature
# dimension being \<=1000).

# %% [markdown]
# ## Getting started
#
# Let's import the required libraries needed to set up a minimal working example.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval
except Exception:
    pass

# %%
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as dsets

from dataeval.core import cluster
from dataeval.quality import Duplicates, Outliers

# %% [markdown]
# ## Loading in data
#
# For the purposes of this demonstration, we are just going to create a generic set of blobs for clustering.
#
# This is to help show all of the clustering functionality in one how-to.

# %%
# Creating 5 clusters
blobs = dsets.make_blobs(
    n_samples=100,
    centers=np.asarray([(-1.5, 1.8), (-1, 3), (0.8, 2.1), (2.8, 1.5), (2.5, 3.5)]),
    cluster_std=0.3,
    random_state=33,
)
test_data, labels = np.asarray(blobs[0]), np.asarray(blobs[1])

# %% [markdown]
# Because the clustering result can be used to detect duplicate and outlier data, we are going to modify the dataset to
# contain a few duplicate datapoints and an outlier.

# %%
test_data[71] = [1, 5]
test_data[79] = test_data[24]
test_data[63] = test_data[58] + 1e-5
labels[79] = labels[24]
labels[63] = labels[58]

# %% [markdown]
# ## Visualizing the clusters

# %%
# Mapping from labels to colors
label_to_color = np.array(["b", "r", "g", "y", "m", "gray"])

# Translate labels to colors using vectorized operation
color_array = label_to_color[labels]

# Set plotting parameters
plot_kwds = {"alpha": 0.5, "s": 50, "linewidths": 0}

# Create scatter plot
plt.scatter(test_data.T[0], test_data.T[1], c=color_array, **plot_kwds)

# Annotate each point in the scatter plot
for i, (x, y) in enumerate(test_data):
    plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0, 1), ha="center")

# %%
# Verify the number of datapoints and that the shape is 2 dimensional
print("Number of samples: ", len(test_data))
print("Array shape:", test_data.ndim)

# %% [markdown]
# ## Cluster the data
#
# We are now ready to cluster the data and inspect the results.\
# There are two different clustering methods, "kmeans" and "hdbscan". These are selected via the _algorithm_ parameter,
# with "hdbscan" being the default.

# %%
# Evaluate the clusters
clusters = cluster(test_data, algorithm="hdbscan", n_clusters=5)

# %%
clusters["clusters"]

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert clusters["clusters"].max() == 4
assert clusters["clusters"].min() == -1

# %% [markdown]
# ### Visualize the resulting clusters

# %%
# Using the same plotting as above
color_array = label_to_color[clusters["clusters"]]
plt.scatter(test_data.T[0], test_data.T[1], c=color_array, **plot_kwds)

# Annotate each point in the scatter plot
for i, (x, y) in enumerate(test_data):
    plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0, 1), ha="center")

# %% [markdown]
# ## Results
#
# We can list out each category followed by the number of items in the category and then display those items on the line
# below.
#
# For the outlier results, the clusterer provides a list of all points that it found to be an outlier.
#
# For the duplicates results, the clusterer provides a list of sets of points which it identified as near duplicates.

# %%
# Show results using the new detector classes
duplicates_detector = Duplicates(cluster_sensitivity=0.1)
duplicates_result = duplicates_detector.from_clusters(clusters)
print("near duplicates: ", duplicates_result.near)

outliers_detector = Outliers()
outliers_result = outliers_detector.from_clusters(test_data, clusters, cluster_threshold=3)
print("outliers: ", outliers_result.outliers)

# %% [markdown]
# We can see that there was one outlier and there are also 2 sets of near duplicates (the intentionally duplicated
# points).

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(outliers_result.outliers) == 1
assert len(duplicates_result.near) == 2

# %% [markdown]
# ## Related concepts
#
# - [Dataset Bias and Coverage](../concepts/DatasetBias.md)
# - [Embeddings](../concepts/Embeddings.md)
# - [Acting on Results](../concepts/ActingOnResults.md)
# - [Data Integrity](../concepts/DataIntegrity.md)
# - [Distribution Shift](../concepts/DistributionShift.md)
#
# ## See also
#
# ### How-to guides
#
# - [How to identify duplicates](./h2_deduplicate.py)
# - [How to visualize cleaning issues](./h2_visualize_cleaning_issues.py)
#
# ### Tutorials
#
# - [Assess an unlabeled data space](./tt_assess_data_space.py)
