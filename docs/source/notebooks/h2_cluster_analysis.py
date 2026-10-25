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
# # How to run clustering analysis
#
# Data does not typically come labeled, and labeling or verifying labels takes time and resources. Clustering groups data of
# shape (samples, features) into similar groups with an algorithm based on HDBSCAN, and {class}`.Outliers` and
# {class}`.Duplicates` can analyze the result to find outliers and near duplicates. See
# [Clustering](../concepts/Clustering.md).
#
# Use the clustering workflow during exploratory data analysis to:
#
# - group a dataset into clusters
# - verify labeling as a quality control
# - find outliers and duplicates in your dataset
#
# You need a two-dimensional array of samples and features, such as flattened images or, preferably, image embeddings with a
# feature dimension of 1000 or fewer. The example needs `dataeval` and `matplotlib`.

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
# ## Create the data
#
# In this demonstration you create a generic set of blobs for clustering. This shows all of the clustering functionality
# in one how-to.

# %%
# Creating 5 clusters
blobs = dsets.make_blobs(
    n_samples=100,
    centers=np.asarray([(-1.5, 1.8), (-1, 3), (0.8, 2.1), (2.8, 1.5), (2.5, 3.5)]),
    cluster_std=0.3,
    random_state=33,
)
test_data = np.asarray(blobs[0])

# %% [markdown]
# Because the clustering result can be used to detect duplicate and outlier data, modify the dataset to contain a few
# duplicate datapoints and an outlier.

# %%
test_data[71] = [1, 5]
test_data[79] = test_data[24]
test_data[63] = test_data[58] + 1e-5

# %% [markdown]
# ## Cluster the data
#
# Cluster the data and inspect the results. There are two clustering methods, "kmeans" and "hdbscan". You select one
# with the `algorithm` parameter, and "hdbscan" is the default.

# %%
# Evaluate the clusters
clusters = cluster(test_data, algorithm="hdbscan", n_clusters=5)

# %%
clusters["clusters"]

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(test_data) == 100
assert test_data.ndim == 2
assert clusters["clusters"].max() == 4
assert clusters["clusters"].min() == -1

# %% [markdown]
# ### Visualize the resulting clusters

# %%
# Color each point by its cluster. Cluster -1 holds the points that belong to no cluster and takes the last color.
label_to_color = np.array(["b", "r", "g", "y", "m", "gray"])
color_array = label_to_color[clusters["clusters"]]
plot_kwds = {"alpha": 0.5, "s": 50, "linewidths": 0}
plt.scatter(test_data.T[0], test_data.T[1], c=color_array, **plot_kwds)

# Annotate each point in the scatter plot
for i, (x, y) in enumerate(test_data):
    plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0, 1), ha="center")

# %% [markdown]
# ## Find outliers and duplicates
#
# For the outlier results, the clusterer provides a list of all points it found to be outliers.
#
# For the duplicate results, the clusterer provides a list of sets of points it identified as near duplicates.

# %%
# Show results using the new detector classes
duplicates_detector = Duplicates(cluster_sensitivity=0.1)
duplicates_result = duplicates_detector.from_clusters(clusters)
print("near duplicates: ", duplicates_result.near)

outliers_detector = Outliers()
outliers_result = outliers_detector.from_clusters(test_data, clusters, cluster_threshold=3)
print("outliers: ", outliers_result.outliers)

# %% [markdown]
# The results show one outlier and 2 sets of near duplicates (the intentionally duplicated points).

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(outliers_result.outliers) == 1
assert len(duplicates_result.near) == 2

# %% [markdown]
# ## Next steps
#
# - [Embeddings](../concepts/Embeddings.md) — Understand feature extraction and vector embeddings for clustering.
# - [Assess an unlabeled data space](./tt_assess_data_space.py) — Evaluate unlabeled image datasets using clustering and coverage metrics.
# - [How to identify duplicates](./h2_deduplicate.py) — Detect exact and near-duplicate images in dataset collections.
