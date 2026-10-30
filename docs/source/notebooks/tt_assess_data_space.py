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
# # Assess an unlabeled data space
#
# This tutorial is a beginner-friendly introduction to exploratory data analysis when labels are not available.
#
# Estimated time to complete: 10 minutes
#
# Relevant ML stages: [Data Engineering](../getting-started/roles/ML_Lifecycle.md#data-engineering)
#
# Relevant personas: [Data Engineer](../getting-started/roles/data_engineer.md), [Data Scientist](../getting-started/roles/data_scientist.md)

# %% [markdown]
# ## What you will do
#
# - Construct [embeddings](../concepts/Embeddings.md) by inferencing the PASCAL VOC 2012 dataset through a model
# - Analyze clustered embeddings to find
#   [outliers](../concepts/DataIntegrity.md#outlier-detection-image-statistics-and-embeddings)
# - Run additional analysis to find gaps in data
#   [coverage](../concepts/DatasetBias.md#measuring-coverage-geometry-in-embedding-space)
#
# ## What you will learn
#
# - Learn to use cluster-based algorithms to measure dataset quality without labels
# - Learn to distinguish whether an image is an outlier or is underrepresented
# - Learn to determine whether flagged images should be added or removed

# %% [markdown]
# ## What you will need
#
# - Basic familiarity with Python

# %% [markdown]
# ## Introduction
#
# Before you build predictive models, identify outliers and check that data groups accurately reflect the underlying
# distribution. When labels are unavailable, for example in unsupervised learning, initial data curation or when
# annotations are expensive, you can still analyze the images alone. To do this, treat each image as a geometric point
# in a high-dimensional abstract space called **feature space**. Once you understand how the images are distributed in
# feature space, you can make the assessments you need.
#
# Group the images into [clusters](../concepts/Clustering.md) to explore the natural structure of the dataset and reveal
# patterns and potential anomalies. Measuring coverage goes a step further: it quantifies how well the clusters
# represent the entire dataset, so that no significant portion of the feature space is overlooked.
#
# Use these techniques to evaluate the quality and representativeness of your data and to avoid biases, missing
# information and overfitting in your models. When you understand the space your data occupies and how it groups, you
# can build more reliable models that generalize well in real-world applications.

# %% [markdown]
# ## Setup
#
# Import the libraries that the example needs.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval dataeval-plots[plotly] maite-datasets
except Exception:
    pass

# %%
import dataeval_plots as dep
import numpy as np
import plotly.io as pio
import torch
from maite_datasets.object_detection import VOCDetection
from torchvision.models import ResNet18_Weights, resnet18

from dataeval import Embeddings
from dataeval.config import set_device, set_max_processes, set_seed
from dataeval.core import cluster, coverage_adaptive
from dataeval.extractors import TorchExtractor
from dataeval.quality import Outliers

# Set default device for notebook
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_device(device)
set_seed(0, all_generators=True)  # fix the random seed for reproducible results

# Use plotly to render plots
dep.set_default_backend("plotly")

# Use the notebook renderer so JS is embedded
pio.renderers.default = "notebook"

# Set the maximum number of processes to use
set_max_processes(4)

# %% [markdown]
# :::{note}
#
# The device is the piece of hardware where the model, data, and other related objects are stored in memory. If a GPU is
# available, this notebook uses it instead of the CPU. To run only on the CPU, change `device` to `"cpu"`. For more
# information, see the
# [PyTorch device page](https://pytorch.org/tutorials/recipes/recipes/changing_default_device.html).
# :::

# %% [markdown]
# ## Construct embeddings
#
# Rather than analyze raw pixels, you will work in [embedding](../concepts/Embeddings.md) space: one feature vector per
# image, with far fewer values than the image itself, which keeps the downstream algorithms fast and light on memory.
#
# ### Download VOC dataset
#
# Download the train split with `VOCDetection` from `maite_datasets`. It has 5717 images with a varying number of
# targets each. You can also run this tutorial with your own MAITE-compliant dataset.

# %%
# Load the training dataset
dataset = VOCDetection(root="./data", image_set="train", year="2012", download=True)

print(f"Images: {len(dataset)}")
print(f"Image shape: {dataset[0][0].shape}")

# %% [markdown]
# ### Extract embeddings
#
# Reduce each image to a 512-dimensional feature vector with a
# [pretrained ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) from
# Torchvision. Point a {class}`.TorchExtractor` at its penultimate `avgpool` layer (its learned features, not the
# 1000-class logits), give it the model's standard resize-and-normalize preprocessing as `transforms`, and run the
# dataset through {class}`.Embeddings`.
#
# For other ways to build embeddings, see [Create embeddings](./h2_create_embeddings.py).

# %%
resnet = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False)
extractor = TorchExtractor(resnet, transforms=ResNet18_Weights.DEFAULT.transforms(), layer_name="avgpool")
embeddings = np.asarray(Embeddings(dataset, extractor=extractor, batch_size=64))

print("embedding shape:", embeddings.shape)

# %% [markdown]
# Each image is now a single 512-dimensional feature vector rather than a full-resolution image.

# %% [markdown]
# ## Cluster the embeddings
#
# In this section, you will use the embeddings you generated to create clusters. A cluster is a group of images that
# have similar characteristics. The clusters let you find images that are either too similar or too dissimilar. Both
# cases can degrade downstream tasks such as data analysis or model training.
#
# ### Create clusters
#
# Run the embeddings through the {func}`.cluster` function to generate the {class}`.ClusterResult` mapping. The function
# needs no rescaling of the embeddings: multiplying them all by a constant does not change the clusters.

# %%
output = cluster(embeddings)

# %% [markdown]
# You have clustered your image embeddings. The `ClusterResult` contains several keys related to clusters, such as the
# minimum spanning tree and linkage array. For this tutorial, you can use the `Outliers` detector's `from_clusters`
# method to analyze the quality of the dataset. For additional use cases, additional benefits and alternative methods,
# read more [about clustering](../concepts/Clustering.md) in DataEval's documentation.

# %% [markdown]
# ## Visualize the outputs
#
# Once you have generated the clusters, you can analyze individual samples with DataEval's detector classes to identify
# potential outliers or duplicates. In this section, you will look at the flagged images and decide on corrective
# measures to improve the quality of the data.
#
# The flagged images fall into two categories:
#
# - duplicates
# - outliers
#
# Look at each of these in turn.
#
# ### Duplicates
#
# The `Duplicates` detector's `from_clusters` method can identify data points that are within certain distances from
# each other in the embedding space.
#
# This tutorial does not focus on duplicates, because no exact duplicates are found in the PASCAL 2012 VOC dataset.
#
# ### Outliers
#
# Outliers are individual data points that did not fit within the radius of any cluster. The `Outliers` detector's
# `from_clusters` method uses adaptive distance-based detection that accounts for varying cluster densities.
#
# Identify outliers because they can skew the training distribution away from the operational distribution, which can
# lead to performance degradation.
#
# Use the `Outliers` detector to see how many images it flags.

# %%
# Use the Outliers.from_clusters() method
detector = Outliers()
result = detector.from_clusters(embeddings, output, cluster_threshold=3.0)

# Extract the outlier indices from the result
outliers = result.aggregate_by_item()["item_index"].to_list()

print(f"Number of outliers: {len(outliers)}")

# %% [markdown]
# That is a reasonable number of outliers to investigate. Calculate the percent of outliers in the dataset.

# %%
print(f"Percent of outliers: {(100 * len(outliers) / len(dataset)):.2f}%")

# %% [markdown]
# ### Outlier inspection questions
#
# When you look at these images, ask yourself the following questions:
#
# - Does this image represent something that would be expected in operation?
# - Is there commonality to the objects in the images?
# - Is there commonality to the backgrounds of the images?
# - Is there commonality to the class of objects in the images?
#
# These questions help you notice patterns such as all objects being located on the left side of the image or all the
# images of a specific class having a specific background. Training a model on data that has such commonalities can
# cause the model to develop biases or limit its ability to generalize to non-training data.
#
# Choose which set of images to examine. Swap the indices in the call to `dep.plot()` by uncommenting your preferred
# indices calculation. You can also adjust the `count` parameter to see more examples.

# %%
# To use the flagged outliers, uncomment the line below
indices = outliers

# To use unflagged images, uncomment the line below
# indices = list(set(range(len(dataset))) - set(outliers))

# %% [markdown]
# Now visualize your chosen set of indices.

# %%
# Plot 6 images from the dataset
dep.plot(
    dataset,
    indices=list(indices)[:6],
    figsize=(10, 6),
    images_per_row=3,
)

# %% [markdown]
# ## Analyze the flagged images
#
# Visualization is a useful way to catch initial differences between images in a dataset. However, it is only a
# preliminary step toward identifying the root cause of an outlier. Ask a few initial questions to ensure that outliers
# are handled properly and represent the underlying distribution of the data.
#
# ### Initial questions
#
# Use this shortlist of questions on the samples above to determine the appropriate data cleaning step:
#
# 1. Are there unexpected artifacts in the image?
#    - Examples: Data collection, processing, and visualization can cause discoloration, blurriness, etc.
#    - Solution: Ensure that images are normalized with the right values, that the channel order is correct and that the
#      color scheme is consistent
#    - See the in-depth [Data Cleaning Guide](./tt_clean_dataset.py) to explore these types of outliers
# 2. Do you have enough data?
#    - Examples: Fewer data points in each cluster can lead to higher deviation and more outliers
#    - Solution: Adding more data can help make the clusters more robust to the variations in images
# 3. Do the images provide useful information?
#    - Examples: Image attributes like number of duplicates, pixel intensity, etc. can cause the clusterer to become
#      biased to the wrong information
#    - Solution: Determine whether the data represents the whole operational distribution
#    - The following section explores this solution
#
# This is only a shortlist of potential causes for outliers based on pixel values. Over time you will add more questions
# and solutions that suit your specific needs.
#
# ### Outliers and edge cases
#
# The most common way to clean data is to remove the outliers entirely. This is an easy and efficient way to stabilize
# the training distribution. However, it can lead to shortages of data, class imbalances, and a loss of potentially
# useful information. Remember that outliers and edge cases are different. Edge cases are statistically rare events that
# are relevant to the downstream task. So look at outliers carefully to ensure that they are irrelevant.
#
# One way to determine whether an image is relevant even when the total distribution considers it an outlier is to
# measure its coverage.

# %% [markdown]
# ## Measure image coverage
#
# {term}`Coverage` is the measurement of the representation of an image's variations in a feature space. When an image
# does not have enough variations, it is **underrepresented**. In this section, you will test for any gaps in the
# coverage of the dataset to find underrepresented images.
#
# ### Calculate coverage
#
# The {func}`.coverage_adaptive` function returns a list of the image indices that it finds to be underrepresented. Such
# an image does not have enough similar images around it. This may sound familiar, because outliers are in a similar
# situation. However, you should handle underrepresented images differently than outliers. The next section explains
# how.
#
# Coverage expects embeddings between 0 and 1. Passing `force_unit_interval=True` rescales them for you.

# %%
cov = coverage_adaptive(embeddings, 50, 0.01, force_unit_interval=True)

print(f"Number of uncovered images: {len(cov['uncovered_indices'])}")

# %% [markdown]
# This tutorial has no labels, so it calls the function directly. When you have a class label for each embedding,
# {class}`.Coverage` finds the same under-sampled images and adds a breakdown of how varied each class is.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
from dataeval.scope import Coverage

labeled = Coverage(num_observations=50, percent=0.01).evaluate(
    np.zeros(len(embeddings), dtype=int), embeddings=embeddings
)
assert np.array_equal(np.sort(labeled.uncovered_indices), np.sort(cov["uncovered_indices"]))

# %% [markdown]
# You can plot a few images using the same function as before.

# %%
dep.plot(dataset, indices=cov["uncovered_indices"][:3], figsize=(10, 3))

# %% [markdown]
# You have now seen a few examples of images that were flagged as uncovered. Compare this information with the outliers
# found earlier to determine which actions to take.

# %%
uncovered_outliers = set(cov["uncovered_indices"].tolist()).intersection(set(outliers))

print(f"Number of outliers found as uncovered images: {len(uncovered_outliers)}")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert embeddings.shape == (len(dataset), 512)
assert 0 < len(outliers) <= 0.05 * len(dataset)
assert len(cov["uncovered_indices"]) > 0

# %% [markdown]
# ## Handling flagged images
#
# You now have enough information to decide what to do with these images. If an image is flagged as both an outlier and
# underrepresented, handle it first as missing coverage, and then as an outlier if the solutions cannot be acted on.
#
# ### Missing coverage
#
# The characteristics of the images with a gap in coverage need additional samples. Collect more images with similar
# information (pose, scene, color, labels if available) or add augmentations.
#
# ### Handling outliers
#
# You can handle outliers with additional sampling, but they are typically removed because they either do not contain
# enough relevant information or would shift the underlying distribution if you added similar items.

# %% [markdown]
# ## Conclusion
#
# In this tutorial you created image embeddings for more efficient calculations, used cluster-based algorithms to find
# outliers, checked for gaps in coverage on PASCAL VOC imagery, and made decisions on possible solutions.
#
# ## What you learned
#
# - Applied cluster-based feature space metrics ({func}`.cluster`, {class}`.Outliers`, {func}`.coverage_adaptive`) to
#   evaluate unlabeled imagery.
# - Differentiated between corrupt/irrelevant outliers (candidates for pruning) and underrepresented operational edge
#   cases (candidates for targeted collection).
# - Combined outlier and coverage flags to choose between removing images and collecting more data.

# %% [markdown]
# ## Next steps
#
# - [Clustering](../concepts/Clustering.md) — Mathematical foundations of minimum spanning tree linkage clustering.
# - [Measuring coverage: geometry in embedding
#   space](../concepts/DatasetBias.md#measuring-coverage-geometry-in-embedding-space) — Read how naive and adaptive
#   coverage find the images in sparse regions of the space.
# - [Run clustering analysis](./h2_cluster_analysis.py) — Cluster a dataset and find outliers and near duplicates
#   in the clusters.
# - [Detect undersampled data subsets](./h2_detect_undersampling.py) — Find the regions of a dataset that few
#   images cover.

# %% [markdown]
# ## On your own
#
# Once you are familiar with DataEval and data analysis, run this analysis on your own dataset. Analyze all of your data
# and not only the training set.
