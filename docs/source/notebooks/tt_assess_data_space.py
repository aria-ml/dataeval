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
# This tutorial is a beginner-friendly introduction to exploratory data analysis when labels are not available. You will
# assess a large set of collected vehicle imagery before anyone annotates it.
#
# Estimated time to complete: 10 minutes
#
# Relevant ML stages: [Data Engineering](../getting-started/roles/ML_Lifecycle.md#data-engineering)
#
# Relevant personas: [Data Engineer](../getting-started/roles/data_engineer.md), [Data Scientist](../getting-started/roles/data_scientist.md)

# %% [markdown]
# ## What you will do
#
# - Construct [embeddings](../concepts/Embeddings.md) by inferencing a set of military vehicle images through a model
# - Analyze clustered embeddings to find duplicates and
#   [outliers](../concepts/DataIntegrity.md#outlier-detection-image-statistics-and-embeddings)
# - Run additional analysis to find gaps in data
#   [coverage](../concepts/DatasetBias.md#measuring-coverage-geometry-in-embedding-space)
#
# ## What you will learn
#
# - Learn to use cluster-based algorithms to measure dataset quality without labels
# - Learn to distinguish whether an image is an outlier or is underrepresented
# - Learn to find the same image stored more than once
# - Learn to determine whether flagged images should be added or removed

# %% [markdown]
# ## What you will need
#
# - Basic familiarity with Python

# %% [markdown]
# ## Introduction
#
# You have collected a large set of vehicle imagery from many sources, and nobody has labeled it yet. Before you spend
# annotation effort or train a model on it, you need to know what the set holds: what does not belong, what repeats and
# what few images cover.
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
from maite_datasets.image_classification import MilitaryVehicles
from torchvision.models import ResNet18_Weights, resnet18

from dataeval import Embeddings
from dataeval.config import set_device, set_max_processes, set_seed
from dataeval.core import cluster, coverage_adaptive
from dataeval.extractors import TorchExtractor
from dataeval.quality import Duplicates, Outliers

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
# ### Download the dataset
#
# Download the train split with `MilitaryVehicles` from `maite_datasets`. It has 7,823 images of 24 types of military
# vehicle and equipment, assembled from other sources, so the image sizes and settings vary. The dataset comes with
# labels. You will ignore them until the end of the duplicates section, as if the images had not been annotated yet. You
# can also run this tutorial with your own MAITE-compliant dataset.

# %%
# Load the training dataset
dataset = MilitaryVehicles(root="./data", image_set="train", download=True)

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
# each other in the embedding space. Each group it returns holds images that look the same to the model. Run it on the
# clusters you already created.

# %%
duplicates = Duplicates().from_clusters(output).data()
groups = duplicates["item_indices"].to_list()

print(f"Duplicate groups: {len(groups)}")
print(f"Images in a group: {len({i for group in groups for i in group})}")

# %% [markdown]
# Some images have a partner that looks the same to the model. Plot the first three pairs to look at them. Each row
# holds one pair.

# %%
pairs = [group for group in groups if len(group) == 2][:3]
dep.plot(dataset, indices=[i for pair in pairs for i in pair], figsize=(10, 6), images_per_row=2)

# %% [markdown]
# Each pair you plotted is the same photo stored twice, which wastes annotation effort and storage. The labels can show
# a bigger problem. This tutorial treats the images as unlabeled, but the dataset does come with labels. Look up the
# label of each image in the pairs, and count how many groups hold images with different labels.

# %%
index2label = dataset.index2label


def label_of(i: int) -> str:
    """Return the class name that the dataset gives to image `i`."""
    return index2label[int(np.argmax(dataset[i][1]))]


for pair in pairs:
    print([label_of(i) for i in pair])

conflicting = [group for group in groups if len({label_of(i) for i in group}) > 1]
print(f"Groups with more than one label: {len(conflicting)} of {len(groups)}")

# %% [markdown]
# Each of the first three pairs is filed under two different labels, and nearly every other group is too. At most one of
# the labels on a photo is right. A model cannot learn both, and when the copies land in different splits, the model
# sees the test image during training. You found the problem from the embeddings alone, without using a label. Keep one
# copy of each photo and decide which label is correct. The [Clean a dataset](./tt_clean_dataset.py) tutorial shows how
# to find and remove duplicates.
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
# indices calculation. You can also change the slice to see more examples.

# %%
# To use the flagged outliers, uncomment the line below
indices = outliers

# To use unflagged images, uncomment the line below
# indices = list(set(range(len(dataset))) - set(outliers))

# %% [markdown]
# Now visualize your chosen set of indices.

# %%
# Plot up to 12 images from the dataset
dep.plot(
    dataset,
    indices=list(indices)[:12],
    figsize=(10, 8),
    images_per_row=4,
)

# %% [markdown]
# Look at the images before you decide what to do. Some are not vehicles at all: a phone case, a T-shirt, an electrical
# connector and parts of a scale-model kit. Others are real photos of equipment in an unusual setting, such as a rocket
# launcher firing. The first kind does not belong in a vehicle dataset. The second kind is an edge case that a deployed
# model could meet. Decide which kind each image is before you remove anything.

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
uncovered_outliers = set(cov["uncovered_indices"].tolist()).intersection(set(outliers))

print(f"Number of outliers found as uncovered images: {len(uncovered_outliers)}")

# %% [markdown]
# Most of the outliers are also uncovered, but the uncovered images are a larger group. Plot six uncovered images that
# are not outliers to see what else the coverage check finds.

# %%
underrepresented = [i for i in cov["uncovered_indices"].tolist() if i not in set(outliers)]
dep.plot(dataset, indices=underrepresented[:6], figsize=(10, 6), images_per_row=3)

# %% [markdown]
# These images are unusual without being extreme. Scale models, a ring and a flag collage are not photos of vehicles, so
# you would remove them as you would an outlier. A close-up of a wheel or of equipment parts is a real view that few
# other images share. Whether you need more of those views depends on what the model has to recognize. Collect more
# close-ups if it must recognize equipment from partial views, and leave them out if it does not.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert (len(dataset), len(dataset.index2label)) == (7823, 24)
assert embeddings.shape == (len(dataset), 512)
assert 0 < len(outliers) <= 0.05 * len(dataset)
assert len(cov["uncovered_indices"]) > 0
# The counts move a little with the hardware, so the checks below use bands the prose stays true in
assert len(pairs) == 3
assert all(len({label_of(i) for i in pair}) == 2 for pair in pairs)  # "filed under two different labels"
assert len(conflicting) >= 0.9 * len(groups)  # "nearly every other group is too"
assert len(set(outliers) & {2055, 5022, 7054, 5094, 4062, 7345, 7365, 7557}) >= 6  # junk and rocket launchers firing
assert len(uncovered_outliers) >= 0.8 * len(outliers)  # "most of the outliers are also uncovered"
assert len(cov["uncovered_indices"]) > len(outliers)  # "a larger group"
assert len(set(underrepresented[:6]) & {3267, 4928, 278, 5104, 3867}) >= 3  # models, close-ups, a ring, a flag

# %% [markdown]
# ## Handling flagged images
#
# You now have enough information to decide what to do with these images. First decide whether an image belongs in the
# dataset at all, because that decides which flag matters. Remove an image that does not belong, whatever else it is
# flagged for. An image that does belong, such as a rare but relevant event, is an edge case. If it is flagged as both
# an outlier and underrepresented, handle it as missing coverage.
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
#
# ### Handling duplicates
#
# Keep one copy of each group of duplicates. When the copies carry different labels, decide which label is right before
# you keep the image. The [Clean a dataset](./tt_clean_dataset.py) tutorial shows how to find and remove duplicates.

# %% [markdown]
# ## Conclusion
#
# In this tutorial you created image embeddings for more efficient calculations, used cluster-based algorithms to find
# duplicates and outliers, checked for gaps in coverage on military vehicle imagery, and made decisions on possible
# solutions.
#
# ## What you learned
#
# - Applied cluster-based feature space metrics ({func}`.cluster`, {class}`.Duplicates`, {class}`.Outliers`,
#   {func}`.coverage_adaptive`) to evaluate unlabeled imagery.
# - Found the same image stored under more than one label.
# - Differentiated between irrelevant outliers (candidates for pruning) and underrepresented operational edge cases
#   (candidates for targeted collection).
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
# - [Clean a dataset](./tt_clean_dataset.py) — Find and remove duplicates and extreme values before you train.

# %% [markdown]
# ## On your own
#
# Once you are familiar with DataEval and data analysis, run this analysis on your own dataset. Analyze all of your data
# and not only the training set.
