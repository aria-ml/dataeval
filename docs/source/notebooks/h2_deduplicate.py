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
# The Duplicates class should be used if you need to find duplicate images in your dataset. It also looks past pixels:
# when your dataset carries object detection targets, it checks whether two items share one annotation (a synthetic
# copy) or share pixels but disagree on annotation (one collect labeled twice) - see the annotation duplicates
# section below.

# %% [markdown]
# ### What you will need
#
# 1. A python environment with following packages installed:
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
from dataclasses import dataclass

import numpy as np
from IPython.display import display
from maite_datasets.image_classification import MNIST

from dataeval import Metadata
from dataeval.config import set_max_processes
from dataeval.data import Indices, View
from dataeval.protocols import DatasetMetadata, DatumMetadata
from dataeval.quality import Duplicates

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
duplicates_ds = View(testing_dataset, Indices(indices_with_duplicates))

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
# The results can be viewed as a DataFrame with exact and near groups enumerated.

# %%
display(results)

# %% [markdown]
# The `Duplicates` class was able to find all 4 exact duplicates out of the 10,000 samples.
#
# It also found several sets of images that are very closely related to each other, and since we are using hand written
# digits we would expect it to find some images that were nearly identical.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert results.exact
assert len(results.exact) == len(duplicates)
for k, v in duplicates.items():
    assert [v, k] in results.exact


# %% [markdown]
# ## Beyond pixels: annotation duplicates
#
# MNIST has no object detection targets, so everything above compared pixels only. When a dataset does carry targets
# (boxes and labels), `Duplicates` also digests the annotation itself - on by default, no argument needed - and two
# accessors narrow the result to the annotation-driven relations:
#
# - `divergent()` - items whose annotation was checked and found to disagree: one collect labeled two different ways.
# - `augmented()` - items sharing one annotation over different pixels: the signature of a synthetic copy.
#
# Build a tiny object detection dataset with one pair of each to see both.


# %%
@dataclass
class BoxTarget:
    """A minimal object detection target: boxes, labels, and scores."""

    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


class TinyObjectDetectionDataset:
    """Four items as two duplicate pairs: one relabeled, one augmented."""

    def __init__(self, images: list[np.ndarray], boxes: list[list[list[float]]], labels: list[list[int]]) -> None:
        self._images = images
        self._boxes = boxes
        self._labels = labels
        self.metadata: DatasetMetadata = DatasetMetadata(id="tiny-od", index2label={0: "object"})

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[np.ndarray, BoxTarget, DatumMetadata]:
        item_labels = np.asarray(self._labels[index], dtype=np.intp)
        target = BoxTarget(
            boxes=np.asarray(self._boxes[index], dtype=np.float32),
            labels=item_labels,
            scores=np.ones(len(item_labels), dtype=np.float32),
        )
        return self._images[index], target, DatumMetadata(id=index)


rng = np.random.default_rng(0)
same_image = rng.random((3, 16, 16))

# Items 0, 1: identical pixels, two different boxes drawn for the same collect - a divergent pair.
# Items 2, 3: identical annotation, two independently sampled images - an augmented pair.
annotation_images = [same_image, same_image.copy(), rng.random((3, 16, 16)), rng.random((3, 16, 16))]
annotation_boxes = [[[1.0, 1.0, 6.0, 6.0]], [[9.0, 9.0, 14.0, 14.0]], [[0.0, 0.0, 5.0, 5.0]], [[0.0, 0.0, 5.0, 5.0]]]
annotation_labels = [[0], [0], [0], [0]]

annotation_ds = TinyObjectDetectionDataset(annotation_images, annotation_boxes, annotation_labels)
annotation_results = Duplicates(hash_radius=0).evaluate(annotation_ds)

# %% [markdown]
# `divergent()` finds the pair whose pixels matched but whose annotation did not - a label-quality issue, not a
# redundancy one.

# %%
display(annotation_results.divergent())

# %% [markdown]
# `augmented()` finds the pair whose annotation matched but whose pixels did not - a synthetic copy's signature.

# %%
display(annotation_results.augmented())

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(annotation_results.divergent()) == 1
assert annotation_results.divergent().data()["item_indices"].item().to_list() == [0, 1]
assert len(annotation_results.augmented()) == 1
assert annotation_results.augmented().data()["item_indices"].item().to_list() == [2, 3]

# %% [markdown]
# ## Next steps
#
# - [Acting on Results](../concepts/ActingOnResults.md) — Learn strategies for addressing dataset issues identified during evaluation.
# - [Clustering](../concepts/Clustering.md) — Understand clustering techniques for grouping similar data points and detecting patterns.
# - [Data Integrity](../concepts/DataIntegrity.md) — Analyze image-level and target-level statistics to identify data quality issues.
# - [Dataset Bias and Coverage](../concepts/DatasetBias.md) — Evaluate bias and coverage across metadata factors in your dataset.
# - [How to find duplicate video](./h2_deduplicate_video.py) — Find duplicate footage, train/test leakage, and redundant frames in video and MOT datasets.
# - [Detecting common augmentations as duplicates](./tt_augmentation_duplicates.py) — Find near-identical images created through synthetic transformations and augmentations.
# - [Introduction to data cleaning](./tt_clean_dataset.py) — Clean and prepare datasets for model training by finding duplicates, outliers, and corrupted data.
# - [How to add intrinsic factors to Metadata](./h2_add_intrinsic_factors.py) — Compute and attach intrinsic factors such as dimensions and pixel statistics to dataset metadata.
# - [How to encode images with ONNX models](./h2_encode_with_onnx.py) — Encode image datasets into vector embeddings using ONNX runtime models.
# - [How to run clustering analysis](./h2_cluster_analysis.py) — Cluster embeddings and factors to discover structural groupings in your data.
# - [How to specify custom statistics on object detection datasets](./h2_custom_image_stats_object_detection.py) — Configure custom image statistics across full images, bounding boxes, and backgrounds.
# - [How to visualize cleaning issues](./h2_visualize_cleaning_issues.py) — Plot and visualize dataset anomalies, duplicates, and quality issues.
