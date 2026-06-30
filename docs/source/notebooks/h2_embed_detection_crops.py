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
# # Embed object detection crops and visualize class clusters
#
# DataEval's embedding-based tools — {class}`.Coverage`, {func}`.ber_mst`,
# {class}`.Balance` — assume **one embedding per label**, which is the image-classification
# shape. An object detection image holds *many* objects of *different* classes, so a single
# whole-image embedding cannot be colored by one class, and these tools do not consume it
# directly.
#
# This guide shows the bridge: {class}`.DetectionCrops` presents an object detection
# dataset's ground-truth boxes *as* an image-classification dataset — one crop per
# detection, labeled with the detection's class — so every per-(image, label) tool works on
# it unchanged. You'll extract an embedding for each object crop from PASCAL VOC and project
# them to 2D to *see* how the classes cluster in embedding space. This is the same
# bounding-box-classification view behind detection feasibility (see
# [Performance Limits](../concepts/PerformanceLimits.md)) and embedding-space coverage.
#
# Estimated time to complete: 10 minutes

# %% [markdown]
# ## What you'll need
#
# - A Python environment with `dataeval`, `dataeval-plots`, `maite-datasets`, and
#   `torchvision` installed.

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

from dataeval import Embeddings, Metadata
from dataeval.data import ClassFilter, DetectionCrops, Select
from dataeval.extractors import TorchExtractor

# Use the GPU if one is available, otherwise the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Render the projection below as an interactive plotly figure.
dep.set_default_backend("plotly")
pio.renderers.default = "notebook"  # embed the plotly JS in the notebook output

# %% [markdown]
# ## 1. Load an object detection dataset
#
# Download the PASCAL VOC 2012 training split with {class}`.VOCDetection`. Each image
# carries a variable number of object detections, each with a bounding box and a class.

# %%
ds = VOCDetection(root="./data", year="2012", image_set="train", download=True)
print(ds)

# %% [markdown]
# ## 2. Focus on a few classes
#
# VOC has 20 classes; coloring every crop of all of them at once is hard to read. Pick a
# handful of visually distinct classes and keep only their detections with
# {class}`.ClassFilter` (applied through {class}`.Select`). This also keeps the example fast
# — fewer classes means fewer crops to embed and project. Skip this step to use the whole
# dataset.

# %%
index2label = ds.metadata["index2label"]  # {0: "aeroplane", 1: "bicycle", ...}
name_to_index = {name: index for index, name in index2label.items()}

focus_classes = ["aeroplane", "bicycle", "bird", "car", "cat", "dog"]
focus_indices = [name_to_index[name] for name in focus_classes]

focused = Select(ds, [ClassFilter(focus_indices)])
print(f"{len(focused)} images contain at least one of {focus_classes}")

# %% [markdown]
# ## 3. Wrap the boxes as a classification dataset
#
# {class}`.DetectionCrops` turns each kept detection into one classification datum — the
# cropped box region, labeled with the detection's class. The result satisfies the
# image-classification shape, so it drops straight into {class}`.Embeddings` and the rest of
# DataEval, with crops aligned 1:1 to labels.
#
# By default each crop is the box squared by extending into the surrounding image
# (`square="expand"`), which avoids distorting the object's aspect ratio when the extractor
# resizes it. Tiny or degenerate boxes are dropped (`min_size`), and the count is reported
# as `n_dropped`.

# %%
crops = DetectionCrops(focused)
print(f"crops (one per detection): {len(crops)}  |  dropped (too small): {crops.n_dropped}")

# %% [markdown]
# ## 4. Extract an embedding per crop
#
# Embed every crop with a
# [pretrained ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html).
# An unmodified ResNet18 *outputs* 1000 ImageNet class logits, but you want its learned
# features, not its classifier — so rather than swapping in a new head, point
# {class}`.TorchExtractor` at the penultimate global-average-pool layer with
# `layer_name="avgpool"`. The extractor registers a forward hook and captures that layer's
# output — a 512-dimensional feature vector — for each crop. Its transforms resize and
# normalize each crop for the model, and {class}`.Embeddings` runs the crops through in
# batches.

# %%
resnet = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False)

extractor = TorchExtractor(
    resnet,
    transforms=ResNet18_Weights.DEFAULT.transforms(),
    device=device,
    layer_name="avgpool",  # capture the penultimate features, not the 1000-class logits
)
embeddings = np.asarray(Embeddings(dataset=crops, extractor=extractor, batch_size=64)[:])
print("embedding shape:", embeddings.shape)

# %% [markdown]
# Read the per-crop class labels from a {class}`.Metadata` built over the crop view. Because
# the crops *are* an image-classification dataset, `class_labels` lines up 1:1 with the
# embeddings — exactly what every per-(image, label) tool expects.

# %%
metadata = Metadata(crops)
print("labels:", len(metadata.class_labels), "| embeddings:", len(embeddings))

# %% [markdown]
# ## 5. Visualize the class clusters
#
# Project the 512-dimensional crop embeddings down to 2D and color each point by its class.
# t-SNE groups visually similar crops together, so well-separated classes form distinct
# clouds while confusable ones overlap — a quick read on how discriminable the object
# classes are in this embedding space. Hover and zoom to explore.

# %%
dep.project(
    embeddings,
    method="tsne",
    labels=metadata.class_labels,
    label_names=metadata.index2label,
    title="VOC object crops in 2D (t-SNE), colored by class",
)

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert embeddings.shape[0] == len(crops) == len(metadata.class_labels)  # crops align 1:1 with labels
assert set(np.unique(metadata.class_labels)).issubset(set(focus_indices))  # only the focus classes remain

# %% [markdown]
# Classes that occupy their own region are easy for a model to tell apart from the others;
# classes whose clouds overlap share visual features and are where confusion (and
# irreducible error) lives. Swapping `method="tsne"` for `"umap"` or `"pca"` gives a
# different view of the same embeddings.

# %% [markdown]
# ## Related concepts
#
# - [Embeddings](../concepts/Embeddings.md) — the feature vectors this guide extracts per
#   crop and what the tools below consume.
# - [Performance Limits](../concepts/PerformanceLimits.md) — the detection-feasibility view
#   that the overlap between class clusters foreshadows.
# - [Dataset Bias and Coverage](../concepts/DatasetBias.md) — measuring how the crop
#   embeddings fill (or leave gaps in) the feature space.
# - [Clustering](../concepts/Clustering.md) — how DataEval groups embeddings, the formal
#   counterpart to the clusters you eyeball here.
#
# ## See also
#
# ### How-to guides
#
# - [How to determine image classification feasibility](./h2_measure_ic_feasibility.py) —
#   put these crop embeddings through BER to bound achievable detection accuracy.
# - [How to detect undersampled data subsets](./h2_detect_undersampling.py) — find classes
#   or regions thin on coverage in the same embedding space.
# - [How to run clustering analysis](./h2_cluster_analysis.py) — cluster crop embeddings to
#   surface outliers and duplicates.
# - [How to specify custom statistics on object detection datasets](./h2_custom_image_stats_object_detection.py)
#   — other tooling that works directly on object detection datasets.
# - [How to encode images with ONNX models](./h2_encode_with_onnx.py) — an alternative to
#   `TorchExtractor` for producing the embeddings.
#
# ### Tutorials
#
# - [Assess an unlabeled data space](./tt_assess_data_space.py) — embed, project, and reason
#   about a dataset end to end.
