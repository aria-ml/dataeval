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
# # How to embed object detection crops and visualize class clusters
#
# DataEval's embedding-based tools, {class}`.Coverage`, {func}`.ber_mst` and {class}`.Balance`, assume one embedding per
# label, the image-classification shape. An object detection image holds many objects of different classes, so one
# whole-image embedding cannot be colored by one class. {class}`.DetectionCrops` presents the ground-truth boxes as an
# image-classification dataset, one crop per detection, so every per-(image, label) tool works on it unchanged. See
# [Performance Limits](../concepts/PerformanceLimits.md).
#
# Use it when you want to embed the objects in a detection dataset, for example to see how the classes cluster in
# embedding space.
#
# You need `dataeval`, `dataeval-plots`, `maite-datasets` and `torchvision`.

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
from dataeval.config import set_device
from dataeval.data import DetectionCrops, Filter, View
from dataeval.extractors import TorchExtractor

# Use the GPU if one is available, otherwise the CPU. Every extractor then runs on that device.
set_device("cuda" if torch.cuda.is_available() else "cpu")

# Render the projection below as an interactive plotly figure.
dep.set_default_backend("plotly")
pio.renderers.default = "notebook"  # embed the plotly JS in the notebook output

# %% [markdown]
# ## Load an object detection dataset
#
# Download the PASCAL VOC 2012 training split with `VOCDetection` from `maite_datasets`. Each image
# carries a variable number of object detections, each with a bounding box and a class.

# %%
ds = VOCDetection(root="./data", year="2012", image_set="train", download=True)
print(ds)

# %% [markdown]
# ## Focus on a few classes
#
# VOC has 20 classes. Coloring every crop of all of them at once is hard to read. Pick a
# handful of visually distinct classes and keep only their detections with
# {class}`.Filter` (applied through {class}`.View`). This also keeps the example fast.
# Fewer classes means fewer crops to embed and project. Skip this step to use the whole
# dataset.

# %%
index2label = ds.metadata.get("index2label", {})  # {0: "aeroplane", 1: "bicycle", ...}
name_to_index = {name: index for index, name in index2label.items()}

focus_classes = ["aeroplane", "bicycle", "bird", "car", "cat", "dog"]
focus_indices = [name_to_index[name] for name in focus_classes]

focused = View(ds, [Filter(classes=focus_indices)])
print(f"{len(focused)} images contain at least one of {focus_classes}")

# %% [markdown]
# ## Wrap the boxes as a classification dataset
#
# {class}`.DetectionCrops` turns each kept detection into one classification datum: the
# cropped box region, labeled with the detection's class. The result satisfies the
# image-classification shape, so it goes directly into {class}`.Embeddings` and the rest of
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
# ## Extract an embedding per crop
#
# Embed every crop with a
# [pretrained ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html).
# An unmodified ResNet18 outputs 1000 ImageNet class logits, but you want its learned
# features, not its classifier. So rather than swapping in a new head, point
# {class}`.TorchExtractor` at the penultimate global-average-pool layer with
# `layer_name="avgpool"`. The extractor registers a forward hook and captures that layer's
# output, a 512-dimensional feature vector, for each crop. Its transforms resize and
# normalize each crop for the model, and {class}`.Embeddings` runs the crops through in
# batches.

# %%
resnet = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False)

extractor = TorchExtractor(
    resnet,
    transforms=ResNet18_Weights.DEFAULT.transforms(),
    layer_name="avgpool",  # capture the penultimate features, not the 1000-class logits
)
embeddings = np.asarray(Embeddings(crops, extractor=extractor, batch_size=64))
print("embedding shape:", embeddings.shape)

# %% [markdown]
# Read the per-crop class labels from a {class}`.Metadata` built over the crop view. Because
# the crops are an image-classification dataset, `class_labels` lines up 1:1 with the
# embeddings. This is what every per-(image, label) tool expects.

# %%
metadata = Metadata(crops)
print("labels:", len(metadata.class_labels), "| embeddings:", len(embeddings))

# %% [markdown]
# ## Visualize the class clusters
#
# Project the 512-dimensional crop embeddings down to 2D and color each point by its class.
# t-SNE groups visually similar crops together, so well-separated classes form distinct
# clouds while confusable ones overlap. This gives a quick read on how discriminable the object
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
# Classes that occupy their own region are easy for a model to tell apart from the others. Classes whose clouds overlap
# share visual features and are where confusion occurs. Swapping `method="tsne"` for `"umap"` or `"pca"` gives a
# different view of the same embeddings.

# %% [markdown]
# ## Next steps
#
# - [Embeddings](../concepts/Embeddings.md) — Feature vectors extracted per crop for evaluation and analysis.
# - [Assess an unlabeled data space](./tt_assess_data_space.py) — Embed, project, and reason about a dataset end to end.
# - [How to determine image classification feasibility](./h2_measure_ic_feasibility.py) — Put crop embeddings through
#   BER to bound achievable classification accuracy.
