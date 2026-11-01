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
# # Embed object detection crops and visualize class clusters
#
# DataEval's embedding-based tools, {class}`.Coverage`, {func}`.ber_mst` and {class}`.Balance`, assume one embedding per
# label, the image-classification shape. An object detection image holds many objects of different classes, so one
# whole-image embedding cannot be colored by one class. {class}`.DetectionCrops` presents the ground-truth boxes as an
# image-classification dataset, one crop per detection, so every per-(image, label) tool works on it unchanged. See
# [Performance limits](../concepts/PerformanceLimits.md).
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
from maite_datasets.object_detection import SeaDrone
from torchvision.models import ResNet18_Weights, resnet18

from dataeval import Embeddings, Metadata
from dataeval.config import set_device
from dataeval.data import DetectionCrops, Limit, Shuffle, View
from dataeval.extractors import TorchExtractor

# Use the GPU if one is available, otherwise the CPU. Every extractor then runs on that device.
set_device("cuda" if torch.cuda.is_available() else "cpu")

# Render the projection below as an interactive plotly figure.
dep.set_default_backend("plotly")
pio.renderers.default = "notebook"  # embed the plotly JS in the notebook output

# %% [markdown]
# ## Load and sample an object detection dataset
#
# Download the SeaDrone validation split with `SeaDrone` from `maite_datasets`. It holds 1,547 aerial frames, and each
# frame carries a variable number of detections, each with a bounding box and a class: swimmers, boats, jetskis,
# life-saving appliances and buoys. Loading it with `lazy=True` reads each frame only when you access it.
#
# The split is stored in id order, so the first 300 frames would come from only two of the three drones. Shuffle before
# you limit, which also keeps the example fast.

# %%
ds = SeaDrone("./data", image_set="val", download=True, lazy=True)
frames = View(ds, [Shuffle(seed=0), Limit(300)])

print(ds)
print(f"{len(frames)} of {len(ds)} frames")

# %% [markdown]
# ## Wrap the boxes as a classification dataset
#
# {class}`.DetectionCrops` turns each kept detection into one classification datum: the cropped box region, labeled with
# the detection's class. It goes directly into {class}`.Embeddings` and the rest of DataEval, with crops aligned 1:1 to
# labels. Most objects here are small, a few dozen pixels across, so the extractor's resize upscales each crop.
#
# By default each crop is the box squared by extending into the surrounding image (`square="expand"`), which keeps the
# object's aspect ratio when the extractor resizes it. Tiny or degenerate boxes are dropped (`min_size`), and the count
# is reported as `n_dropped`.

# %%
crops = DetectionCrops(frames)
print(f"crops (one per detection): {len(crops)}  |  dropped (too small): {crops.n_dropped}")

# %% [markdown]
# ## Extract an embedding per crop
#
# Embed every crop with a [pretrained
# ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html). An unmodified ResNet18
# outputs 1000 ImageNet class logits, but you want its learned features, so point {class}`.TorchExtractor` at the
# penultimate global-average-pool layer with `layer_name="avgpool"`. The extractor captures that layer's output, a
# 512-dimensional vector, for each crop. Its transforms resize and normalize each crop, and {class}`.Embeddings` runs
# the crops through in batches.

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
# Read the per-crop class labels from a {class}`.Metadata` built over the crop view. It lines up 1:1 with the
# embeddings, which is what every per-(image, label) tool expects.

# %%
metadata = Metadata(crops)
print("labels:", len(metadata.class_labels), "| embeddings:", len(embeddings))

# %% [markdown]
# ## Visualize the class clusters
#
# Project the 512-dimensional crop embeddings down to 2D and color each point by its class. t-SNE groups similar crops
# together, so well-separated classes form distinct clouds while confusable ones overlap. Hover and zoom to explore.

# %%
dep.project(
    embeddings,
    method="tsne",
    labels=metadata.class_labels,
    label_names=metadata.index2label,
    title="SeaDrone object crops in 2D (t-SNE), colored by class",
)

# %% [markdown]
# Boats form their own region and swimmers form one large cloud. Jetskis sit at the edge of the boat region, and buoys
# and life-saving appliances form small tight clumps around the swimmer cloud. Classes that occupy their own region are
# easy for a model to tell apart, and classes whose clouds touch are where confusion occurs. Swapping `method="tsne"`
# for `"umap"` or `"pca"` gives another view of the same embeddings.

# %% [markdown]
# ## Check how many neighbors are the same object
#
# A tight clump may be one object seen in several neighboring frames. For each crop, find the most similar other crop by
# cosine similarity and compare the ids of the two source frames.

# %%
frame_id = np.array([int(frames[i][2]["id"]) for i in range(len(frames))])
frame_ids = frame_id[crops.item_indices]  # the source frame of each crop

unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
similarity = unit @ unit.T
np.fill_diagonal(similarity, -np.inf)
gap = np.abs(frame_ids - frame_ids[similarity.argmax(axis=1)])

print(f"Most similar crop comes from the same frame: {np.mean(gap == 0):.0%}")
print(f"Most similar crop comes from a frame within 10 ids: {np.mean(gap <= 10):.0%}")

# %% [markdown]
# The most similar crop rarely comes from the same frame, but for about 58% of the crops it comes from a frame within 10
# ids, a neighboring frame. The clumps are mostly one object seen again and again, not separate objects. If you score or
# split on these embeddings, hold out whole blocks of neighboring frames, because the neighbor of a crop is usually a
# copy of it.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert embeddings.shape[0] == len(crops) == len(metadata.class_labels)  # crops align 1:1 with labels
assert len(crops) == 1938
assert crops.n_dropped == 0
assert np.bincount(metadata.class_labels, minlength=6).tolist() == [0, 1304, 411, 64, 65, 94]  # no "not_used" crops
assert len(frame_ids) == len(crops)
assert len({str(ds[i][2]["drone"]) for i in range(300)}) == 2  # the first 300 frames cover two of three drones
assert np.mean(gap == 0) < 0.1  # "rarely"
assert abs(np.mean(gap <= 10) - 0.58) < 0.03  # "about 58%"
sides = np.concatenate([
    np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
    for b in (np.asarray(frames[i][1].boxes) for i in range(len(frames)))
])
assert np.median(sides) < 64  # "a few dozen pixels across"

# %% [markdown]
# ## Next steps
#
# - [Embeddings](../concepts/Embeddings.md) — Feature vectors extracted per crop for evaluation and analysis.
# - [Assess an unlabeled data space](./tt_assess_data_space.py) — Embed, project, and reason about a dataset end to end.
# - [Assess whether a classification requirement is reachable](./tt_assess_ic_feasibility.py) — Check whether
#   embeddings like these can meet an accuracy requirement.
