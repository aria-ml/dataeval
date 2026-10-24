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
# # How to create embeddings

# %% [markdown]
# ## Problem statement
#
# Most DataEval analyses that group or compare images, such as [clustering](../concepts/Clustering.md), duplicate
# detection and [coverage](../concepts/DatasetBias.md#measuring-coverage-geometry-in-embedding-space), start from
# [embeddings](../concepts/Embeddings.md): one vector for each image.
#
# The recipe is the same whichever framework holds your model. You choose an extractor, then pass the dataset and the
# extractor to {class}`.Embeddings`. This guide builds embeddings with PyTorch and with ONNX from the same weights, checks
# that the two agree, and shows the two extractors that need no model.

# %% [markdown]
# ### When to use
#
# Use this guide when you want to:
#
# - turn a dataset into embeddings with a pre-trained PyTorch or ONNX model
# - move a model from PyTorch to ONNX and confirm that the embeddings do not change
# - build a baseline from raw pixels or from local image features, with no model

# %% [markdown]
# ### What you will need
#
# 1. An image dataset. This guide uses 100 images of the VOC2012 validation split.
# 1. A Python environment with the following packages installed:
#    - `dataeval[onnx,opencv]`
#    - `maite-datasets`
#    - `torchvision`

# %% [markdown]
# ## Getting started
#
# Import the libraries you need for a minimal working example.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval[onnx,opencv] maite-datasets torchvision
except Exception:
    pass

# %%
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from maite_datasets.object_detection import VOCDetection
from torchvision.models import ResNet18_Weights, resnet18

from dataeval import Embeddings
from dataeval.data import Limit, Resize, View
from dataeval.extractors import BoVWExtractor, FlattenExtractor, OnnxExtractor, TorchExtractor
from dataeval.utils.onnx import to_encoding_model

# %% [markdown]
# ## Load the dataset
#
# Load the validation split of VOC2012 and keep the first 100 images. Limit the dataset with a {class}`.View` instead of
# slicing the embeddings afterward, so the extractor only processes the images you keep.

# %%
dataset = VOCDetection(root="./data", image_set="val", year="2012", download=True)
subset = View(dataset, Limit(100))

print(f"Images: {len(subset)}")
print(f"Shape of the first image: {subset[0][0].shape}")

# %% [markdown]
# ## Create embeddings with PyTorch
#
# Wrap a pre-trained model in a {class}`.TorchExtractor`, then pass the dataset and the extractor to
# {class}`.Embeddings`. Three arguments decide what the embeddings are:
#
# - `transforms` applies the preprocessing the model was trained with. Torchvision publishes it with the weights.
# - `layer_name` selects the layer whose output becomes the embedding. Without it, the extractor returns the model output,
#   which for ResNet18 is 1000 ImageNet class scores and not image features. `avgpool` is the layer before the classifier.
# - `batch_size` on `Embeddings` sets how many images are loaded and encoded at a time. It does not change the result.

# %%
weights = ResNet18_Weights.DEFAULT
torch_extractor = TorchExtractor(
    resnet18(weights=weights, progress=False),
    transforms=weights.transforms(),
    layer_name="avgpool",
)
torch_embeddings = Embeddings(subset, extractor=torch_extractor, batch_size=32)

print(f"Embeddings shape: {torch_embeddings.shape}")

# %% [markdown]
# `Embeddings` computes the vectors when you first use them and keeps them. Pass the object itself to a DataEval
# evaluator, or call `np.asarray` on it when you need a plain array.

# %% [markdown]
# ## Create embeddings with ONNX
#
# ONNX stores a model in a framework-independent file. Export the same pre-trained ResNet18 so you can compare the two
# paths. If you already have an ONNX model, skip the export and start at `to_encoding_model`.
#
# Export a fresh copy of the weights. `TorchExtractor` moves the model it receives to the configured device, so the model
# above is not the one to export.

# %%
model_path = Path("data") / "resnet18.onnx"
model_path.parent.mkdir(exist_ok=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    torch.onnx.export(
        resnet18(weights=weights, progress=False).eval(),
        (torch.zeros(1, 3, 224, 224),),
        str(model_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
        dynamo=False,
    )

# %% [markdown]
# An exported classifier outputs class scores. `to_encoding_model` finds the layer before the classifier, which is the
# layer you named with `layer_name="avgpool"` above, and returns a model that outputs it along with the layer name.

# %%
encoding_model, embedding_layer = to_encoding_model(model_path)

print(f"Embedding layer: {embedding_layer}")

# %% [markdown]
# The torchvision preprocessing expects a tensor, but `OnnxExtractor` gives each transform a NumPy array. Wrap the same
# preprocessing so that both paths see identical inputs, then name the embedding layer in `output_name`.


# %%
def numpy_transform(transform: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[np.ndarray], np.ndarray]:
    """Run a torchvision preprocessing transform on one NumPy image and return a NumPy array."""

    def apply(image: np.ndarray) -> np.ndarray:
        return transform(torch.as_tensor(np.asarray(image))).numpy()

    return apply


onnx_extractor = OnnxExtractor(
    encoding_model,
    transforms=numpy_transform(weights.transforms()),
    output_name=embedding_layer,
)
onnx_embeddings = Embeddings(subset, extractor=onnx_extractor, batch_size=32)

print(f"Embeddings shape: {onnx_embeddings.shape}")

# %% [markdown]
# ## Check that both paths agree
#
# Compare the two sets of embeddings with cosine similarity. A value close to 1 means two vectors point the same way.
# Small differences in the last digits are expected, because the two runtimes order floating-point operations
# differently.

# %%
torch_array = np.asarray(torch_embeddings)
onnx_array = np.asarray(onnx_embeddings)
cosine = (torch_array * onnx_array).sum(axis=1) / (
    np.linalg.norm(torch_array, axis=1) * np.linalg.norm(onnx_array, axis=1)
)

print(f"Shapes: {torch_array.shape} and {onnx_array.shape}")
print(f"Lowest cosine similarity: {cosine.min():.6f}")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert torch_array.shape == onnx_array.shape == (100, 512)
assert cosine.min() > 0.999

# %% [markdown]
# ## Create embeddings without a model
#
# {class}`.FlattenExtractor` and {class}`.BoVWExtractor` need no pre-trained model. Flattening raw pixels is a baseline
# and not an assessment, because distances between raw pixels follow brightness and alignment more than content.
# `BoVWExtractor` describes each image by a histogram of local features and is the extractor to use for near-duplicates
# that survived a rotation or a rescale.
#
# `FlattenExtractor` needs every image to have one shape, and VOC images differ in size. Resize them in a view first.
# A view defines the dataset, so every DataEval evaluator then sees the resized images.
#
# `BoVWExtractor` learns its vocabulary of local features from the images you give `fit`. Fit it yourself on the images
# that should define the vocabulary. If you call it unfitted, it fits on that one call, which inside `Embeddings` is a
# single batch, and it raises a warning.

# %%
resized = View(subset, Resize((128, 128)))

flatten_embeddings = Embeddings(resized, extractor=FlattenExtractor(), batch_size=32)
bovw_extractor = BoVWExtractor(vocab_size=64).fit(resized)
bovw_embeddings = Embeddings(resized, extractor=bovw_extractor, batch_size=32)

print(f"Flatten embeddings shape: {flatten_embeddings.shape}")
print(f"BoVW embeddings shape: {bovw_embeddings.shape}")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert flatten_embeddings.shape == (100, 3 * 128 * 128)
assert bovw_embeddings.shape == (100, 64)

# %% [markdown]
# :::{note}
#
# `Embeddings` stores one row for each image. An extractor that returns one row for each detection, such as a detection
# model wrapped in {class}`.ScoresExtractor`, cannot be wrapped in it, and `Embeddings` raises an error when the number of
# rows differs from the number of images. Give those extractors straight to a drift detector.
#
# :::

# %% [markdown]
# ## Next steps
#
# - [Embeddings](../concepts/Embeddings.md) — Learn what embeddings represent and how they are used across DataEval.
# - [Validation and Trust](../concepts/ValidationAndTrust.md#feature-extractors-a-dependency-of-both-layers) — See what each extractor embeds and how each can mislead a metric.
# - [How to run clustering analysis](./h2_cluster_analysis.py) — Cluster embeddings to find groups, outliers and duplicates.
# - [How to detect undersampled data subsets](./h2_detect_undersampling.py) — Use embeddings to find thin regions of a dataset.
