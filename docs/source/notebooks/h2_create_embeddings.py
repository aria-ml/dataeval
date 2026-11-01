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
# # Create embeddings
#
# Most DataEval analyses that group or compare images start from [embeddings](../concepts/Embeddings.md), one vector for
# each image. You choose an extractor, then pass the dataset and the extractor to {class}`.Embeddings`.
#
# Use this guide when you want to:
#
# - turn a dataset into embeddings with a pre-trained PyTorch or ONNX model
# - build a baseline from raw pixels or from local image features, with no model
#
# It uses 300 MilitaryVehicles photographs and needs `dataeval[onnx,opencv]`, `maite-datasets` and `torchvision`.

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
from maite_datasets.image_classification import MilitaryVehicles
from torchvision.models import ResNet18_Weights, resnet18

from dataeval import Embeddings, Metadata
from dataeval.data import Limit, Resize, Shuffle, View
from dataeval.extractors import BoVWExtractor, FlattenExtractor, OnnxExtractor, TorchExtractor
from dataeval.utils.onnx import to_encoding_model

# %% [markdown]
# ## Load the dataset
#
# Load the test split of MilitaryVehicles and keep 300 images at random. The split is stored class by class, so shuffle
# before you limit. Limit the dataset with a {class}`.View` and not by slicing the embeddings afterward, so the
# extractor only processes the images you keep.

# %%
dataset = MilitaryVehicles("./data", image_set="test", download=True, lazy=True)
subset = View(dataset, [Shuffle(seed=0), Limit(300)])
labels = Metadata(subset).class_labels

print(f"Images: {len(subset)}")
print(f"Shape of the first image: {subset[0][0].shape}")

# %% [markdown]
# ## Create embeddings with PyTorch
#
# Wrap a pre-trained model in a {class}`.TorchExtractor`, then pass the dataset and the extractor to
# {class}`.Embeddings`. Two arguments of the extractor decide what the embeddings are:
#
# - `transforms` applies the preprocessing the model was trained with. Torchvision publishes it with the weights.
# - `layer_name` selects the layer whose output becomes the embedding. Without it, the extractor returns the model
#   output, which for ResNet18 is 1000 ImageNet class scores and not image features. `avgpool` is the layer before the
#   classifier.

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
# `Embeddings` computes the vectors on first use. Pass the object itself to a DataEval
# evaluator, or call `np.asarray` on it when you need a plain array.

# %% [markdown]
# ## Create embeddings with ONNX
#
# Export the pre-trained ResNet18 to ONNX. If you already have an ONNX model, skip the export and start at
# `to_encoding_model`.
#
# Export a fresh copy of the weights. `TorchExtractor` moves the model it receives to the configured device, so the
# model above is not the one to export.

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
# `to_encoding_model` finds the layer before the classifier, which is the `avgpool` layer above, and returns a model
# that outputs it along with the layer name.

# %%
encoding_model, embedding_layer = to_encoding_model(model_path)

print(f"Embedding layer: {embedding_layer}")


# %% [markdown]
# The torchvision preprocessing expects a tensor, but `OnnxExtractor` gives each transform a NumPy array. Wrap the same
# preprocessing so that both paths see identical inputs, and name the embedding layer in `output_name`.


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

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
# the ONNX path reproduces the PyTorch embeddings
torch_array = np.asarray(torch_embeddings)
onnx_array = np.asarray(onnx_embeddings)
cosine = (torch_array * onnx_array).sum(axis=1) / (
    np.linalg.norm(torch_array, axis=1) * np.linalg.norm(onnx_array, axis=1)
)
assert torch_array.shape == onnx_array.shape == (300, 512)
assert cosine.min() > 0.999

# %% [markdown]
# ## Create embeddings without a model
#
# {class}`.FlattenExtractor` and {class}`.BoVWExtractor` need no pre-trained model. Flattening raw pixels is a baseline
# and not an assessment, because distances between raw pixels follow brightness and alignment more than content.
# `BoVWExtractor` describes each image by a histogram of local features and suits near-duplicates that survived a
# rotation or a rescale.
#
# `FlattenExtractor` needs every image to have one shape, and the photographs differ in size, so resize them in a view
# first. `BoVWExtractor` learns its vocabulary from the images you give `fit`, so fit it on the images that should
# define it. Unfitted, it fits on one batch inside `Embeddings` and raises a warning.

# %%
resized = View(subset, Resize((128, 128)))

flatten_embeddings = Embeddings(resized, extractor=FlattenExtractor(), batch_size=32)
bovw_extractor = BoVWExtractor(vocab_size=64).fit(resized)
bovw_embeddings = Embeddings(resized, extractor=bovw_extractor, batch_size=32)

print(f"Flatten embeddings shape: {flatten_embeddings.shape}")
print(f"BoVW embeddings shape: {bovw_embeddings.shape}")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert flatten_embeddings.shape == (300, 3 * 128 * 128)
assert bovw_embeddings.shape == (300, 64)


# %% [markdown]
# ## Check what the embeddings capture
#
# Embeddings help only if images of one kind land near each other. For each photograph, find its nearest neighbor by
# cosine similarity and check whether it shows the same type of vehicle. By chance that happens about 6% of the time.


# %%
def same_class_neighbors(embeddings: Embeddings) -> float:
    """Share of the images whose nearest other image has the same class."""
    unit = np.asarray(embeddings, dtype=np.float64)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -np.inf)
    return float((labels[similarity.argmax(axis=1)] == labels).mean())


chance = float(((np.bincount(labels) / len(labels)) ** 2).sum())
rates = {
    "ResNet18, PyTorch": same_class_neighbors(torch_embeddings),
    "ResNet18, ONNX": same_class_neighbors(onnx_embeddings),
    "Flattened pixels": same_class_neighbors(flatten_embeddings),
    "BoVW": same_class_neighbors(bovw_embeddings),
}

print(f"{'By chance':>17}: {chance:.1%}")
for name, rate in rates.items():
    print(f"{name:>17}: {rate:.1%}")

# %% [markdown]
# The pre-trained model finds a neighbor of the same type about five times as often as chance, and the ONNX copy matches
# it. Flattened pixels and BoVW are near or below chance, because they describe how a photograph looks and not which
# vehicle it shows. Use them as baselines, or for near-duplicates, and not for grouping by class.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(set(labels.tolist())) == 24  # the sample covers every class
assert abs(chance - 0.056) < 0.005
assert 4 < rates["ResNet18, PyTorch"] / chance < 6  # "about five times"
assert abs(rates["ResNet18, ONNX"] - rates["ResNet18, PyTorch"]) < 0.01  # "matches it"
assert rates["Flattened pixels"] < 1.5 * chance
assert rates["BoVW"] < 0.5 * rates["ResNet18, PyTorch"]  # BoVW varies between runs, so compare, not pin

# %% [markdown]
# :::{note}
#
# `Embeddings` stores one row for each image, so it raises an error for an extractor that returns one row for each
# detection, such as a detection model wrapped in {class}`.ScoresExtractor`. Give those extractors straight to a drift
# detector.
#
# :::

# %% [markdown]
# ## Next steps
#
# - [Embeddings](../concepts/Embeddings.md) — Learn what embeddings represent and how they are used across DataEval.
# - [Feature extractors: a dependency of both
#   layers](../concepts/ValidationAndTrust.md#feature-extractors-a-dependency-of-both-layers) — See what each extractor
#   embeds and how each can mislead a metric.
# - [Run clustering analysis](./h2_cluster_analysis.py) — Cluster embeddings to find groups, outliers and
#   duplicates.
