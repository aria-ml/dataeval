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
# # How to detect uncertainty drift with a MAITE model
#
# As data moves into regions that a deployed model was not trained on, its predictions grow less confident. An upward
# shift in the {term}`Shannon entropy<Shannon Entropy>` of the predicted probabilities is a label-free drift signal. See
# [uncertainty-based drift detection](../concepts/DistributionShift.md#uncertainty-based-drift-detection). A model that
# follows DataEval's ONNX/LiteRT contract needs no `postprocess_fn`, unlike the
# [tutorial](./tt_detect_drift_with_uncertainty.py): {class}`.OnnxImageClassifier` and {class}`.ScoresExtractor` decode
# its predictions.
#
# Use it to monitor a deployed ONNX or LiteRT model without operational labels, or to reuse a model that ships with a
# `model-metadata.json`.
#
# You need `dataeval[onnx]` and `maite-datasets`, and `torch` to export the demo model.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # %pip install -q dataeval[onnx] maite-datasets
except Exception:
    pass

# %%
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from maite_datasets.image_classification import CIFAR10, MNIST
from torch import nn

from dataeval.config import set_seed
from dataeval.data import Indices, Limit, Shuffle, View
from dataeval.extractors import ScoresExtractor, UncertaintyExtractor
from dataeval.models import OnnxImageClassifier
from dataeval.shift import DriftWasserstein

set_seed(0, all_generators=True)

# %% [markdown]
# ## Prepare the model and its metadata
#
# {class}`.OnnxImageClassifier` wraps two artifacts you would normally already have: the model file (`.onnx`, or
# `.tflite` for {class}`.LiteRtImageClassifier`) and a `model-metadata.json` that declares the input and output
# contract. To stay self-contained, this guide trains and exports a small MNIST classifier in the collapsed cells below.
# In your own workflow, skip them and point the classifier at the model and metadata you already have.


# %% tags=["hide-input"]
class TinyCNN(nn.Module):
    """A small MNIST classifier emitting raw logits."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WithSoftmax(nn.Module):
    """Wrap a logits model so its ONNX ``scores`` output is a probability distribution."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.base(x), dim=1)


# %% tags=["hide-input"]
# Train the demo classifier on a small slice of MNIST (a few seconds on CPU).
mnist_train = MNIST("./data", image_set="train", download=True)
train_ids = np.random.default_rng(0).permutation(len(mnist_train))[:4000]
x_train = torch.tensor(np.stack([np.asarray(mnist_train[i][0], dtype=np.float32) / 255.0 for i in train_ids]))
y_train = torch.tensor(np.stack([np.asarray(mnist_train[i][1]).argmax() for i in train_ids]))

model = TinyCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
model.train()
for _ in range(3):
    order = torch.randperm(len(x_train))
    for start in range(0, len(x_train), 128):
        batch = order[start : start + 128]
        optimizer.zero_grad()
        loss_fn(model(x_train[batch]), y_train[batch]).backward()
        optimizer.step()
model.eval()

# %% [markdown]
# The exported model emits a softmax `scores` output, which is what the classification contract expects. The metadata
# declares a single-channel (`GRAYSCALE`) 28x28 input and 10 output classes.

# %%
Path("data").mkdir(exist_ok=True)
model_path = "data/mnist-demo.onnx"
metadata_path = "data/mnist-demo-metadata.json"

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    torch.onnx.export(
        WithSoftmax(model).eval(),
        (torch.zeros(1, 1, 28, 28),),
        model_path,
        input_names=["image"],
        output_names=["scores"],
        dynamic_axes={"image": {0: "batch"}, "scores": {0: "batch"}},
        opset_version=13,
        dynamo=False,
    )

_ = Path(metadata_path).write_text(
    json.dumps({
        "interface": {"name": "JATIC_ONNX", "version": "v1"},
        "io": {
            "batchSize": -1,
            "interface": "IMAGE_CLASSIFICATION",
            "input": {"channels": "GRAYSCALE", "height": 28, "width": 28},
            "output": {"nClasses": 10},
        },
    }),
    encoding="utf-8",
)

# %% [markdown]
# ## Load the datasets
#
# Two disjoint MNIST slices, `train` and `val`, calibrate the baseline for normal variation between same-distribution
# samples. A third, `test`, should show no drift. The `cifar` split is CIFAR-10 imagery, a shifted operational stream
# that should. The model's input builder converts CIFAR-10's RGB 32x32 images to the 28x28 grayscale that the metadata
# declares.

# %%
SAMPLES_PER_SPLIT = 200

mnist_test = MNIST("./data", image_set="test", download=True)

# Three disjoint, exchangeable MNIST slices: two references that calibrate the baseline
# (train, val) and a held-out set that should NOT drift (test).
perm = np.random.default_rng(1).permutation(len(mnist_test))
train_idx, val_idx, test_idx = np.array_split(perm[: 3 * SAMPLES_PER_SPLIT], 3)
trainset = View(mnist_test, Indices(train_idx.tolist()))
valset = View(mnist_test, Indices(val_idx.tolist()))
testset = View(mnist_test, Indices(test_idx.tolist()))

# Shifted "operational" data: CIFAR-10 natural images
cifarset = View(CIFAR10("./data", image_set="test", download=True), [Shuffle(0), Limit(SAMPLES_PER_SPLIT)])

print(f"train: {len(trainset)}  val: {len(valset)}  test: {len(testset)}  cifar: {len(cifarset)}")

# %% [markdown]
# ## Build the uncertainty feature extractor
#
# Three pieces compose into one feature extractor:
#
# 1. {class}`.OnnxImageClassifier` loads the model and its metadata and returns one `(n_classes,)` score array per
#    image.
# 1. {class}`.ScoresExtractor` adapts that model into a {class}`~dataeval.protocols.FeatureExtractor`, stacking the
#    scores into an `(n_images, n_classes)` array.
# 1. {class}`.UncertaintyExtractor` converts each row of class probabilities into one normalized-entropy value.
#
# Because the model outputs a softmax distribution, use `preds_type="probs"`.

# %%
classifier = OnnxImageClassifier(model_path, metadata_path)
scores = ScoresExtractor(classifier)
uncertainty = UncertaintyExtractor(scores, preds_type="probs", normalize=True)

print(uncertainty)

# %% [markdown]
# ## Compute and compare the uncertainty
#
# Calling the extractor on a dataset returns an `(n_images, 1)` array of normalized entropy, one value per image.

# %%
h_train = uncertainty(trainset)
h_val = uncertainty(valset)
h_test = uncertainty(testset)
h_cifar = uncertainty(cifarset)

print(
    f"mean entropy -> train: {h_train.mean():.3f}  val: {h_val.mean():.3f}  "
    f"test: {h_test.mean():.3f}  cifar: {h_cifar.mean():.3f}"
)

# %% [markdown]
# An empirical CDF (ECDF) makes the shift easy to read: if the CIFAR-10 curve sits to the right of the MNIST curves, the
# model is more uncertain on the shifted imagery. The in-distribution splits should overlap closely.

# %%
plt.figure()
for label, h in [("train", h_train), ("val", h_val), ("test", h_test), ("cifar", h_cifar)]:
    plt.ecdf(h.flatten(), label=label)
plt.xlabel("Normalized entropy")
plt.ylabel("Cumulative density")
plt.title("Prediction uncertainty by split")
plt.legend()
plt.show()

# %% [markdown]
# The MNIST splits cluster together while CIFAR-10 is shifted toward higher entropy.

# %% [markdown]
# ## Detect drift with Wasserstein distance
#
# {class}`.DriftWasserstein` measures the {term}`Wasserstein distance<Wasserstein Distance>` between the reference and
# incoming uncertainty distributions and flags drift when it grows beyond a calibrated baseline. Unlike most detectors,
# it takes two in-distribution references in `fit()`, a training set and a validation set. Their distance defines the
# normal variation between two same-distribution samples, so the detector alarms only when incoming data differs from
# training more than validation did.

# %%
drift_detector = DriftWasserstein().fit(np.asarray(h_train), np.asarray(h_val))

result_test = drift_detector.predict(np.asarray(h_test))
print(f"MNIST held-out -> drift: {result_test.drifted}  (ratio: {result_test.distance:.2f})")

result_cifar = drift_detector.predict(np.asarray(h_cifar))
print(f"CIFAR-10       -> drift: {result_cifar.drifted}  (ratio: {result_cifar.distance:.2f})")

# %% [markdown]
# The held-out MNIST slice stays within the calibrated baseline (no drift), while CIFAR-10 exceeds it (drift detected).
# The detector turned the visual gap in the ECDF into an automated, label-free alert.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert h_cifar.mean() > h_test.mean()
assert result_cifar.drifted
assert not result_test.drifted

# %% [markdown]
# ## Next steps
#
# - [Uncertainty-based drift detection](../concepts/DistributionShift.md#uncertainty-based-drift-detection) — Explore
#   Uncertainty-based drift detection.
# - [Detect drift with prediction uncertainty](./tt_detect_drift_with_uncertainty.py) — Monitor deployed object
#   detectors for drift using prediction uncertainty.
# - [How to measure train and test dataset divergence](./h2_measure_divergence.py) — Calculate divergence metrics
#   between baseline and operational data distributions.
