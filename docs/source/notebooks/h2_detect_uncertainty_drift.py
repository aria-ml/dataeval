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
# # Detect uncertainty drift with a MAITE model
#
# As data moves into regions that a deployed model was not trained on, the confidence of its predictions can change. A
# shift in the {term}`Shannon entropy<Shannon Entropy>` of the predicted probabilities is a label-free drift signal. See
# [uncertainty-based drift detection](../concepts/DistributionShift.md#uncertainty-based-drift-detection). A model that
# follows DataEval's ONNX/LiteRT contract needs no adapter code: {class}`.OnnxImageClassifier` and
# {class}`.ScoresExtractor` decode its predictions. For other models, such as detectors, pass a
# `postprocess_fn` to {class}`.TorchExtractor`.
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
import torch.nn.functional as F
from maite_datasets.image_classification import MilitaryVehicles
from maite_datasets.object_detection import SeaDrone
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from dataeval import Embeddings, Metadata
from dataeval.config import set_device, set_seed
from dataeval.data import Indices, Limit, Resize, Shuffle, View
from dataeval.extractors import ScoresExtractor, TorchExtractor, UncertaintyExtractor
from dataeval.models import OnnxImageClassifier
from dataeval.shift import DriftUnivariate

set_seed(0, all_generators=True)
set_device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## Prepare the model and its metadata
#
# {class}`.OnnxImageClassifier` wraps two artifacts you would normally already have: the model file (`.onnx`, or
# `.tflite` for {class}`.LiteRtImageClassifier`) and a `model-metadata.json` that declares the input and output
# contract. To stay self-contained, this guide builds and exports a small vehicle classifier in the collapsed cells
# below: a frozen ResNet18 with a linear head, fitted to 3,000 MilitaryVehicles training photographs. In your own
# workflow, skip them and point the classifier at the model and metadata you already have.


# %% tags=["hide-input"]
class Normalize(nn.Module):
    """Standardize [0, 1] RGB images with the ImageNet mean and standard deviation."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class VehicleClassifier(nn.Module):
    """A frozen backbone and a linear head, emitting a probability for each vehicle type."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.net = nn.Sequential(Normalize(), backbone)
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.head(self.net(x)), dim=1)


# %% tags=["hide-input"]
# Fit the head to the features of 3,000 training photographs, resized to the size the model takes (a few seconds).
train_ds = MilitaryVehicles("./data", image_set="train", download=True, lazy=True)
fit_ds = View(train_ds, [Shuffle(seed=0), Limit(3000)])

resnet = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False)
backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten()).eval()  # everything before the classifier
extractor = TorchExtractor(nn.Sequential(Normalize(), backbone), transforms=lambda image: image / 255.0)
features = torch.tensor(np.asarray(Embeddings(View(fit_ds, Resize((224, 224))), extractor=extractor, batch_size=64)))
targets = torch.tensor(Metadata(fit_ds).class_labels).long()

head = nn.Linear(features.shape[1], len(train_ds.metadata["index2label"]))
optimizer = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-2)
for _ in range(20):
    order = torch.randperm(len(features))
    for start in range(0, len(features), 128):
        batch = order[start : start + 128]
        optimizer.zero_grad()
        F.cross_entropy(head(features[batch]), targets[batch]).backward()
        optimizer.step()

model = VehicleClassifier(backbone.cpu(), head).eval()

# %% [markdown]
# The exported model emits a softmax `scores` output, which is what the classification contract expects. The metadata
# declares a three-channel (`RGB`) 224x224 input and 24 output classes. The classifier resizes each photograph to the
# declared size and scales it to `[0, 1]`, so the photographs need no preparation.

# %%
Path("data").mkdir(exist_ok=True)
model_path = "data/vehicles-demo.onnx"
metadata_path = "data/vehicles-demo-metadata.json"

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    torch.onnx.export(
        model,
        (torch.zeros(1, 3, 224, 224),),
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
            "input": {"channels": "RGB", "height": 224, "width": 224},
            "output": {"nClasses": 24},
        },
    }),
    encoding="utf-8",
)

# %% [markdown]
# ## Load the datasets
#
# Split the MilitaryVehicles test photographs, which the model never saw, into two disjoint random slices: a `reference`
# that calibrates the baseline for normal variation, and a `held_out` set that should not drift. The `aerial` set is 400
# SeaDrone frames, a shifted operational stream that the model was not built for.

# %%
test_ds = MilitaryVehicles("./data", image_set="test", download=True, lazy=True)

perm = np.random.default_rng(1).permutation(len(test_ds))
reference = View(test_ds, Indices(perm[:800].tolist()))
held_out = View(test_ds, Indices(perm[800:].tolist()))

# Shifted "operational" data: aerial frames of open water
aerial = View(SeaDrone("./data", image_set="val", download=True, lazy=True), [Shuffle(0), Limit(400)])

print(f"reference: {len(reference)}  held out: {len(held_out)}  aerial: {len(aerial)}")

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
h_reference = uncertainty(reference)
h_held_out = uncertainty(held_out)
h_aerial = uncertainty(aerial)

print(
    f"median entropy -> reference: {np.median(h_reference):.3f}  held out: {np.median(h_held_out):.3f}  "
    f"aerial: {np.median(h_aerial):.3f}"
)

# %% [markdown]
# An empirical CDF (ECDF) makes the shift easy to read. The two sets of test photographs should overlap closely. The
# aerial curve sits to the left, so the model is more confident on aerial frames than on its own test photographs.
# Uncertainty drift detects a change of confidence in either direction. It does not say that the model is wrong.

# %%
plt.figure()
for label, h in [("reference", h_reference), ("held out", h_held_out), ("aerial", h_aerial)]:
    plt.ecdf(h.flatten(), label=label)
plt.xlabel("Normalized entropy")
plt.ylabel("Cumulative density")
plt.title("Prediction uncertainty by dataset")
plt.legend()
plt.show()

# %% [markdown]
# ## Detect drift with a KS test
#
# {class}`.DriftUnivariate` with `method="ks"` fits the reference entropy and compares each incoming batch of 50 images
# with it. A KS test is calibrated, so about 5% of batches that come from the reference distribution are flagged. Count
# the flagged batches rather than judge one batch, because a single batch is one draw. The
# [tutorial](./tt_detect_drift_with_uncertainty.py) checks how this monitor behaves under other shifts.

# %%
BATCH = 50
detector = DriftUnivariate(method="ks").fit(np.asarray(h_reference))


def flagged(entropy: np.ndarray) -> list[bool]:
    """Whether the detector flags each consecutive batch of the entropy values."""
    return [
        detector.predict(entropy[start : start + BATCH]).drifted for start in range(0, len(entropy) - BATCH + 1, BATCH)
    ]


held_out_flags = flagged(np.asarray(h_held_out))
aerial_flags = flagged(np.asarray(h_aerial))

print(f"Held-out batches flagged: {sum(held_out_flags)} of {len(held_out_flags)}")
print(f"Aerial batches flagged:   {sum(aerial_flags)} of {len(aerial_flags)}")

# %% [markdown]
# The detector flags almost none of the held-out batches and most of the aerial ones. It turned the gap in the ECDF into
# an automated, label-free alert. The count is small, so read it as a rate with room around it and not as a precise one.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert (len(reference), len(held_out), len(aerial)) == (800, 821, 400)
assert np.median(h_aerial) < np.median(h_reference)  # "sits to the left"
assert abs(np.median(h_held_out) - np.median(h_reference)) < 0.05  # "overlap closely"
assert len(held_out_flags) == 16
assert len(aerial_flags) == 8
assert sum(held_out_flags) <= 3  # "almost none"
assert sum(aerial_flags) >= 5  # "most"

# %% [markdown]
# ## Next steps
#
# - [Uncertainty-based drift detection](../concepts/DistributionShift.md#uncertainty-based-drift-detection) — Explore
#   Uncertainty-based drift detection.
# - [Detect drift with prediction uncertainty](./tt_detect_drift_with_uncertainty.py) — Monitor a classifier for drift
#   with its own uncertainty, and check what the monitor can and cannot see.
# - [Measure train and test dataset divergence](./h2_measure_divergence.py) — Calculate divergence metrics
#   between baseline and operational data distributions.
