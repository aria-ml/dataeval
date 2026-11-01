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
# # Measure dataset sufficiency for image classification
#
# When data collection is expensive, you can evaluate a model on a small, preliminary dataset and extrapolate its
# performance to a larger one. {class}`.Sufficiency` trains your model on growing subsets of the data to build
# [sufficiency](../concepts/PerformanceLimits.md#sufficiency-the-diminishing-returns-curve) curves, then projects them.
#
# Use it when you want to know whether collecting more data is worthwhile.
#
# You need a model architecture, the metrics you want to evaluate, a dataset and `tabulate`.

# %% tags=["remove_cell"]
# Google Colab Only
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval maite-datasets
    # !export LC_ALL="en_US.UTF-8"
    # !export LD_LIBRARY_PATH="/usr/lib64-nvidia"
    # !export LIBRARY_PATH="/usr/local/cuda/lib64/stubs"
    # !ldconfig /usr/lib64-nvidia
except Exception:
    pass

# %%
from collections.abc import Sequence
from typing import Any, cast

import dataeval_plots as dep
import numpy as np
import plotly.io as pio
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from IPython.display import display
from maite_datasets.image_classification import MilitaryVehicles
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset as TorchDataset
from torchvision.models import ResNet18_Weights, resnet18

from dataeval import Embeddings, Metadata, config
from dataeval.extractors import TorchExtractor
from dataeval.performance import Sufficiency
from dataeval.protocols import Dataset, DatumMetadata

DatumType = tuple[NDArray[np.number[Any]], NDArray[np.number[Any]], DatumMetadata]

# Set seed for reproducibility
config.set_seed(0, all_generators=True)

# Set hardware based on system
device = "cuda" if torch.cuda.is_available() else "cpu"
config.set_device(device=device)

# Use plotly to render plots, with the notebook renderer so the JS is embedded
dep.set_default_backend("plotly")
pio.renderers.default = "notebook"

# %% [markdown]
# ## Load the data and create the model
#
# Before you calculate the sufficiency of a dataset, load the dataset and define the model architecture. You use the
# MilitaryVehicles `train` and `test` splits, 7,823 and 1,621 photographs of 24 vehicle types.
#
# The model is a linear layer on the features of a frozen, pre-trained ResNet18, made as in
# [Create embeddings](./h2_create_embeddings.py). Each of the many training runs then takes a fraction of a second.

# %%
train_ds = MilitaryVehicles("./data", image_set="train", download=True, lazy=True)
test_ds = MilitaryVehicles("./data", image_set="test", download=True, lazy=True)

weights = ResNet18_Weights.DEFAULT
extractor = TorchExtractor(
    resnet18(weights=weights, progress=False), transforms=weights.transforms(), layer_name="avgpool"
)
train_embeddings = np.asarray(Embeddings(train_ds, extractor=extractor, batch_size=64))
test_embeddings = np.asarray(Embeddings(test_ds, extractor=extractor, batch_size=64))
train_labels = Metadata(train_ds).class_labels
test_labels = Metadata(test_ds).class_labels

# %% [markdown]
# A test photograph that has a near-copy in the training set measures memory and not generalization, so remove it. Here
# a near-copy is a test image whose cosine similarity to a training image is at least 0.95. Then standardize the
# features with the training statistics.

# %%
NUM_CLASSES = len(train_ds.metadata["index2label"])


def unit(x: np.ndarray) -> np.ndarray:
    """Scale each row to unit length."""
    return x / np.linalg.norm(x, axis=1, keepdims=True)


has_copy = (unit(test_embeddings) @ unit(train_embeddings).T).max(axis=1) >= 0.95
print(f"{has_copy.sum()} of {len(has_copy)} test images have a near-copy in the training set")

mean, std = train_embeddings.mean(axis=0), train_embeddings.std(axis=0) + 1e-6
train_features = ((train_embeddings - mean) / std).astype(np.float32)
test_features = ((test_embeddings - mean) / std).astype(np.float32)[~has_copy]
test_labels = test_labels[~has_copy]


class EmbeddingsDataset:
    """Features and one-hot labels as a dataset of (input, target, metadata) items."""

    def __init__(self, features: NDArray[np.float32], labels: NDArray[np.intp]) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> DatumType:
        target = np.zeros(NUM_CLASSES, dtype=np.float32)
        target[self.labels[index]] = 1.0
        return self.features[index], target, DatumMetadata(id=index)


# The evaluation set, and a random sample of 2,000 training images to build the learning curve from
evaluation_set = EmbeddingsDataset(test_features, test_labels)
sample = np.random.default_rng(0).permutation(len(train_features))[:2000]
sample_set = EmbeddingsDataset(train_features[sample], train_labels[sample])

model = nn.Linear(train_features.shape[1], NUM_CLASSES).to(device)

# %% [markdown]
# ## Define the strategies
#
# Training and evaluation functions are heavily dependent on the hyperparameters defined by a user. These can include
# metrics, loss functions, optimizers, model architectures, input sizes, etc.
#
# To allow the Sufficiency class to handle this situation, DataEval uses
# [protocols](https://typing.python.org/en/latest/spec/protocol.html). Sufficiency requires two specific protocols
# called {class}`.TrainingStrategy` and {class}`.EvaluationStrategy`.\
# Below, you define the strategies that align with this notebook and combine them into a {class}`.Sufficiency.Config`
# that can be given to the `Sufficiency` class.
#
# ### Training strategy


# %%
class TrainingStrategy:
    def train(self, model: nn.Module, dataset: Dataset[DatumType], indices: Sequence[int]):
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-2)

        # Define the dataloader for training
        dataloader = DataLoader(Subset(cast(TorchDataset, dataset), indices), batch_size=128, shuffle=True)

        for _epoch in range(30):
            for batch in dataloader:
                # Load features and one-hot encoded targets to device
                X = torch.Tensor(batch[0]).to(device)
                y = torch.argmax(torch.asarray(batch[1], dtype=torch.int).to(device), dim=1)
                optimizer.zero_grad()
                criterion(model(X), y).backward()
                optimizer.step()


# %% [markdown]
# ### Evaluation strategy
#
# Report the accuracy and the balanced accuracy, the mean of the per-class recalls, since the classes range from 119 to
# 424 photographs.


# %%
class EvaluationStrategy:
    def evaluate(self, model: nn.Module, dataset: Dataset[DatumType]) -> dict[str, float]:
        # Metrics of interest
        metrics = {
            "Accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device),
            "Balanced accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES, average="macro").to(
                device
            ),
        }
        # Set model layers into evaluation mode
        model.eval()
        dataloader = DataLoader(cast(TorchDataset, dataset), batch_size=128)
        # Tell PyTorch to not track gradients, greatly speeds up processing
        with torch.no_grad():
            for batch in dataloader:
                X = torch.Tensor(batch[0]).to(device)
                y = torch.argmax(torch.asarray(batch[1], dtype=torch.int).to(device), dim=1)
                preds = model(X)
                for metric in metrics.values():
                    metric.update(preds, y)
        return {name: float(metric.compute().cpu()) for name, metric in metrics.items()}


# %% [markdown]
# ### Reset strategy
#
# The `Sufficiency` class resets the model between runs with a `reset_strategy`. If you do not pass one, it restores a
# copy of the model taken when you create `Sufficiency`. This notebook passes its own, which re-initializes the
# parameters so that each run starts from a fresh initialization. Here is a simple implementation for PyTorch models:


# %%
def reset_model(model: nn.Module) -> nn.Module:
    """Reset all parameters in a PyTorch model."""

    @torch.no_grad()
    def weight_reset(m: nn.Module) -> None:
        reset_fn = getattr(m, "reset_parameters", None)
        if callable(reset_fn):
            reset_fn()

    return model.apply(fn=weight_reset)


# %% [markdown]
# ### Sufficiency config
#
# Do not forget to initialize your strategy classes!

# %%
training_strategy = TrainingStrategy()
evaluation_strategy = EvaluationStrategy()

config_ = Sufficiency.Config(
    training_strategy=training_strategy,
    evaluation_strategy=evaluation_strategy,
    reset_strategy=reset_model,
    runs=5,
    substeps=10,
)

# %% [markdown]
# ## Evaluate sufficiency
#
# Attach the custom training and evaluation functions to the Sufficiency metric and define the number of models to train
# in parallel (stability), as well as the number of steps along the learning curve to evaluate.

# %%
# Instantiate sufficiency metric
suff = Sufficiency(
    model=model,
    config=config_,
)

# %% [markdown]
# Now evaluate the metric to train the models and produce the learning curve. Each run draws random training indices
# with replacement, so the class order of the dataset does not matter.

# %%
# Train & test model
output = suff.evaluate(sample_set, evaluation_set)

# %%
# Print out sufficiency output in a table format
output.to_dataframe()

# %%
# Plot the output using dataeval-plots library
for plot in dep.plot(output, backend="plotly"):
    display(plot)

# %% [markdown]
# The curve rises quickly and flattens. Using it, you can project performance under larger datasets with the same model,
# and check the projection against a model you train on all 7,823 images.

# %%
projection = output.project([4000, len(train_features), 20000])
display(projection)

# %%
full_set = EmbeddingsDataset(train_features, train_labels)
rng = np.random.default_rng(1)

measured = []
for _run in range(5):
    run_model = reset_model(model)
    training_strategy.train(run_model, full_set, rng.integers(0, len(full_set), size=len(full_set)).tolist())
    measured.append(evaluation_strategy.evaluate(run_model, evaluation_set)["Accuracy"])

print(f"Projected accuracy at {len(full_set)} images: {projection['Accuracy'][1]:.3f}")
print(
    f"Measured accuracy at {len(full_set)} images: {np.mean(measured):.3f} (range {min(measured):.3f} to {max(measured):.3f})"
)

# %% [markdown]
# The curve built on 2,000 images projects the accuracy at 7,823 images to within a point of what the model reaches,
# well inside the spread between runs. That check reaches about four times past the data in the curve. The projections
# beyond it are further extrapolations, so treat them as orders of magnitude.

# %% [markdown]
# ## Predict sample requirements
#
# You can also predict the amount of training samples required to achieve specific performance thresholds.
#
# Suppose you want to see how many samples are needed to hit 40%, 45% and 50% accuracy and balanced accuracy.

# %%
# Initialize the array of desired thresholds to apply to all metrics
needed = output.inv_project([0.40, 0.45, 0.50])
display(needed)

# %% [markdown]
# Each further five points of accuracy takes about four times as many images: 40% needs about 12,000, 45% about 45,000
# and 50% about 190,000. More photographs of this kind raise the accuracy of this model slowly. A better model, such as
# a fine-tuned backbone, changes the curve, and the same evaluation shows whether it does.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
# Verify the learning curve: five runs over ten steps up to the 2,000 sampled training images, and the model learns
metrics = {"Accuracy", "Balanced accuracy"}
curve = output.to_dataframe()
assert set(curve.columns) == {"step", *metrics}
assert curve.height == 10
assert curve["step"][-1] == len(sample_set) == 2000
assert {name: measure.shape for name, measure in output.measures.items()} == dict.fromkeys(metrics, (5, 10))
assert curve["Accuracy"][-1] > 2 * curve["Accuracy"][0]
assert abs(int(has_copy.sum()) - 167) <= 5  # the text removes the near-copies
assert len(evaluation_set) == len(test_ds) - int(has_copy.sum())

# Verify the projections: valid values that do not fall as the dataset grows, and that match the measurement
for metric in metrics:
    values = projection[metric].to_list()
    assert min(values) >= 0
    assert max(values) <= 1
    assert values == sorted(values)
assert projection["step"].to_list() == [4000, 7823, 20000]
gap = abs(projection["Accuracy"][1] - np.mean(measured))
assert gap < 0.01  # "within a point"
assert gap < max(measured) - min(measured)  # "inside the spread between runs"

# Verify the sample requirements: each five points costs several times more data
assert needed["target"].to_list() == [0.40, 0.45, 0.50]
first, second, third = needed["Accuracy"].to_list()
assert 10_000 < first < 14_000  # "about 12,000"
assert 38_000 < second < 52_000  # "about 45,000"
assert 150_000 < third < 230_000  # "about 190,000"
assert 3 < second / first < 5  # "about four times"
assert 3 < third / second < 5

# %% [markdown]
# ## Next steps
#
# - [Performance limits](../concepts/PerformanceLimits.md) — Understand sufficiency curves and diminishing returns in
#   dataset scaling.
# - [Assess whether a classification requirement is reachable](./tt_assess_ic_feasibility.py) — Check a requirement
#   against the evidence in your images before you collect more data.
# - [Identify gaps in a labeled dataset](./tt_identify_dataset_gaps.py) — Find sparse or underrepresented classes and
#   regions in the feature space.
