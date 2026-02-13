---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: dataeval
  language: python
  name: python3
---

# How to measure dataset sufficiency for image classification

This guide walks you through analyzing an image classification model's hypothetical performance.

Estimated time to complete: 10 minutes

Relevant ML stages: [Model Development](../concepts/users/ML_Lifecycle.md#model-development)

Relevant personas: ML Engineer

+++

## Problem statement

For machine learning tasks, often we would like to evaluate the performance of a model on a small, preliminary dataset.
In situations where data collection is expensive, we would like to extrapolate hypothetical performance out to a larger
dataset.

DataEval has introduced a method projecting performance via _[sufficiency](../concepts/Sufficiency.md) curves_.

+++

### When to use

The {class}`.Sufficiency` class should be used when you would like to extrapolate hypothetical performance. For example,
if you have a small dataset, and would like to know if it is worthwhile to collect more data.

+++

### What you will need

1. A particular model architecture.
1. Metric(s) that we would like to evaluate.
1. A dataset of interest.
1. A Python environment with the following packages installed:
   - `tabulate`

+++

### Getting started

Let's import the required libraries needed to set up a minimal working example

```{code-cell} ipython3
---
tags: [remove_cell]
---
# Google Colab Only
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    %pip install -q dataeval maite-datasets
    !export LC_ALL="en_US.UTF-8"
    !export LD_LIBRARY_PATH="/usr/lib64-nvidia"
    !export LIBRARY_PATH="/usr/local/cuda/lib64/stubs"
    !ldconfig /usr/lib64-nvidia
except Exception:
    pass
```

```{code-cell} ipython3
import os
from collections.abc import Sequence
from typing import Any, cast

import dataeval_plots as dep
import numpy as np
import plotly.io as pio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from IPython.display import display  # noqa: A004
from maite_datasets.image_classification import MNIST
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset as TorchDataset

from dataeval import config
from dataeval.performance import Sufficiency
from dataeval.protocols import Dataset, DatumMetadata
from dataeval.selection import Limit, Select

DatumType = tuple[NDArray[np.number[Any]], NDArray[np.number[Any]], DatumMetadata]

# Set seed for reproducibility
config.set_seed(0, all_generators=True)

# Set hardware based on system
device = "cuda" if torch.cuda.is_available() else "cpu"
config.set_device(device=device)

# Additional reproducibility and printing options
np.set_printoptions(formatter={"float": lambda x: f"{x:0.4f}"})
torch.set_float32_matmul_precision("high")
torch._dynamo.config.suppress_errors = True
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Use plotly to render plots
dep.set_default_backend("plotly")

# Use the notebook renderer so JS is embedded
pio.renderers.default = "notebook"
```

## Load data and create model

Before calculating the sufficiency of a dataset, the dataset must be loaded and the model architecture defined. We will
walk through these in the following steps.

### Loading MNIST data

Load the MNIST data and split it into training and test datasets. For this notebook, we will use subsets of the training
(2500) and test (500) data.

```{code-cell} ipython3
# Configure the dataset transforms

transforms = [
    lambda x: x / 255.0,  # scale to [0, 1]
    lambda x: x.astype(np.float32),  # convert to float32
]

# Download the mnist dataset and apply the transforms and subset the data
train_ds = Select(MNIST("./data", image_set="train", transforms=transforms, download=True), selections=[Limit(2500)])
test_ds = Select(MNIST("./data", image_set="test", transforms=transforms, download=True), selections=[Limit(500)])
```

### Creating a PyTorch model

Next, we define the network architecture that will be trained and then evaluated throughout the sufficiency calculation.

```{code-cell} ipython3
# Define our network architecture
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(6400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Compile the model (cast sets the type to Net as compile returns an Unknown)
model: Net = cast(Net, torch.compile(Net().to(device)))
```

## Strategy protocols

Training and evaluation functions are heavily dependent on the hyperparameters defined by a user. These can include
metrics, loss functions, optimizers, model architectures, input sizes, etc.

To allow the Sufficiency class to handle this situation, DataEval uses
[_Protocols_](https://typing.python.org/en/latest/spec/protocol.html). Sufficiency requires two specific protocols
called {class}`.TrainingStrategy` and {class}`.EvaluationStrategy`.\
Below we will define the strategies that align with this notebook and combine them into a {class}`.Sufficiency.Config`
that can be given to the `Sufficiency` class.

### Training strategy

```{code-cell} ipython3
class MNISTTrainingStrategy:
    def train(self, model: nn.Module, dataset: Dataset[DatumType], indices: Sequence[int]):
        # Defined only for this testing scenario
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        epochs = 10

        # Define the dataloader for training
        dataloader = DataLoader(Subset(cast(TorchDataset, dataset), indices), batch_size=8)

        for _epoch in range(epochs):
            for batch in dataloader:
                # Load data/images to device
                X = torch.Tensor(batch[0]).to(device)
                # Load one-hot encoded targets/labels to device
                y = torch.argmax(torch.asarray(batch[1], dtype=torch.int).to(device), dim=1)
                # Zero out gradients
                optimizer.zero_grad()
                # Forward propagation
                outputs = model(X)
                # Compute loss
                loss = criterion(outputs, y)
                # Back prop
                loss.backward()
                # Update weights/parameters
                optimizer.step()
```

### Evaluation strategy

```{code-cell} ipython3
class MNISTEvaluationStrategy:
    def evaluate(self, model: nn.Module, dataset: Dataset[DatumType]) -> dict[str, float]:
        # Metrics of interest
        metrics = {
            "Accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device),
            "AUROC": torchmetrics.AUROC(task="multiclass", num_classes=10).to(device),
            "TPR at 0.5 Fixed FPR": torchmetrics.ROC(task="multiclass", average="macro", num_classes=10).to(device),
        }
        result = {}
        # Set model layers into evaluation mode
        model.eval()
        dataloader = DataLoader(cast(TorchDataset, dataset), batch_size=8)
        # Tell PyTorch to not track gradients, greatly speeds up processing
        with torch.no_grad():
            for batch in dataloader:
                # Load data/images to device
                X = torch.Tensor(batch[0]).to(device)
                # Load one-hot encoded targets/labels to device
                y = torch.argmax(torch.asarray(batch[1], dtype=torch.int).to(device), dim=1)
                preds = model(X)
                for metric in metrics.values():
                    metric.update(preds, y)
            # Compute ROC curve
            false_positive_rate, true_positive_rate, _ = metrics["TPR at 0.5 Fixed FPR"].compute()
            # determine interval to examine
            desired_rate = 0.5
            closest_desired_index = torch.argmin(torch.abs(false_positive_rate - desired_rate)).item()
            # return corresponding tpr value
            result["TPR at 0.5 Fixed FPR"] = true_positive_rate[closest_desired_index].cpu()
            result["Accuracy"] = metrics["Accuracy"].compute().cpu()
            result["AUROC"] = metrics["AUROC"].compute().cpu()
        return result
```

### Reset strategy

The `Sufficiency` class requires a `reset_strategy` that resets the model's parameters between runs. This ensures each
run starts from a fresh initialization. Here's a simple implementation for PyTorch models:

```{code-cell} ipython3
def reset_model(model: nn.Module) -> nn.Module:
    """Reset all parameters in a PyTorch model."""

    @torch.no_grad()
    def weight_reset(m: nn.Module) -> None:
        reset_fn = getattr(m, "reset_parameters", None)
        if callable(reset_fn):
            reset_fn()

    return model.apply(fn=weight_reset)
```

### Sufficiency config

Do not forget to initialize your strategy classes!

```{code-cell} ipython3
mnist_config = Sufficiency.Config(
    training_strategy=MNISTTrainingStrategy(),
    evaluation_strategy=MNISTEvaluationStrategy(),
    reset_strategy=reset_model,
    runs=5,
    substeps=10,
)
```

## Initialize sufficiency metric

Attach the custom training and evaluation functions to the Sufficiency metric and define the number of models to train
in parallel (stability), as well as the number of steps along the learning curve to evaluate.

```{code-cell} ipython3
# Instantiate sufficiency metric
suff = Sufficiency(
    model=model,
    train_ds=train_ds,
    test_ds=test_ds,
    config=mnist_config,
)
```

## Evaluate sufficiency

Now we can evaluate the metric to train the models and produce the learning curve.

```{code-cell} ipython3
# Train & test model
output = suff.evaluate()
```

```{code-cell} ipython3
# Print out sufficiency output in a table format
output.to_dataframe()
```

```{code-cell} ipython3
# Print out projected output values
output.project([1000, 2500, 5000])
```

```{code-cell} ipython3
# Plot the output using dataeval-plots library
for plot in dep.plot(output, backend="plotly"):
    display(plot)
```

## Results

Using these learning curves, we can project performance under much larger datasets (with the same models).

+++

## Predicting sample requirements

We can also predict the amount of training samples required to achieve specific performance thresholds.

Let's say we wanted to see how many samples are needed to hit 90%, 93%, and 99% accuracy, area under the receiver
operating characteristic, and true positive rate at a fixed false positive rate of 0.5.

```{code-cell} ipython3
# Initialize the array of desired thresholds to apply to all metrics
output.inv_project([0.90, 0.93, 0.99])
```

With a value of "-1" samples, the projection shows that given the current model, hitting an accuracy of 99% is
improbable.
