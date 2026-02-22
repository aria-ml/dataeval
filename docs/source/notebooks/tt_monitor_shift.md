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

# Monitor shifts in operational data

This guide provides a beginner friendly introduction on monitoring post deployment data shifts.

Estimated time to complete: 5 minutes

Relevant ML stages: [Monitoring](../concepts/users/ML_Lifecycle.md#monitoring)

Relevant personas: Machine Learning Engineer, T&E Engineer

## What you'll do

- Construct [embeddings](../concepts/Embeddings.md) by training a simple neural network
- Compare the embeddings between a training and operational set
- Use chunked drift detection to monitor drift across data segments
- Compare the label distributions between a training and operational set

## What you'll learn

- Learn how to analyze embeddings for operational drift
- Learn how to use chunked drift detection for temporal monitoring
- Learn how to analyze label distributions

## What you'll need

- Knowledge of Python
- Beginner knowledge of PyTorch or neural networks

+++

## Introduction

Monitoring is a critical step in the [AI/ML lifecycle](../concepts/users/ML_Lifecycle.md). When a model is deployed,
data can, and generally will, [drift](../concepts/Drift.md) from the distribution on which the model was originally
trained. One critical step in AI T&E is the detection of changes in the operational distribution so that they may be
proactively addressed. While some change might not affect performance, significant deviation is often associated with
model degradation.

For this tutorial, you will use the popular
[2012 VOC](https://huggingface.co/datasets/HuggingFaceM4/pascal_voc/tree/main) computer vision dataset to detect drift
between the image distribution of the `train` split and the `val` split, which will represent an operational dataset in
this guide. You will then determine if the labels within these two datasets has high
[parity](../concepts/LabelParity.md), or equivalent label distributions.

+++

## Setup

You'll begin by importing the necessary libraries for this tutorial.

```{code-cell} ipython3
---
tags: [remove_cell]
---
try:
    import google.colab  # noqa: F401

    %pip install -q dataeval maite-datasets
except Exception:
    pass
```

```{code-cell} ipython3
import numpy as np
import torch
from IPython.display import display  # noqa: A004
from maite_datasets.object_detection import VOCDetection
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms.v2 import GaussianNoise

from dataeval import Embeddings, Metadata
from dataeval.core import label_parity
from dataeval.extractors import TorchExtractor
from dataeval.shift import DriftChunkedOutput, DriftMMD, DriftMVDC, DriftUnivariate

# Set a random seed
rng = np.random.default_rng(213)

# Set default torch device for notebook
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
```

> **More on device**
>
> The device is set above as it will be used in subsequent steps. The device is the piece of hardware where the model,
> data, and other related objects are stored in memory. If a GPU is available, this notebook will use that hardware
> rather than the CPU. To force running only on the CPU, change `device` to `"cpu"` For more information, see the
> [PyTorch device page](https://pytorch.org/tutorials/recipes/recipes/changing_default_device.html).

+++

## Constructing embeddings

An important concept in many aspects of machine learning is {term}`Dimensionality Reduction`. While this step is not
always necessary, it is good practice to use embeddings over raw images to improve the speed and memory efficiency of
many workflows without sacrificing downstream performance.

### Define model architecture

In this section, you will use a
[pretrained ResNet18 model](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) from
Torchvision to reduce the dimensionality of the VOC dataset.

```{code-cell} ipython3
resnet = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False)

# Replace the final fully connected layer with a Linear layer
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 128)
```

### Download VOC dataset

With the model created on the device set at the beginning, you will download the train and validation splits of the 2012
VOC Dataset.

```{code-cell} ipython3
# Load the training dataset
train_ds = VOCDetection("./data", year="2012", image_set="train", download=True)
print(train_ds)
print(f"Image 0 shape: {train_ds[0][0].shape}")
```

```{code-cell} ipython3
# Load the "operational" dataset
operational_ds = VOCDetection("./data", year="2012", image_set="val", download=True)
print(operational_ds)
print(f"Image 0 shape: {train_ds[0][0].shape}")
```

It is good to notice a few points about each dataset:

- Number of datapoints
- Resize size

These two values give an estimate of the memory impact that each dataset has. The following step will modify the resize
size by creating model embeddings for each image to reduce this impact.

+++

### Extract embeddings

Now it is time to process the datasets through your model. Aggregating the model outputs gives you the embeddings of the
data. This will be helpful in determining drift between the training and operational splits.

+++

Below you will call the helper function and create embeddings for both the train and operational splits. The labels will
also be saved so they can be used in a later step.

```{code-cell} ipython3
# Define pretrained model transformations
transforms = ResNet18_Weights.DEFAULT.transforms()

# Create extractor with model and transforms
extractor = TorchExtractor(resnet, transforms=transforms)

# Create training batches and targets
train_embs = Embeddings(train_ds, extractor=extractor, batch_size=64)

# Create operational batches and targets
operational_embs = Embeddings(operational_ds, extractor=extractor, batch_size=64)
```

Notice that the shape of embeddings is different than before.

#### Previously

Training shape - (5717, 3, 442, 500)\
Operational shape - (5823, 3, 442, 500)

#### After embeddings

```{code-cell} ipython3
print(f"({len(train_embs)}, {train_embs[0].shape})")  # (5717, shape)
print(f"({len(operational_embs)}, {operational_embs[0].shape})")  # (5823, shape)
```

The reduced shape of both the training and operational datasets will improve the performance of the upcoming drift
algorithms without impacting the accuracy of the results.

+++

## Test for drift

In this step, you will be checking for drift between the training embeddings and the operational embeddings from before.
If drift is detected, a model trained on this training data should be retrained with new operational data. This can help
mitigate performance degradation in a deployed model. Visit our [About Drift](../concepts/Drift.md) page to learn more.

### Drift detectors

DataEval offers a few drift detectors: {class}`.DriftMMD`, {class}`.DriftUnivariate`, and {class}`.DriftMVDC`.

Since each detector outputs a binary decision on whether drift is detected, a **majority vote** will be used to make the
determination of drift.\
To learn more about these algorithms, see the [theory behind drift detection](../concepts/Drift.md#what-is-drift)
concept page.

### Fit the detectors

Each drift detector needs a reference set that the operational set will be compared against. In the following code, you
will set the reference data to the training embeddings.

```{code-cell} ipython3
# A type alias for all of the drift detectors
DriftDetector = DriftMMD | DriftUnivariate | DriftMVDC

# Create a mapping for the detectors to iterate over
detectors: dict[str, DriftDetector] = {
    "MMD": DriftMMD().fit(train_embs),
    "CVM": DriftUnivariate(method="cvm").fit(train_embs),
    "KS": DriftUnivariate(method="ks").fit(train_embs),
    "MVDC": DriftMVDC().fit(train_embs),
}
```

### Make predictions

Now that the detectors are setup, predictions can be made against the operational embeddings you made earlier.

```{code-cell} ipython3
# Iterate and print the name of the detector class and its boolean drift prediction
for name, detector in detectors.items():
    print(f"{name} detected drift? {detector.predict(operational_embs).drifted}")
```

Did you expect these results?

There is no drift detected between the train and operational embeddings because they come from very similar
distributions.\
Ideally, your training data and your validation data, which we used as operational, come from the same distribution.
This is the purpose of [data splitters](https://scikit-learn.org/stable/api/sklearn.model_selection.html#splitters).

So how do we know if the detectors can detect drift?

Well, add some random Gaussian noise to the operational embeddings and find out.

```{code-cell} ipython3
# Define transform with added gaussian noise
noisy_transforms = [transforms, GaussianNoise()]

# Create extractor with noisy transforms
noisy_extractor = TorchExtractor(resnet, transforms=noisy_transforms)

# Applies gaussian noise to images before processing
noisy_embs = Embeddings(operational_ds, extractor=noisy_extractor, batch_size=64)
```

```{code-cell} ipython3
# Iterate and print the name of the detector class and its boolean drift prediction
for name, detector in detectors.items():
    print(f"{name} detected drift? {detector.predict(noisy_embs).drifted}")
```

Now drift is detected!

Adding Gaussian noise was enough to cause a noticeable change in the drift detectors, but this is not always the case.
There are many [types of drift](../concepts/Drift.md#formal-definition-and-types-of-drift) that data can and will
experience.

In this step, you learned how to take your generated embeddings and detect drift between the training and operational
image data. While there was no drift originally, you were able to add small perturbations to the data that did affect
the data distributions and cause drift.

+++

### Monitor drift over time with chunking

In real deployments, operational data arrives in batches over time. Rather than comparing all operational data at once,
you can use **chunking** to split the data into segments and monitor how drift evolves across each chunk. This helps
identify _when_ drift begins to appear.

DataEval's drift detectors support chunking through the `chunk_count` or `chunk_size` parameters on `fit()`. During
fitting, the detector establishes a baseline by computing the metric across chunks of the reference data. During
prediction, each chunk of test data is compared against this baseline, returning a {class}`.DriftChunkedOutput` with
per-chunk results.

#### Simulate gradual drift onset

To illustrate how chunking reveals _when_ drift begins, you will build a combined dataset where the first 40% of samples
are clean operational embeddings and the remaining 60% are noisy. This simulates a scenario where data quality degrades
partway through a monitoring window.

```{code-cell} ipython3
# Build a combined array: first 40% clean, last 60% noisy
n_operational = len(operational_embs)
split_idx = int(n_operational * 0.4)

combined_embs = np.concatenate([operational_embs[:split_idx], noisy_embs[split_idx:]])
print(f"Combined shape: {combined_embs.shape} (clean: {split_idx}, noisy: {n_operational - split_idx})")
```

#### Fit detectors with chunking

```{code-cell} ipython3
# Re-fit detectors with chunking enabled (5 chunks each)
chunked_detectors: dict[str, DriftDetector] = {
    "MMD": DriftMMD().fit(train_embs, chunk_count=5),
    "CVM": DriftUnivariate(method="cvm").fit(train_embs, chunk_count=5),
    "KS": DriftUnivariate(method="ks").fit(train_embs, chunk_count=5),
    "MVDC": DriftMVDC(threshold=(0.45, 0.65)).fit(train_embs, chunk_count=5),
}
```

#### Predict on combined data and display chunk results

```{code-cell} ipython3
for name, detector in chunked_detectors.items():
    result = detector.predict(combined_embs)
    print(f"\n{name} - Overall drift detected: {result.drifted} (metric: {result.metric_name})")
    if isinstance(result, DriftChunkedOutput):
        display(result.chunk_results)
```

The first two chunks (covering the clean 40%) should show no drift, while the later chunks (covering the noisy 60%)
should trigger drift alerts. This chunk-level view makes it easy to pinpoint _when_ in a data stream drift begins.

Next you will look at the labels' distributions.

+++

## Evaluate parity

+++

Instead of looking at the images, you can compare the distributions of the labels using a method called
[label parity](../concepts/LabelParity.md).\
There is parity between two sets of labels if the label frequencies are approximately equal.

You will now compare the label distributions using the `label_parity` function.

```{code-cell} ipython3
# Get the metadata for each dataset
train_md = Metadata(train_ds)
operational_md = Metadata(operational_ds)

# The VOC dataset has 20 classes
label_parity(train_md.class_labels, operational_md.class_labels, num_classes=20)["p_value"]
```

From the {func}`.label_parity` function, you can see that it calculated a p_value of ~**0.95**. Since this is close to
1.0, it can be said that the two distributions **have** class label parity, or similar distributions.

+++

## Conclusion

In this tutorial, you have learned to create embeddings from the VOC dataset, look for drift between two sets of data,
and calculate the parity of two label distributions. These are important steps when monitoring data as drift and lack of
parity can affect a model's ability to achieve performance recorded during model training. When data drift is detected
or the label distributions lack parity, it is a good idea to consider retraining the model and incorporating operational
data into the dataset.

+++

## What's next

DataEval plays a small, but impactful role in data monitoring as a metrics library.\
Visit these additional resources for more information on other aspects:

- Read about the entire [monitoring in AI/ML](../concepts/users/ML_Lifecycle.md#monitoring) stage
- Explore DataEval's [API reference](../reference/autoapi/dataeval/index.rst) for drift and other monitoring tools
