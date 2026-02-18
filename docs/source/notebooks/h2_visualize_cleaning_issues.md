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

# How to visualize cleaning issues

+++

## Problem statement

Exploratory data analysis (EDA) can be overwhelming. There are so many things to check. Duplicates in your dataset,
bad/corrupted images in the set, blurred or bright/dark images, the list goes on.

DataEval created a [data cleaning](../concepts/DataCleaning.md) class to assist you with your EDA so you can start
training your models on high quality data.

+++

### When to use

The cleaning class should be used during the initial EDA process or if you are trying to verify that you have the right
data in your dataset.

+++

### What you will need

1. A dataset to analyze
1. A Python environment with the following packages installed:
   - `dataeval`
   - `maite-datasets`

+++

## Getting started

Let's import the required libraries needed to set up a minimal working example

```{code-cell} ipython3
---
tags: [remove_cell]
---
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    %pip install -q dataeval maite-datasets
except Exception:
    pass
```

```{code-cell} ipython3
import polars as pl
from maite_datasets.image_classification import CIFAR10

from dataeval import Metadata
from dataeval.quality import Outliers

_ = pl.Config.set_tbl_rows(-1)
```

## Loading in the data

We are going to start by loading in the CIFAR-10 dataset.

The CIFAR-10 dataset contains 60,000 images - 50,000 in the train set and 10,000 in the test set. For the purposes of
this demonstration, we are just going to use the test set.

```{code-cell} ipython3
# Load in the CIFAR10 dataset
testing_dataset = CIFAR10("./data", image_set="test", download=True)
```

## Cleaning the dataset

Now we can begin finding those images which are significantly different from the rest of the data.

```{code-cell} ipython3
# Initialize the Duplicates class
outliers = Outliers(outlier_method="zscore", outlier_threshold=3.5)

# Evaluate the data
results = outliers.evaluate(testing_dataset)
```

The results are a dictionary with the keys being the image that has an issue in one of the listed properties below:

- Brightness
- Blurriness
- Missing
- Zero
- Width
- Height
- Size
- Aspect Ratio
- Channels
- Depth

```{code-cell} ipython3
print(f"Total number of images with an issue: {len(results.aggregate_by_item())}")
```

```{code-cell} ipython3
# View issues by metric
results.aggregate_by_metric()
```

```{code-cell} ipython3
# View issues by class
results.aggregate_by_class(Metadata(testing_dataset))
```

```{code-cell} ipython3
---
tags: [remove_cell]
---
### TEST ASSERTION CELL ###
assert results.issues.shape[0] == 499
```
