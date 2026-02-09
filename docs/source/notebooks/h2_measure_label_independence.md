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

# How to measure label independence

+++

## _Problem Statement_

For machine learning tasks, a discrepancy in label frequencies between train and test datasets can result in poor model
performance.

To help with this, DataEval has a tool that compares the label distributions of two datasets.

+++

### _When to use_

DataEval provides a {func}`.label_parity` function to use when you would like to determine if two datasets have
statistically independent labels.

+++

### _What you will need_

1. A Python environment with the following packages installed:
   - dataeval
1. A labeled training image dataset
1. A labeled test image dataset to evaluate the label distribution of

+++

### _Setting up_

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
except Exception:
    pass
```

```{code-cell} ipython3
from maite_datasets.image_classification import MNIST

from dataeval import Metadata
from dataeval.core import label_parity
```

## Load the data

While you can use your own dataset, for this example we imported the `MNIST` dataset and will use it going forward. It
was imported from the DataEval utils package.

```{code-cell} ipython3
train_ds = MNIST("./data", image_set="train", download=True)
test_ds = MNIST("./data", image_set="test", download=True)

train_md = Metadata(train_ds)
test_md = Metadata(test_ds)

# Get the labels from the collated dataset targets
train_labels = train_md.class_labels
test_labels = test_md.class_labels
```

## Evaluate label statistical independence

Now, let's look at how to use DataEval's label statistics analyzer. Using the {func}`.label_parity` function, compute
the chi-squared value of hypothesis that test_ds has the same class distribution as train_ds by specifying the labels of
the two datasets to be compared. It also returns the p-value of the test.

```{code-cell} ipython3
results = label_parity(train_labels, test_labels)
print(
    f"The chi-squared value for the two label distributions is {results['chi_squared']}, "
    f"with p-value {results['p_value']}"
)
```

```{code-cell} ipython3
---
tags: [remove_cell]
---
### TEST ASSERTION CELL ###
assert 0 <= results["chi_squared"] < 4
assert 0.9 < results["p_value"] <= 1.0
```
