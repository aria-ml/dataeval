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

# How to add intrinsic factors to Metadata

+++

## Problem Statement

When performing analysis on datasets, metadata may sometimes be sparse or unavailable. Adding metadata to a dataset for
analysis may be necessary at times, and can come in the forms of calculated intrinsic values or additional information
originally unavailable on the source dataset.

This guide will show you how to add in the calculated statistics from DataEval's {func}`.calculate` function to the
metadata for bias analysis.

+++

### _When to use_

Adding metadata factors should be done when little or no metadata is available on the dataset, or to gain insights
specific to metadata of interest that is not present natively in the dataset metadata.

+++

### _What you will need_

1. A dataset to analyze
1. A Python environment with the following packages installed:
   - `dataeval`
   - `dataeval-plots[plotly]`
   - `maite-datasets`

+++

## _Getting Started_

First import the required libraries needed to set up the example.

```{code-cell} ipython3
---
tags: [remove_cell]
---
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    %pip install -q dataeval dataeval-plots[plotly] maite-datasets
except Exception:
    pass
```

```{code-cell} ipython3
import dataeval_plots as dep
import plotly.io as pio
import polars as pl
from IPython.display import display
from maite_datasets.image_classification import CIFAR10

from dataeval import Metadata
from dataeval.bias import Balance, Diversity, Parity
from dataeval.core import calculate
from dataeval.flags import ImageStats
from dataeval.selection import Limit, Select, Shuffle

_ = pl.Config.set_tbl_rows(-1)
# Use plotly to render plots
dep.set_default_backend("plotly")
dep.set_default_backend("matplotlib")  # LOL BUMP BUMP

# Use the notebook renderer so JS is embedded
pio.renderers.default = "notebook"
```

## Load the dataset

Begin by loading in the CIFAR-10 dataset.

The CIFAR-10 dataset contains 60,000 images - 50,000 in the train set and 10,000 in the test set. We will use a shuffled
sample of 20,000 images from both sets.

```{code-cell} ipython3
# Load in the CIFAR10 dataset and limit to 20,000 images with random shuffling
cifar10 = Select(CIFAR10("data", image_set="base", download=True), [Limit(20000), Shuffle(seed=0)])
print(cifar10)
```

## Inspect the metadata

You can begin by inspecting the available factor names in the dataset.

```{code-cell} ipython3
metadata = Metadata(cifar10)
print(f"Factor names: {metadata.factor_names}")
```

A quick check of the {func}`.balance` of the single factor will show no mutual information between the classes and the
`batch_num` which indicates the on-disk binary file the image was extracted from.

```{code-cell} ipython3
# Balance at index 0 is always class
Balance().evaluate(metadata).balance[2]
```

## Add image statistics to the metadata

In order to perform additional bias analysis on the dataset when no meaningful metadata are provided, you will augment
the metadata with statistics of the images using the {func}`.calculate` function.

Begin by running `calculate` for the `PIXEL` and `VISUAL` image stats for the dataset and adding the stats factors to
the `Metadata`.

```{code-cell} ipython3
# Calculate pixel and visual statistics
calc_results = calculate(cifar10, stats=ImageStats.PIXEL | ImageStats.VISUAL)

# Append the factors to the metadata
metadata.add_factors(calc_results["stats"])
```

Next you will add the `calculate` output to the metadata as factors, and exclude factors that are uniform or without
significance.

Additionally, you will specify a binning strategy for continuous statistical factors, which are, for our purposes,
continuous. For this example, bin everything into 10 uniform-width bins.

```{code-cell} ipython3
# Exclude the id and batch_num as it is not a relevant factor for bias analysis
metadata.exclude = ["id", "batch_num"]

# Provide binning for the continuous statistical factors using 5 uniform-width bins for each factor
keys = ("mean", "std", "var", "skew", "kurtosis", "entropy", "brightness", "darkness", "sharpness", "contrast", "zeros")
metadata.continuous_factor_bins = dict.fromkeys(keys, 5)
```

## Perform bias analysis

Now you can run the bias analysis evaluators {class}`.Balance`, {class}`.Diversity` and {class}`.Parity` on the dataset
metadata augmented with intrinsic statistical factors.

```{code-cell} ipython3
balance_output = Balance().evaluate(metadata)
```

```{code-cell} ipython3
dep.plot(balance_output)
```

Notice the very high mutual information between the variance and standard deviation of image intensities, which is
expected. Mean image intensity correlates with brightness, darkness, and contrast. However, none of the intrinsic
factors correlate strongly with class label.

```{code-cell} ipython3
dep.plot(balance_output, plot_classwise=True)
```

Classwise balance also indicates minimal correlation of image statistics and individual classes. Uniform mutual
information between individual classes and all class labels indicates balanced class representation in the subsampled
dataset.

```{code-cell} ipython3
diversity_output = Diversity().evaluate(metadata)
dep.plot(diversity_output)
```

The diversity index also indicates uniform sampling of classes within the dataset. The apparently low diversity of
kurtosis across the dataset may indicate an inadequate binning strategy (for metric computation) given that the other
statistical moments appear to be more evenly distributed. Further investigation and iteration could be done to assess
sensitivity to binning strategy.

```{code-cell} ipython3
parity_output = Parity().evaluate(metadata)
display(parity_output.factors)
```

You can now augment your datasets with additional metadata information, either from additional sources or using
`dataeval` statistical functions for insights into your data.
