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

# How to identify duplicates

+++

## Problem Statement

One of the first steps in Exploratory Data Analysis (EDA) is to check for duplicates. Duplicates add no new information
and can distort model training by over-emphasizing features that in appear in the duplicates.

DataEval provides a Duplicates class to assist you in removing duplicates so you can start training your models on high
quality data.

+++

### _When to use_

The Duplicates class should be used if you need to find duplicate images in your dataset.

+++

### _What you will need_

1. A python envornment with following packages installed:
   - dataeval
   - maite-datasets
1. A dataset to analyze

+++

## _Getting Started_

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
from dataclasses import asdict

import numpy as np
from maite_datasets.image_classification import MNIST

from dataeval import Metadata
from dataeval.quality import Duplicates
from dataeval.selection import Indices, Select
```

## Loading in the data

Load the MNIST data and create the dataset.

The MNIST dataset contains 70,000 images - 60,000 in the train set and 10,000 in the test set. For the purposes of this
demonstration, we are just going to use the test set.

```{code-cell} ipython3
# Load in the mnist dataset
testing_dataset = MNIST(root="./data/", image_set="test", download=True)

# Get the labels
labels = Metadata(testing_dataset).class_labels
```

Because the MNIST dataset does not contain any exact duplicates we are going to adjust the dataset to include some.

```{code-cell} ipython3
# Creating some indices to duplicate
print("Exact duplicates")
duplicates = {}
for i in [1, 2, 5, 9]:
    matching_indices = np.where(labels == i)[0]
    print(f"\t{i} - ({matching_indices[23]}, {matching_indices[78]})")
    duplicates[int(matching_indices[78])] = int(matching_indices[23])
```

```{code-cell} ipython3
# Create a subset with the identified duplicate indices swapped
indices_with_duplicates = [duplicates.get(i, i) for i in range(len(testing_dataset))]
duplicates_ds = Select(testing_dataset, Indices(indices_with_duplicates))
```

## Finding the Duplicates

Now we are asking our Duplicates class to find the needle in the haystack. There are only 4 exact duplicates.

```{code-cell} ipython3
# Initialize the Duplicates class to begin to identify duplicate images.
identifyDuplicates = Duplicates()

# Evaluate the data
results = identifyDuplicates.evaluate(duplicates_ds)
```

The results can be returned as a dictionary with exact and near as the keys. So we will extract those to view the
results.

```{code-cell} ipython3
for category, dupe_types in results.data().items():
    for dupe_type, groups in asdict(dupe_types).items():
        if groups is not None:
            print(f"{dupe_type} duplicate {category} : {len(groups)}")
            for group in groups:
                print(f"\t{group}")
```

The `Duplicates` class was able to find all 4 exact duplicates out of the 10,000 samples.

It also found several sets of images that are very closely related to each other, and since we are using hand written
digits we would expect it to find some images that were nearly identical.

```{code-cell} ipython3
---
tags: [remove_cell]
---
### TEST ASSERTION CELL ###
assert results.items.exact is not None
assert len(results.items.exact) == len(duplicates)
for k, v in duplicates.items():
    assert [v, k] in results.items.exact
```
