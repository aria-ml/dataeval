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

# How to measure train and test dataset divergence

+++

## _Problem Statement_

When evaluating new testing data, or comparing two datasets, we often want to have a quantitative way of comparing and
evaluating shifts in covariates. HP divergence is a nonparametric divergence metric which gives the distance between two
datasets. A divergence of 0 means that the two datasets are approximately identically distributed. A divergence of 1
means the two datasets are completely separable.

+++

### _When to use_

The `Divergence` class should be used when you would like to know how far two datasets are diverged for one another. For
example, if you would like to measure operational drift.

+++

### _What you will need_

1. A Python environment with the following packages installed:
   - dataeval
   - maite-datasets
2. A set of image embeddings for each dataset (usually obtained with an AutoEncoder)

+++

### _Setting up_

Let's import the required libraries needed to set up a minimal working example

```{code-cell} ipython3
:tags: [remove_cell]

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

from dataeval import Embeddings
from dataeval.core import divergence_fnn
from dataeval.extractors import FlattenExtractor
```

## Loading in data

Load the MNIST data and create the training dataset. For the purposes of this example, we will use subsets of the
training (4000) data.

```{code-cell} ipython3
# Load in the training mnist dataset and use the first 4000
train_ds = MNIST(root="./data/", image_set="train", download=True)

# Create extractor
extractor = FlattenExtractor()

# Extract the first 4000 embeddings
embeddings = Embeddings(train_ds, extractor=extractor, batch_size=400)[:4000]
```

```{code-cell} ipython3
print("Number of samples: ", len(embeddings))
print("Image shape:", embeddings[0].shape)
```

## Calculate initial divergence

Let's calculate the divergence using nearest neighbor disagreements between the first 2000 images and the second 2000
images from this sample.

```{code-cell} ipython3
data_a = embeddings[:2000]
data_b = embeddings[2000:]
```

```{code-cell} ipython3
div = divergence_fnn(data_a, data_b)
print(div)
```

We estimate that the divergence between these (identically distributed) images sets is at or close to 0.

+++

## Loading in corrupted data

Now let's load in a corrupted mnist dataset.

```{code-cell} ipython3
corrupted_ds = MNIST(root="./data", image_set="train", corruption="translate", download=True)

# Create extractor
corrupted_extractor = FlattenExtractor()
corrupted_emb = Embeddings(corrupted_ds, extractor=corrupted_extractor, batch_size=64)[:2000]
```

```{code-cell} ipython3
print("Number of corrupted samples: ", len(corrupted_emb))
print("Corrupted image shape:", corrupted_emb[0].shape)
```

## Calculate corrupted divergence

Now lets calculate the Divergence between this corrupted dataset and the original images

```{code-cell} ipython3
div = divergence_fnn(data_a, corrupted_emb)
print(div)
```

```{code-cell} ipython3
:tags: [remove_cell]

### TEST ASSERTION CELL ###
assert div["divergence"] > 0.95
```

We conclude that the translated MNIST images are significantly different from the original images.
