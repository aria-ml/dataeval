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
# # Measure train and test dataset divergence
#
# When you evaluate new test data or compare two datasets, you often want a quantitative measure of the shift in
# covariates. HP [divergence](../concepts/Divergence.md) is a nonparametric measure of the distance between two
# datasets: 0 means they are approximately identically distributed, and 1 means they are completely separable.
#
# Use {func}`.divergence_fnn` or {func}`.divergence_mst` when you want to know how far apart two datasets are, for
# example to measure [operational drift](../concepts/DistributionShift.md#drift-detection).
#
# You need `dataeval`, `maite-datasets`, `torchvision` and a set of image embeddings for each dataset. This guide builds
# them from MilitaryVehicles photographs.

# %% tags=["remove_cell"]
# Google Colab Only
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval maite-datasets torchvision
except Exception:
    pass

# %%
from collections.abc import Callable

import numpy as np
from maite_datasets.image_classification import MilitaryVehicles
from torchvision.models import ResNet18_Weights, resnet18

from dataeval import Embeddings
from dataeval.core import divergence_fnn, divergence_mst
from dataeval.data import Limit, Shuffle, View
from dataeval.extractors import TorchExtractor

# %% [markdown]
# ## Load the data
#
# Load the MilitaryVehicles train and test splits. You use 1,600 training images, as two samples of 800, and 800 test
# images. Both splits are stored class by class, so shuffle before you limit.
#
# The embeddings are the features of a pre-trained ResNet18, one 512-dimensional vector per image, made as in
# [Create embeddings](./h2_create_embeddings.py). Raw pixels would mostly follow brightness and alignment, and
# brightness is the change you test for below.

# %%
SAMPLE = 800

train_ds = MilitaryVehicles("./data", image_set="train", download=True, lazy=True)
test_ds = MilitaryVehicles("./data", image_set="test", download=True, lazy=True)

weights = ResNet18_Weights.DEFAULT
extractor = TorchExtractor(
    resnet18(weights=weights, progress=False), transforms=weights.transforms(), layer_name="avgpool"
)


def embed(dataset: MilitaryVehicles, size: int) -> np.ndarray:
    """Embed a random sample of the dataset. The extractor holds no state, so one instance serves every dataset."""
    return np.asarray(Embeddings(View(dataset, [Shuffle(seed=0), Limit(size)]), extractor=extractor, batch_size=64))


train_embeddings = embed(train_ds, 2 * SAMPLE)
test_embeddings = embed(test_ds, SAMPLE)

# %% [markdown]
# ## Calculate the initial divergence
#
# Calculate the divergence using nearest neighbor disagreements between the first 800 training images and the second
# 800.

# %%
data_a = train_embeddings[:SAMPLE]
data_b = train_embeddings[SAMPLE:]

# %%
div = divergence_fnn(data_a, data_b)
print(div)

# %% [markdown]
# The two halves are samples of the same dataset, so this value is a reference, and it is above 0. A single value is one
# draw, so split the same 1,600 embeddings at random several more times to see how far two samples of one dataset drift
# apart by chance. HP divergence has no significance threshold, so
# [read a divergence against a reference](../concepts/Divergence.md#limitations) like this one.

# %%
reference = []
for seed in range(8):
    order = np.random.default_rng(seed).permutation(len(train_embeddings))
    reference.append(divergence_fnn(train_embeddings[order[:SAMPLE]], train_embeddings[order[SAMPLE:]])["divergence"])

print(f"Two samples of the training set, 8 splits: mean {np.mean(reference):.3f}, largest {max(reference):.3f}")

# %% [markdown]
# Two samples of one dataset reach about 0.05 by chance, so a divergence is a shift only when it sits clearly above
# that.

# %% [markdown]
# ## Compare train and test
#
# Calculate the divergence between the training images and the test images.

# %%
train_test = divergence_fnn(data_a, test_embeddings)["divergence"]
print(f"Train against test: {train_test:.3f}")


# %% [markdown]
# The test images are as close to the training images as two training samples are to each other. The two splits have
# different class mixes, as [Measure label independence](./h2_measure_label_independence.py) shows. Divergence compares
# where the images sit in embedding space and not how many each class contributes, so it does not see that difference.

# %% [markdown]
# ## Load the dimmed data
#
# Operational data is rarely collected in the conditions of the training data. Simulate low light by dimming the test
# photographs. A dataset takes a transform as an argument, so each dimmed copy is the test split with every pixel scaled
# by a factor. The transform must take one argument, the image. Calculate the divergence between the training
# images and each dimmed copy, with both estimators.


# %%
def dim(factor: float) -> Callable[[np.ndarray], np.ndarray]:
    """Return a transform that scales every pixel by the factor."""
    return lambda image: (image * factor).astype(np.uint8)


dimmed = {}
for factor in (0.9, 0.7, 0.5):
    dim_ds = MilitaryVehicles("./data", image_set="test", download=True, lazy=True, transforms=dim(factor))
    dim_embeddings = embed(dim_ds, SAMPLE)
    dimmed[factor] = {
        "fnn": divergence_fnn(data_a, dim_embeddings)["divergence"],
        "mst": divergence_mst(data_a, dim_embeddings)["divergence"],
    }
    print(f"Test images at {factor:.0%} brightness: fnn {dimmed[factor]['fnn']:.3f}, mst {dimmed[factor]['mst']:.3f}")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert train_embeddings.shape == (2 * SAMPLE, 512)
assert test_embeddings.shape == (SAMPLE, 512)
assert max(reference) < 0.08  # "about 0.05"
assert train_test <= max(reference) + 0.02  # "as close as two training samples"
assert abs(dimmed[0.9]["fnn"] - max(reference)) < 0.03  # "close to the largest reference value"
assert abs(dimmed[0.7]["fnn"] - 0.18) < 0.02
assert abs(dimmed[0.5]["fnn"] - 0.38) < 0.02
assert dimmed[0.7]["fnn"] > 3 * max(reference)
assert dimmed[0.9]["fnn"] < dimmed[0.7]["fnn"] < dimmed[0.5]["fnn"] < 0.5  # "far from 1"
assert all(abs(v["fnn"] - v["mst"]) < 0.02 for v in dimmed.values())  # "the estimators agree"
assert (
    0.45 < np.asarray(dim_ds[0][0]).mean() / np.asarray(test_ds[0][0]).mean() < 0.55
)  # the dimming reaches the images

# %% [markdown]
# The divergence grows as the images get darker. At 90% brightness it is about 0.05, close to the largest value that two
# training samples reach, so at this sample size a 10% dimming is not clearly a shift. At 70% it is about 0.18 and at
# 50% about 0.38, well above the reference. Even half brightness is far from 1, so do not read the scale as proportional
# to how much the images changed. The two estimators agree to within 0.02.

# %% [markdown]
# ## Next steps
#
# - [Embeddings](../concepts/Embeddings.md) — Learn how feature embeddings represent dataset samples in low-dimensional
#   space.
# - [Distribution shift](../concepts/DistributionShift.md) — Understand covariate shift and drift detection across
#   datasets.
# - [Monitor shifts in operational data](./tt_monitor_shift.py) — Detect distribution shift between training and
#   operational data.
