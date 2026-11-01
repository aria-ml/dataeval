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
# # Measure label independence
#
# A discrepancy in label frequencies between training and test datasets can hurt model performance. The
# [label parity](../concepts/DistributionShift.md#label-parity) test compares the label distributions of two datasets.
#
# Use {func}`.label_parity` when you want to know whether two datasets have statistically independent labels. It is a
# symmetric two-sample test of homogeneity: the null hypothesis is that both datasets draw their labels from one class
# distribution.
#
# You need `dataeval` and a labeled training dataset and test dataset. This guide uses MilitaryVehicles.

# %% tags=["remove_cell"]
# Google Colab Only
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval maite-datasets
except Exception:
    pass

# %%
import numpy as np
from maite_datasets.image_classification import MilitaryVehicles

from dataeval import Metadata
from dataeval.core import label_parity

# %% [markdown]
# ## Load the data
#
# While you can use your own dataset, this example uses MilitaryVehicles, which holds 24 types of ground vehicle in its
# `train` and `test` splits. Only the labels are needed, so `lazy=True` skips loading the images.

# %%
train_ds = MilitaryVehicles("./data", image_set="train", download=True, lazy=True)
test_ds = MilitaryVehicles("./data", image_set="test", download=True, lazy=True)

# Get the labels from the collated dataset targets
train_labels = Metadata(train_ds).class_labels
test_labels = Metadata(test_ds).class_labels

# %% [markdown]
# ## Evaluate label statistical independence
#
# Use the {func}`.label_parity` function to compute the chi-squared value of the hypothesis that `test_ds` has the same
# class distribution as `train_ds`. Specify the labels of the two datasets to compare. The function also returns the
# p-value of the test.
#
# The test needs to know which labels are independent of each other. In MilitaryVehicles every image carries a single
# label, so each label is its own independent item, and you declare that with `assume_independent=True`. For object
# detection or tracking datasets, where one image or sequence contributes many labels, you would instead pass the item
# each label came from as `expected_groups` and `observed_groups`, for example `Metadata.item_indices`. See
# [label parity](../concepts/DistributionShift.md#label-parity) for details.

# %%
results = label_parity(train_labels, test_labels, assume_independent=True)
print(
    f"The chi-squared value for the two label distributions is {results['chi_squared']:.1f}, "
    f"with p-value {results['p_value']:.1e} and score {results['score']:.2f}"
)

# %% [markdown]
# The p-value is far below 0.05, so the test finds that the training and test class distributions differ. Read the
# `score` before the p-value. It is the bias-corrected Cramér's V between dataset and class, an effect size that stays
# near 0 when the two datasets share a distribution. The p-value assumes the images are independent, and
# MilitaryVehicles holds families of repeat photographs of the same vehicle, so the p-value overstates how surprising
# the gap is. The `insufficient` key is False because the table is not too thin to trust, and `test` reads `pearson`
# because no groups were given.

# %% [markdown]
# ## Check the score against a baseline
#
# A score of 0.19 only means something against what two samples of one distribution give. Split the training set at
# random into two halves, test them against each other, and repeat with 20 seeds.

# %%
controls = []
for seed in range(20):
    order = np.random.default_rng(seed).permutation(len(train_labels))
    half = len(order) // 2
    controls.append(label_parity(train_labels[order[:half]], train_labels[order[half:]], assume_independent=True))

print(f"Largest score over 20 random halves of the training set: {max(c['score'] for c in controls):.3f}")
print(f"Random halves with p < 0.05: {sum(c['p_value'] < 0.05 for c in controls)} of 20")

# %% [markdown]
# Random halves of the training set stay at a score below 0.04 and none of them is flagged, so the score of 0.19 between
# the training and test sets is about five times anything chance produces here. The two splits have different class
# mixes: `BM-30` is 3.3% of the training images and 8.6% of the test images, and `BMP-1` is 5.0% against 1.7%. The test
# does not say which split is wrong. Decide whether the test set should reflect the mix you expect in operation, and if
# it should, rebalance one of the splits.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
names = {name: index for index, name in train_ds.metadata["index2label"].items()}


def share(labels: np.ndarray, name: str) -> float:
    """Share of the labels that belong to the named class."""
    return float((labels == names[name]).mean())


assert len(train_labels) == 7823
assert len(test_labels) == 1621
assert results["p_value"] < 1e-50
assert not results["insufficient"]
assert results["test"] == "pearson"
assert abs(results["chi_squared"] - 364.5) < 0.1  # the text says the distributions differ
assert abs(results["score"] - 0.19) < 0.01  # and quotes 0.19
assert max(c["score"] for c in controls) < 0.04  # "below 0.04"
assert sum(c["p_value"] < 0.05 for c in controls) == 0  # "none of them is flagged"
assert results["score"] > 4.5 * max(c["score"] for c in controls)  # "about five times"
assert round(share(train_labels, "BM-30") * 100, 1) == 3.3
assert round(share(test_labels, "BM-30") * 100, 1) == 8.6
assert round(share(train_labels, "BMP-1") * 100, 1) == 5.0
assert round(share(test_labels, "BMP-1") * 100, 1) == 1.7

# %% [markdown]
# ## Next steps
#
# - [Distribution shift](../concepts/DistributionShift.md) — Learn about types of shift and techniques to quantify
#   distribution changes.
# - [Identify bias and correlations](./tt_identify_bias.py) — Analyze class and metadata correlations to spot potential
#   dataset bias.
# - [Measure train and test dataset divergence](./h2_measure_divergence.py) — Calculate divergence metrics
#   between baseline and operational data distributions.
