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
# # How to measure label independence

# %% [markdown]
# ## Problem statement
#
# For machine learning tasks, a discrepancy in label frequencies between train and test datasets can result in poor model
# performance.
#
# To help with this, DataEval has a [label parity](../concepts/DistributionShift.md#label-parity) tool that compares the
# label distributions of two datasets.

# %% [markdown]
# ### When to use
#
# DataEval provides a {func}`.label_parity` function to use when you would like to determine if two datasets have
# statistically independent labels. It is a symmetric two-sample test of homogeneity: the null hypothesis is that both
# datasets draw their labels from one class distribution.

# %% [markdown]
# ### What you will need
#
# 1. A Python environment with the following packages installed:
#    - dataeval
# 1. A labeled training image dataset
# 1. A labeled test image dataset to evaluate the label distribution of

# %% [markdown]
# ## Getting started
#
# Let's import the required libraries needed to set up a minimal working example

# %% tags=["remove_cell"]
# Google Colab Only
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval maite-datasets
except Exception:
    pass

# %%
from maite_datasets.image_classification import MNIST

from dataeval import Metadata
from dataeval.core import label_parity

# %% [markdown]
# ## Load the data
#
# While you can use your own dataset, for this example we imported the `MNIST` dataset and will use it going forward. It
# was imported from the DataEval utils package.

# %%
train_ds = MNIST("./data", image_set="train", download=True)
test_ds = MNIST("./data", image_set="test", download=True)

train_md = Metadata(train_ds)
test_md = Metadata(test_ds)

# Get the labels from the collated dataset targets
train_labels = train_md.class_labels
test_labels = test_md.class_labels

# %% [markdown]
# ## Evaluate label statistical independence
#
# Now, let's look at how to use DataEval's label statistics analyzer. Using the {func}`.label_parity` function, compute
# the chi-squared value of the hypothesis that test_ds has the same class distribution as train_ds by specifying the
# labels of the two datasets to be compared. It also returns the p-value of the test.
#
# The test needs to know which labels are independent of each other. In `MNIST` every image carries exactly one label,
# so each label is its own independent item, and you declare that with `assume_independent=True`. For object detection
# or tracking datasets, where one image or sequence contributes many labels, you would instead pass the item each label
# came from as `expected_groups` and `observed_groups`, for example `Metadata.item_indices`. See
# [label parity](../concepts/DistributionShift.md#label-parity) for details.

# %%
results = label_parity(train_labels, test_labels, assume_independent=True)
print(
    f"The chi-squared value for the two label distributions is {results['chi_squared']}, "
    f"with p-value {results['p_value']}"
)

# %% [markdown]
# The chi-squared value is about **2.93** and the p-value is about **0.97**. A p-value this close to 1.0 means the test
# observed no difference between the `MNIST` train and test class distributions. That is the absence of an observed
# shift, not proof that the distributions match: it says the samples did not show a shift at the power they had. The
# result also carries a `score`, the effect size, which is 0.0 here because the bias-corrected measure finds no
# association between dataset and class. `insufficient` is False because the table is not too thin to trust, and `test`
# reads `pearson` because no groups were given.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert 0 <= results["chi_squared"] < 4
assert 0.9 < results["p_value"] <= 1.0

# %% [markdown]
# ## Next steps
#
# - [Distribution Shift](../concepts/DistributionShift.md) — Learn about types of shift and techniques to quantify distribution changes.
# - [Dataset Bias and Coverage](../concepts/DatasetBias.md) — Understand how class imbalance and missing data impact model reliability.
# - [Monitor shifts in operational data](./tt_monitor_shift.py) — Track dataset shift and covariate drift in operational data over time.
# - [Assess an unlabeled data space](./tt_assess_data_space.py) — Evaluate unlabeled image data structure and clustering quality.
# - [Identify bias and correlations](./tt_identify_bias.py) — Analyze class and metadata correlations to spot potential dataset bias.
# - [How to measure train and test dataset divergence](./h2_measure_divergence.py) — Calculate divergence metrics between baseline and operational data distributions.
# - [How to detect undersampled data subsets](./h2_detect_undersampling.py) — Find rare or underrepresented clusters in your dataset.
# - [How to run clustering analysis](./h2_cluster_analysis.py) — Cluster feature embeddings to inspect data grouping.
