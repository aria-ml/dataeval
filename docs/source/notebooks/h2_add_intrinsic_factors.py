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
# # How to add intrinsic factors to Metadata
#
# Bias analysis needs factors to correlate with the labels, and a dataset often arrives with few or none.
# {func}`.image_stats` measures intrinsic factors from the pixels, such as brightness, sharpness and contrast, and
# {meth}`.Metadata.add_factors` attaches them to the {class}`.Metadata` so {class}`.Balance` and {class}`.Diversity` can
# analyze them. See
# [Dataset Bias and Coverage](../concepts/DatasetBias.md#measuring-bias-mutual-information-and-evenness).
#
# Use it when the dataset carries little or no metadata, or when you want to test a property it does not record.
#
# You need `dataeval`, `dataeval-plots` and `maite-datasets`. This guide uses CIFAR-10.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval dataeval-plots maite-datasets
except Exception:
    pass

# %%
import dataeval_plots as dep
import polars as pl
from maite_datasets.image_classification import CIFAR10

from dataeval import Metadata
from dataeval.bias import Balance, Diversity
from dataeval.core import image_stats
from dataeval.data import Limit, Shuffle, View
from dataeval.flags import ImageStats

_ = pl.Config.set_tbl_rows(-1)
dep.set_default_backend("matplotlib")

# %% [markdown]
# ## Load the dataset
#
# Load a shuffled sample of 20,000 CIFAR-10 images and wrap it in a `Metadata`. Its only factor is `batch_num`, the
# on-disk file each image came from.

# %%
# Load in the CIFAR10 dataset and limit to 20,000 images with random shuffling
cifar10 = View(CIFAR10("data", image_set="base", download=True), [Limit(20000), Shuffle(seed=0)])
print(cifar10)

# %%
metadata = Metadata(cifar10)
print(f"Factor names: {metadata.factor_names}")

# %% [markdown]
# ## Add image statistics as factors
#
# Run `image_stats` for the `PIXEL` and `VISUAL` statistics and hand the whole result to `add_factors`. Passing the
# result rather than its `stats` mapping also passes its source index, which places each value on the row it describes.

# %%
# Calculate pixel and visual statistics
calc_results = image_stats(cifar10, stats=ImageStats.PIXEL | ImageStats.VISUAL)

# Append the factors to the metadata
metadata.add_factors(calc_results)

# %% [markdown]
# `batch_num` names the file an image was stored in, not anything about the image, so exclude it. Bin each continuous
# statistic into five uniform-width bins.

# %%
# Exclude the batch_num as it is not a relevant factor for bias analysis
metadata.exclude = ["batch_num"]

# Bin each continuous statistic into 5 uniform-width bins
keys = ("mean", "std", "var", "skew", "kurtosis", "entropy", "brightness", "darkness", "sharpness", "contrast", "zeros")
metadata.continuous_factor_bins = dict.fromkeys(keys, 5)

# %% [markdown]
# ## Check the balance
#
# {class}`.Balance` reports the mutual information between every pair of factors and between each factor and the class.

# %%
balance_output = Balance().evaluate(metadata)

# %%
dep.plot(balance_output)

# %% [markdown]
# Variance and standard deviation share nearly all their information, as expected, and mean intensity tracks brightness,
# darkness and contrast. None of the statistics correlates strongly with the class label.

# %%
dep.plot(balance_output, plot_classwise=True)

# %% [markdown]
# Classwise balance agrees. No single statistic separates one class from the rest.

# %% [markdown]
# ## Check the diversity
#
# {class}`.Diversity` measures how evenly each factor's bins are sampled.

# %%
diversity_output = Diversity().evaluate(metadata)
dep.plot(diversity_output)

# %% [markdown]
# Classes are sampled uniformly. Kurtosis has the lowest diversity of the statistical moments, which may mean five
# uniform bins fit it poorly. Try a different binning to check.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
pairs = balance_output.factors
assert pairs.filter((pl.col("factor1") == "std") & (pl.col("factor2") == "var"))["mi_value"][0] > 0.9
for other in ("brightness", "darkness", "contrast"):
    assert pairs.filter((pl.col("factor1") == "mean") & (pl.col("factor2") == other))["is_correlated"][0]
assert (balance_output.balance["mi_value"][1:] < 0.1).all()
assert not balance_output.classwise["is_imbalanced"].any()

spread = diversity_output.factors
assert spread.filter(pl.col("factor_name") == "class_label")["diversity_value"][0] > 0.99
moments = spread.filter(pl.col("factor_name").is_in(["mean", "std", "var", "skew", "kurtosis"]))
assert moments.sort("diversity_value")["factor_name"][0] == "kurtosis"

# %% [markdown]
# ## Next steps
#
# - [Dataset Bias and Coverage](../concepts/DatasetBias.md) — Understand how class imbalance and missing data impact
#   model reliability.
# - [Identify bias and correlations](./tt_identify_bias.py) — Analyze class and metadata correlations to spot potential
#   dataset bias.
# - [How to control and reuse a factor's binning](./h2_control_factor_binning.py) — Declare the cuts you mean and apply
#   them unchanged to the next dataset.
