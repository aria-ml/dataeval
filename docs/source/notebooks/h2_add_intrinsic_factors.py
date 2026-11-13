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
# # Add intrinsic factors to Metadata
#
# Bias analysis needs factors to correlate with the labels, and a dataset often arrives with few or none.
# {func}`.image_stats` measures intrinsic factors from the pixels, such as brightness, sharpness and contrast, and
# {meth}`.Metadata.add_factors` attaches them to the {class}`.Metadata` so {class}`.Balance` and {class}`.Diversity` can
# analyze them. See
# [Dataset bias](../concepts/DatasetBias.md#measuring-bias-mutual-information-and-evenness).
#
# Use it when the dataset carries little or no metadata, or when you want to test a property it does not record.
#
# You need `dataeval`, `dataeval-plots` and `maite-datasets`. This guide uses MilitaryVehicles.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval dataeval-plots maite-datasets
except Exception:
    pass

# %%
import dataeval_plots as dep
import numpy as np
import polars as pl
from maite_datasets.image_classification import MilitaryVehicles

from dataeval import Metadata
from dataeval.bias import Balance, Diversity
from dataeval.core import image_stats
from dataeval.flags import ImageStats

dep.set_default_backend("matplotlib")

# %% [markdown]
# ## Load the dataset
#
# Load the MilitaryVehicles train split, 7,823 photographs of 24 vehicle types, and wrap it in a `Metadata`. The dataset
# records nothing about the photographs themselves, so the `Metadata` starts with no factors.

# %%
train_ds = MilitaryVehicles("./data", image_set="train", download=True, lazy=True)
metadata = Metadata(train_ds)
print(f"Factor names: {metadata.factor_names}")

# %% [markdown]
# ## Add image statistics as factors
#
# Run `image_stats` for the `PIXEL` and `VISUAL` statistics and hand the whole result to `add_factors`. Passing the
# result rather than its `stats` mapping also passes its source index, which places each value on the row it describes.
# The `missing` statistic counts missing pixels and is zero for every photograph, so exclude it.

# %%
# Calculate pixel and visual statistics
calc_results = image_stats(train_ds, stats=ImageStats.PIXEL | ImageStats.VISUAL)

# Append the factors to the metadata and drop the constant one
metadata.add_factors(calc_results)
metadata.exclude = ["missing"]
print(f"Factor names: {metadata.factor_names}")

# %% [markdown]
# ## Bin the statistics
#
# `Balance` and `Diversity` read each continuous factor as bins, so the bins decide what they report. Most of these
# statistics are skewed: the kurtosis of a few photographs is far above the rest. Equal-width bins then put nearly every
# image in one bin. Equal-count bins, cut at the quantiles, give each bin the same share of images. Compare the share of
# images in the busiest of five bins.

# %%
keys = (
    "mean",
    "std",
    "var",
    "skew",
    "kurtosis",
    "entropy",
    "saturation",
    "brightness",
    "darkness",
    "sharpness",
    "contrast",
    "zeros",
)
values = {key: np.asarray(calc_results["stats"][key], dtype=float) for key in keys}
# Cut at the quantiles, with infinite outer edges so that the largest value falls in the last bin
quantile_edges = {key: [-np.inf, *np.unique(np.quantile(values[key], [0.2, 0.4, 0.6, 0.8])), np.inf] for key in keys}


def busiest_bin(key: str, edges: list[float] | np.ndarray) -> float:
    """Share of the images that fall in the most crowded bin."""
    return float(np.histogram(values[key], edges)[0].max() / len(values[key]))


for key in ("kurtosis", "zeros"):
    equal_width = busiest_bin(key, np.linspace(values[key].min(), values[key].max(), 6))
    print(
        f"{key:>8}: busiest bin holds {equal_width:.1%} of the images with equal-width bins, "
        f"{busiest_bin(key, quantile_edges[key]):.1%} with equal-count bins"
    )

# %% [markdown]
# Use the equal-count edges. A bin count is equal-width, so pass explicit edges instead. The outer edges are infinite
# because a top edge at the maximum would put the maximum in a bin of its own, and a tenth of these photographs have the
# largest darkness, 255. See [Control and reuse a factor's binning](./h2_control_factor_binning.py).

# %%
metadata.continuous_factor_bins = quantile_edges

# %% [markdown]
# ## Check the balance
#
# {class}`.Balance` reports the mutual information between every pair of factors and between each factor and the class.

# %%
balance_output = Balance().evaluate(metadata)

# %%
dep.plot(balance_output)

# %% [markdown]
# No statistic tells you the vehicle type: the `class_label` row is zero throughout. The statistics do track each other.
# Variance and standard deviation share all their information, as expected, and mean intensity tracks contrast and
# darkness.

# %%
dep.plot(balance_output, plot_classwise=True)

# %% [markdown]
# Classwise balance agrees. No class stands out, and the largest value is 0.015, for the smallest class. The
# photographs' exposure and sharpness are spread evenly across the vehicle types.

# %% [markdown]
# ## Check the diversity
#
# {class}`.Diversity` measures how evenly each factor's bins are sampled.

# %%
diversity_output = Diversity().evaluate(metadata)
dep.plot(diversity_output)

# %% [markdown]
# The classes are not sampled uniformly: they range from 119 to 424 photographs, and the diversity is 0.96. Every
# continuous statistic reads 1.00, and that is the binning and not the photographs. Equal-count bins hold the same share
# of images by construction, so the evenness is fixed. With equal-width bins the same photographs read the opposite way:

# %%
width_metadata = Metadata(train_ds)
width_metadata.add_factors(calc_results)
width_metadata.exclude = ["missing"]
width_metadata.continuous_factor_bins = dict.fromkeys(keys, 5)

width_diversity = Diversity().evaluate(width_metadata).factors
for key in ("kurtosis", "zeros"):
    print(f"{key:>8}: diversity {width_diversity.filter(pl.col('factor_name') == key)['diversity_value'][0]:.3f}")

# %% [markdown]
# Read the diversity of a continuous statistic only against the bins you chose. For the class, it is an honest measure.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert metadata.factor_names == sorted(keys)
assert np.all(np.asarray(calc_results["stats"]["missing"]) == 0)
assert busiest_bin("kurtosis", np.linspace(values["kurtosis"].min(), values["kurtosis"].max(), 6)) > 0.99
assert busiest_bin("kurtosis", quantile_edges["kurtosis"]) < 0.25

pairs = balance_output.factors
assert pairs.filter((pl.col("factor1") == "std") & (pl.col("factor2") == "var"))["mi_value"][0] > 0.99
assert abs(pairs.filter((pl.col("factor1") == "contrast") & (pl.col("factor2") == "mean"))["mi_value"][0] - 0.94) < 0.02
assert abs(pairs.filter((pl.col("factor1") == "darkness") & (pl.col("factor2") == "mean"))["mi_value"][0] - 0.78) < 0.03
assert (balance_output.balance["mi_value"][1:] < 0.01).all()  # "zero throughout"
largest_classwise = float(balance_output.classwise["mi_value"].to_numpy().max())
assert abs(largest_classwise - 0.015) < 0.003  # "the largest value is 0.015"
assert balance_output.classwise.sort("mi_value")["class_name"][-1] == "30N6E"
assert not balance_output.classwise["is_imbalanced"].any()

counts = np.bincount(metadata.class_labels)
assert (counts.min(), counts.max()) == (119, 424)
spread = diversity_output.factors
assert abs(spread.filter(pl.col("factor_name") == "class_label")["diversity_value"][0] - 0.96) < 0.01
for key in keys:
    assert spread.filter(pl.col("factor_name") == key)["diversity_value"][0] > 0.999  # "reads 1.00"
assert 0.10 < np.mean(values["darkness"] == 255) < 0.11  # "a tenth of these photographs"
for key in ("kurtosis", "zeros"):
    assert width_diversity.filter(pl.col("factor_name") == key)["diversity_value"][0] < 0.01

# %% [markdown]
# ## Next steps
#
# - [Dataset bias](../concepts/DatasetBias.md) — Understand how class imbalance and missing data impact
#   model reliability.
# - [Identify bias and correlations](./tt_identify_bias.py) — Analyze class and metadata correlations to spot potential
#   dataset bias.
# - [Control and reuse a factor's binning](./h2_control_factor_binning.py) — Declare the cuts you mean and apply
#   them unchanged to the next dataset.
