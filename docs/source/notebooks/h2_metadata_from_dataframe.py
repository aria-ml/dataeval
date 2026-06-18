# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dataeval
#     language: python
#     name: python3
# ---

# %% [markdown]
# # How to build a MetadataLike object from a DataFrame

# %% [markdown]
# ## Problem statement
#
# DataEval's bias evaluators ({class}`.Balance`, {class}`.Diversity`,
# {class}`.Parity`) do not need your images - they only need your
# **factors and labels**. They accept any object that satisfies the
# {class}`.MetadataLike` protocol, which is just four properties:
#
# - `factor_names` - the names of the metadata factors
# - `factor_data` - a `(n_samples, n_factors)` integer array of *discretized*
#   factor values (continuous factors must be pre-binned to integers)
# - `class_labels` - one label per sample
# - `is_discrete` - a flag per factor (discrete vs continuous), same length as
#   `factor_names`
#
# and two optional properties: `index2label` (class-name mapping) and
# `item_indices` (which source image each label came from).
#
# When your factors are already tabular, you can implement `MetadataLike` directly
# from a `DataFrame` and run bias analysis **without loading a single image**. This
# is lighter than wrapping a full dataset (see the related how-tos) and is all you
# need when you only care about labels and metadata.
#
# You will build one reusable adapter and apply it to an image classification
# catalog and an object detection catalog.

# %% [markdown]
# ### When to use
#
# Use this when your labels and metadata factors live in a table and you want to
# run bias/diversity/parity analysis on them - without decoding images or building
# a full {class}`.AnnotatedDataset`.

# %% [markdown]
# ### What you will need
#
# 1. A table of factors and labels (here, pandas `DataFrame`s)
# 1. A Python environment with the following packages installed:
#    - `dataeval`
#    - `pandas`

# %% [markdown]
# ## Getting started
#
# First import the required libraries needed to set up the example.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval pandas
except Exception:
    pass

# %%
import numpy as np
import pandas as pd

from dataeval.bias import Balance
from dataeval.protocols import MetadataLike


# %% [markdown]
# ## Write the adapter
#
# The adapter discretizes the factor columns - the one preprocessing step
# `MetadataLike` requires - and exposes the four properties of the protocol plus
# the two optional ones:
#
# - **Discrete factors** (categorical or already-integer) are integer-encoded with
#   `pandas.factorize`.
# - **Continuous factors** are binned into integer codes with `pandas.cut`; you
#   choose the number of bins per factor, which controls how finely the range is
#   grouped.
#
# Only the columns you pass become factors - any other column in the DataFrame is
# ignored. `item_indices` maps each row back to a source image: for a classification
# table (one row per image) the mapping is 1:1; for an object detection table (one
# row per box) several rows share an image.


# %%
class DataFrameMetadata:
    """A lightweight object implementing the MetadataLike protocol.

    Builds the arrays the bias evaluators need directly from DataFrame columns:
    discrete factors are integer-encoded and continuous factors are binned, so
    ``factor_data`` is fully discretized.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        label_col: str,
        discrete_factors: list[str] | None = None,
        continuous_factors: dict[str, int] | None = None,
        index2label: dict[int, str] | None = None,
        item_index_col: str | None = None,
    ) -> None:
        discrete_factors = discrete_factors or []
        continuous_factors = continuous_factors or {}

        columns: list[np.ndarray] = []
        self._factor_names: list[str] = []
        self._is_discrete: list[bool] = []

        # Discrete factors: map each distinct value to an integer code
        for name in discrete_factors:
            columns.append(pd.factorize(dataframe[name])[0].astype(np.int64))
            self._factor_names.append(name)
            self._is_discrete.append(True)

        # Continuous factors: bin into the requested number of integer bins
        for name, n_bins in continuous_factors.items():
            binned = pd.cut(dataframe[name], bins=n_bins, labels=False)
            columns.append(np.asarray(binned, dtype=np.int64))
            self._factor_names.append(name)
            self._is_discrete.append(False)

        self._factor_data = np.stack(columns, axis=1) if columns else np.empty((len(dataframe), 0), dtype=np.int64)
        self._class_labels = dataframe[label_col].to_numpy(dtype=np.intp)
        self._index2label = index2label or {}
        # Each label's source image. Default (no column) is the 1:1 case where
        # every row is its own image - correct for classification catalogs.
        self._item_indices = (
            dataframe[item_index_col].to_numpy(dtype=np.intp)
            if item_index_col is not None
            else np.arange(len(dataframe), dtype=np.intp)
        )

    @property
    def factor_names(self) -> list[str]:
        return self._factor_names

    @property
    def factor_data(self) -> np.ndarray:
        return self._factor_data

    @property
    def is_discrete(self) -> list[bool]:
        return self._is_discrete

    @property
    def class_labels(self) -> np.ndarray:
        return self._class_labels

    @property
    def index2label(self) -> dict[int, str]:
        return self._index2label

    @property
    def item_indices(self) -> np.ndarray:
        return self._item_indices


# %% [markdown]
# ## From an image classification DataFrame
#
# A classification catalog has **one row per image**: the label plus any metadata
# factors. Here `weather` is categorical (discrete) and `altitude_m` is continuous.

# %%
rng = np.random.default_rng(0)
ic_index2label = {0: "cat", 1: "dog", 2: "bird"}
weather_options = ["clear", "rainy", "foggy"]

ic_catalog = pd.DataFrame({
    "label": rng.integers(0, 3, size=90),
    "weather": rng.choice(weather_options, size=90),
    "altitude_m": rng.uniform(50, 150, size=90),
})
ic_catalog.head()

# %% [markdown]
# Build the adapter, binning the continuous `altitude_m` factor into 4 bins. No
# `item_index_col` is needed because each row is already its own image.

# %%
ic_meta = DataFrameMetadata(
    ic_catalog,
    label_col="label",
    discrete_factors=["weather"],
    continuous_factors={"altitude_m": 4},
    index2label=ic_index2label,
)

print(f"Is a MetadataLike:  {isinstance(ic_meta, MetadataLike)}")
print(f"factor_names:       {ic_meta.factor_names}")
print(f"is_discrete:        {ic_meta.is_discrete}")
print(f"factor_data shape:  {ic_meta.factor_data.shape}")
print(f"class_labels shape: {ic_meta.class_labels.shape}")

# %% [markdown]
# Run a bias evaluator on it. {class}`.Balance` measures the normalized mutual
# information between each factor and the class labels.

# %%
ic_balance = Balance().evaluate(ic_meta)
print(ic_balance.balance)

# %% [markdown]
# ## From an object detection DataFrame
#
# An object detection catalog in *long* format has **one row per box**. Each row
# carries its image's factors, so `factor_data` and `class_labels` are naturally
# per-detection. The extra step is `item_indices`, which records the image each
# box belongs to.

# %%
od_index2label = {0: "person", 1: "car", 2: "bicycle"}

rows = []
for image_index in range(40):
    weather = weather_options[image_index % 3]
    altitude = float(50 + image_index)
    # A variable number of boxes per image - hence one row per box
    for _ in range(int(rng.integers(1, 5))):
        rows.append({
            "image_index": image_index,  # which image this box came from
            "label": int(rng.integers(0, 3)),
            "weather": weather,  # image-level factor, repeated per box
            "altitude_m": altitude,  # image-level factor, repeated per box
        })

od_catalog = pd.DataFrame(rows)
od_catalog.head()

# %% [markdown]
# Build the adapter, this time passing `item_index_col` so each detection maps
# back to its source image.

# %%
od_meta = DataFrameMetadata(
    od_catalog,
    label_col="label",
    discrete_factors=["weather"],
    continuous_factors={"altitude_m": 4},
    index2label=od_index2label,
    item_index_col="image_index",
)

print(f"Is a MetadataLike:    {isinstance(od_meta, MetadataLike)}")
print(f"Total detections:     {len(od_meta.class_labels)}")
print(f"Unique source images: {len(np.unique(od_meta.item_indices))}")
print(f"factor_data shape:    {od_meta.factor_data.shape}")

# %% [markdown]
# The same evaluator works unchanged - it operates on the per-detection factors
# and labels.

# %%
od_balance = Balance().evaluate(od_meta)
print(od_balance.balance)

# %% [markdown]
# That is all it takes: one adapter that discretizes factor columns turns either
# kind of tabular catalog into a `MetadataLike` ready for {class}`.Balance`,
# {class}`.Diversity`, and {class}`.Parity` - no images required.

# %% [markdown]
# ## Related concepts
#
# - [Dataset Bias and Coverage](../concepts/DatasetBias.md)
# - [Acting on Results](../concepts/ActingOnResults.md)
#
# ## See also
#
# ### How-to guides
#
# - [How to wrap a DataFrame-backed image classification dataset](./h2_wrap_dataframe_ic_dataset.py)
# - [How to wrap a DataFrame-backed object detection dataset](./h2_wrap_dataframe_od_dataset.py)
#
# ### Tutorials
#
# - [Identify bias and correlations](./tt_identify_bias.py)
