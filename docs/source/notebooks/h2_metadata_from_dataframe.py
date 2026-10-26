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
# # How to build a MetadataLike object from a DataFrame
#
# DataEval's bias evaluators, {class}`.Balance`, {class}`.Diversity` and {class}`.Parity`, need your factors and labels but
# not your images. They accept any object that satisfies the {class}`.MetadataLike` protocol, so a pandas `DataFrame` can
# feed them directly.
#
# Use this when your labels and metadata factors already live in a table and you want to run bias analysis on them without
# decoding images or building a full {class}`.AnnotatedDataset`.
#
# You need `dataeval` and `pandas`.

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
# `MetadataLike` is four properties:
#
# - `factor_names`: the names of the metadata factors
# - `factor_data`: a `(n_samples, n_factors)` integer array of discretized
#   factor values (continuous factors must be pre-binned to integers)
# - `class_labels`: one label per sample
# - `is_binned`: a flag per factor, same length as `factor_names`. It is `True` where
#   you produced that factor's integers by cutting a range into bins, and `False`
#   where each integer stands for a value of its own (a category, a count)
#
# ```{note}
# `is_binned` replaced `is_discrete` in v1.1; containers providing only
# `is_discrete` stopped being recognized as metadata in v1.2. See
# {ref}`what to change and why <metadatalike-is-binned>`.
# ```
#
# and two optional properties: `index2label` (class-name mapping) and
# `item_indices` (which source image each label came from).
#
# The adapter discretizes the factor columns, the one preprocessing step the protocol
# requires:
#
# - Discrete factors (categorical or already-integer) are integer-encoded with
#   `pandas.factorize`.
# - Continuous factors are binned into integer codes with `pandas.cut`. You
#   choose the number of bins per factor, which controls how finely the range is
#   grouped.
#
# Only the columns you pass become factors. Any other column in the DataFrame is
# ignored. `item_indices` maps each row back to a source image. For a classification
# table (one row per image), the mapping is 1:1. For an object detection table (one
# row per box), several rows share an image.


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
        self._is_binned: list[bool] = []

        # Discrete factors: map each distinct value to an integer code
        for name in discrete_factors:
            columns.append(pd.factorize(dataframe[name])[0].astype(np.int64))
            self._factor_names.append(name)
            self._is_binned.append(False)  # codes stand for the values themselves

        # Continuous factors: bin into the requested number of integer bins
        for name, n_bins in continuous_factors.items():
            binned = pd.cut(dataframe[name], bins=n_bins, labels=False)
            columns.append(np.asarray(binned, dtype=np.int64))
            self._factor_names.append(name)
            self._is_binned.append(True)  # pd.cut produced bin indices

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
    def is_binned(self) -> list[bool]:
        return self._is_binned

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
# ## Apply the adapter to an image classification DataFrame
#
# A classification catalog has one row per image: the label plus any metadata
# factors. Here `weather` is categorical (discrete) and `altitude_m` is continuous.

# %% tags=["hide-input"]
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
print(f"is_binned:          {ic_meta.is_binned}")
print(f"factor_data shape:  {ic_meta.factor_data.shape}")
print(f"class_labels shape: {ic_meta.class_labels.shape}")

# %% [markdown]
# Run a bias evaluator on it. {class}`.Balance` measures the normalized mutual
# information between each factor and the class labels.

# %%
ic_balance = Balance().evaluate(ic_meta)
print(ic_balance.balance)

# %% [markdown]
# ## Apply the adapter to an object detection DataFrame
#
# An object detection catalog in long format has one row per box. Each row
# carries its image's factors, so `factor_data` and `class_labels` are naturally
# per-detection. The extra step is `item_indices`, which records the image each
# box belongs to.

# %% tags=["hide-input"]
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
# The same evaluator works unchanged. It operates on the per-detection factors
# and labels.

# %%
od_balance = Balance().evaluate(od_meta)
print(od_balance.balance)

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
# Verify that both adapters satisfy the protocol and discretize the factors
for meta, catalog in ((ic_meta, ic_catalog), (od_meta, od_catalog)):
    assert isinstance(meta, MetadataLike)
    assert meta.factor_names == ["weather", "altitude_m"]
    assert meta.is_binned == [False, True]
    assert meta.factor_data.shape == (len(catalog), 2)
    assert meta.factor_data.dtype.kind == "i"
    assert meta.class_labels.shape == (len(catalog),)
    assert set(meta.factor_data[:, 0]) == {0, 1, 2}  # three weather values
    assert meta.factor_data[:, 1].min() >= 0  # four altitude bins
    assert meta.factor_data[:, 1].max() <= 3

# Verify the item indices: one image for each row in classification, several rows for an image in detection
assert np.array_equal(ic_meta.item_indices, np.arange(len(ic_catalog)))
assert len(np.unique(od_meta.item_indices)) == 40
assert len(od_meta.class_labels) == len(od_catalog) > 40

# Verify that Balance scored the class and both factors on both tables
for balance in (ic_balance, od_balance):
    assert set(balance.balance["factor_name"].to_list()) == {"class_label", "weather", "altitude_m"}

# %% [markdown]
# ## Next steps
#
# - [Dataset Bias and Coverage](../concepts/DatasetBias.md) — Understand how class imbalance and missing data impact model reliability.
# - [Identify bias and correlations](./tt_identify_bias.py) — Analyze class and metadata correlations to spot potential dataset bias.
# - [How to wrap a DataFrame as a dataset](./h2_wrap_dataframe_dataset.py) — Wrap a pandas DataFrame catalog as a classification or detection dataset, with optional lazy image loading.
