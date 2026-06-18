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
# # How to wrap a DataFrame-backed image classification dataset

# %% [markdown]
# ## Problem statement
#
# Many datasets are catalogued as tabular data: a CSV, parquet file, or database
# query that lists one row per image with columns for the image location, its
# label, and any associated metadata (weather, sensor, altitude, capture time,
# etc.). A natural way to hold this in memory is a
# [pandas](https://pandas.pydata.org/) `DataFrame`.
#
# DataEval does not require any particular dataset class. Its evaluators consume
# any object that satisfies the {class}`.AnnotatedDataset` protocol - a minimal
# interface of `__len__`, `__getitem__`, and a `metadata` property. This means you
# can put a thin adapter around your DataFrame and use it directly with DataEval.
#
# This guide shows you how to wrap a DataFrame in an image classification dataset
# that DataEval can analyze.

# %% [markdown]
# ### When to use
#
# Wrap a DataFrame when your images and their labels/metadata are described by a
# tabular catalog and you want to run DataEval analyses without first exporting to
# a directory layout or a built-in dataset format.

# %% [markdown]
# ### What you will need
#
# 1. A tabular catalog of your data (here, a pandas `DataFrame`)
# 1. A Python environment with the following packages installed:
#    - `dataeval`
#    - `pandas`
#    - `pillow`

# %% [markdown]
# ## Getting started
#
# First import the required libraries needed to set up the example.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval pandas pillow
except Exception:
    pass

# %%
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from dataeval import Metadata
from dataeval.protocols import AnnotatedDataset, DatasetMetadata, DatumMetadata

# %% [markdown]
# ## Build an example catalog
#
# In a real project your catalog would already exist and point at images on disk.
# To keep this guide self-contained, you will write a small handful of images to a
# temporary directory and describe them in a `DataFrame`.
#
# The DataFrame has one row per image and these columns:
#
# - `filepath` - where the image lives on disk
# - `label` - the integer class index
# - `weather`, `altitude_m` - per-image metadata factors you want to analyze

# %%
data_dir = Path(tempfile.mkdtemp())
rng = np.random.default_rng(0)

index2label = {0: "cat", 1: "dog", 2: "bird"}
weather_options = ["clear", "rainy", "foggy"]

rows = []
# Creating 90 128x128 images
for i in range(90):
    # Stand-in for a real image file - replace with your own images on disk
    pixels = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    filepath = data_dir / f"img_{i:03d}.png"
    Image.fromarray(pixels).save(filepath)

    rows.append({
        "filepath": str(filepath),
        "label": i % 3,
        "weather": weather_options[(i // 3) % 3],
        "altitude_m": float(50 + i),
    })

catalog = pd.DataFrame(rows)
catalog.head()


# %% [markdown]
# ## Write the adapter
#
# DataEval's evaluators only need three things from a dataset, which together make
# up the {class}`.AnnotatedDataset` protocol:
#
# - `__len__()` - the number of items
# - `__getitem__(index)` - a `(image, target, datum_metadata)` tuple for one item
#    - `image` - an array of shape `(C, H, W)`
#    - `target` - the label as a one-hot encoded array
#    - `datum_metadata` - a `dict` of per-item metadata, which must contain an `id`
# - a `metadata` property - a {class}`.DatasetMetadata` describing the dataset as a
#   whole, including the `index2label` class-name mapping
#
# The adapter below reads each row of the DataFrame on demand, decodes the image
# referenced by `filepath`, and assembles those three pieces.


# %%
class DataFrameICDataset:
    """An image classification dataset backed by a pandas DataFrame.

    Each row describes one image via an image-path column, a label column, and any
    number of metadata columns that are surfaced to DataEval as datum metadata.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        index2label: dict[int, str],
        image_col: str = "filepath",
        label_col: str = "label",
        metadata_cols: list[str] | None = None,
        dataset_id: str = "dataframe-catalog",
    ) -> None:
        # reset_index keeps positional indexing aligned with __getitem__
        self._df = dataframe.reset_index(drop=True)
        self.index2label = index2label
        self._image_col = image_col
        self._label_col = label_col
        self._metadata_cols = metadata_cols or []
        # DatasetMetadata advertises the class-name mapping to DataEval
        self.metadata: DatasetMetadata = DatasetMetadata(id=dataset_id, index2label=index2label)

    def __len__(self) -> int:
        return len(self._df)

    def _one_hot(self, label: int) -> np.ndarray:
        target = np.zeros(len(self.index2label), dtype=np.float32)
        target[label] = 1.0
        return target

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, DatumMetadata]:
        row = self._df.iloc[index]

        # Decode the image and convert to channels-first (C, H, W)
        image = np.asarray(Image.open(row[self._image_col]).convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)

        # One-hot encode the label as the classification target
        target = self._one_hot(int(row[self._label_col]))

        # Pull in the desired metadata columns; remember metadata is per-image and must include an "id" field
        datum_metadata = DatumMetadata(id=index, **{col: row[col] for col in self._metadata_cols})

        return image, target, datum_metadata


# %% [markdown]
# Instantiate the adapter over your catalog, pointing it at the metadata columns you
# want DataEval to see.

# %%
dataset = DataFrameICDataset(catalog, index2label, metadata_cols=["weather", "altitude_m"])
print(f"Dataset length: {len(dataset)}")

# %% [markdown]
# Inspect a single item to confirm the shapes and types match what DataEval
# expects. Verify that images have channels first, that targets are one-hot
# encoded, and that datum metadata has an id and one value per category.


# %%
image, target, datum_metadata = dataset[0]
print(f"image shape:    {image.shape} ({image.dtype})")
print(f"target (label): {target}")
print(f"datum metadata: {datum_metadata}")

# %% [markdown]
# Because the protocols are runtime-checkable, you can verify structurally that
# the adapter is a valid DataEval dataset.

# %%
print(f"Is an AnnotatedDataset: {isinstance(dataset, AnnotatedDataset)}")

# %% [markdown]
# ## Analyze it with DataEval
#
# The adapter now works anywhere a DataEval dataset is expected. Build a
# {class}`.Metadata` object from it - DataEval reads the labels and the per-item
# metadata you exposed.

# %%
metadata = Metadata(dataset)

# "id" is a per-item identifier, not a meaningful factor for bias analysis
metadata.exclude = ["id"]
print(f"Factor names: {metadata.factor_names}")

# %% [markdown]
# That is all it takes: a small adapter turns any DataFrame-described dataset into
# something DataEval can analyze, with no need to restructure your files on disk.

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
# - [How to wrap a DataFrame-backed object detection dataset](./h2_wrap_dataframe_od_dataset.py)
# - [How to build a MetadataLike object from a DataFrame](./h2_metadata_from_dataframe.py)
# - [How to delay image loading until needed](./h2_lazy_load_images.py)
# - [How to add intrinsic factors to Metadata](./h2_add_intrinsic_factors.py)
#
# ### Tutorials
#
# - [Identify bias and correlations](./tt_identify_bias.py)
