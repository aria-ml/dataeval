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
# # How to delay image loading until needed

# %% [markdown]
# ## Problem statement
#
# Decoding an image from disk is expensive compared to reading its label or
# metadata. Many DataEval analyses never touch the pixels at all - the
# {class}`.Metadata` and bias evaluators ({class}`.Balance`, {class}`.Diversity`,
# {class}`.Parity`) only read each item's **target** and **datum metadata**. If
# your dataset eagerly decodes every image inside `__getitem__`, those analyses
# pay the full decoding cost for data they discard.
#
# The fix is to return a lightweight, file-backed proxy from `__getitem__` that
# only decodes the image on first array access. Analyses that ignore the image run
# far faster, while analyses that do need pixels (such as {func}`.compute_stats`)
# transparently trigger the decode when they read the array.
#
# This guide shows you how to build such a proxy and measure the speedup. It
# follows the same pattern used internally by the
# [`maite-datasets`](https://github.com/aria-ml/maite-datasets) package, whose
# downloadable datasets accept a `lazy=True` flag.

# %% [markdown]
# ### When to use
#
# Use lazy image loading when you iterate a dataset for analyses that only need
# labels or metadata - bias, diversity, label distribution, or any custom pass
# over the catalog - and want to avoid decoding images you never inspect.

# %% [markdown]
# ### What you will need
#
# 1. A dataset whose images live on disk and are decoded per item
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
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image

from dataeval import Metadata
from dataeval.bias import Balance
from dataeval.core import compute_stats
from dataeval.flags import ImageStats
from dataeval.protocols import DatasetMetadata, DatumMetadata

# %% [markdown]
# ## Build an example catalog
#
# As in the [DataFrame-backed dataset guide](./h2_wrap_dataframe_ic_dataset.py),
# write some images to disk and describe them in a `DataFrame`. Here we use a few
# hundred moderately sized images so the decoding cost is large enough to measure.

# %%
data_dir = Path(tempfile.mkdtemp())
rng = np.random.default_rng(0)

index2label = {0: "cat", 1: "dog", 2: "bird"}
weather_options = ["clear", "rainy", "foggy"]

rows = []
# Creating 300 256x256 images
for i in range(300):
    pixels = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    filepath = data_dir / f"img_{i:03d}.png"
    Image.fromarray(pixels).save(filepath)
    rows.append({
        "filepath": str(filepath),
        "label": i % 3,
        "weather": weather_options[(i // 3) % 3],
    })

catalog = pd.DataFrame(rows)
catalog.head()


# %% [markdown]
# ## Build a generic lazy array helper
#
# The proxy needs to satisfy the {class}`.Array` protocol so DataEval treats it
# like any other image array, but defer the actual decode. We make it a reusable
# helper by keeping it agnostic about *where* the image comes from: it holds a
# `source` and a `loader` callable, and only runs the loader on first numpy access.
# The `source` can be anything your loader understands - a file path, a URL, raw
# bytes, an archive member, or a database key. The key ideas:
#
# - **`shape`** is resolved from a cheap `shape_loader` (e.g. an image header read)
#   when provided, so analyses that only need dimensions never trigger a decode.
# - **`__array__` / `__getitem__` / `__iter__`** materialize the array on first use
#   and cache it, so repeated access is free.
#
# This mirrors the `LazyArray` implementation in
# [`maite-datasets`](https://github.com/aria-ml/maite-datasets/blob/main/src/maite_datasets/_lazy.py).


# %%
SourceT = TypeVar("SourceT")


class LazyArray(Generic[SourceT]):
    """A source-backed array that runs ``loader(source)`` on first numpy access.

    Parameters
    ----------
    source : SourceT
        Anything the loader understands - a file path, URL, byte buffer, etc.
    loader : Callable[[SourceT], NDArray]
        Turns ``source`` into the materialized array. Runs lazily, at most once.
    shape_loader : Callable[[SourceT], tuple[int, ...]] | None
        Optional cheap way to get the shape without materializing (e.g. reading an
        image header). When omitted, ``shape`` falls back to a full materialize.
    """

    def __init__(
        self,
        source: SourceT,
        loader: Callable[[SourceT], NDArray[Any]],
        shape_loader: Callable[[SourceT], tuple[int, ...]] | None = None,
    ) -> None:
        self._source = source
        self._loader = loader
        self._shape_loader = shape_loader
        self._array: NDArray[Any] | None = None
        self._shape: tuple[int, ...] | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        if self._array is not None:
            return self._array.shape
        if self._shape is None:
            # Use the cheap shape_loader when available; otherwise materialize
            self._shape = self._shape_loader(self._source) if self._shape_loader else self._materialize().shape
        return self._shape

    def _materialize(self) -> NDArray[Any]:
        if self._array is None:
            self._array = self._loader(self._source)
        return self._array

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> NDArray[Any]:
        arr = self._materialize()
        return arr.astype(dtype) if dtype is not None else arr

    def __getitem__(self, key: Any) -> Any:
        return self._materialize()[key]

    def __iter__(self):
        return iter(self._materialize())

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        state = "loaded" if self._array is not None else "lazy"
        return f"LazyArray(source={self._source!r}, shape={self.shape}, {state})"


# %% [markdown]
# To use the helper you supply two small functions for your source type. Here the
# source is a file path on disk, so we provide a loader that decodes the image and
# a shape reader that only reads the header. For a different source - say bytes from
# object storage - you would swap in loaders that understand that source instead.


# %%
def load_chw(path: str) -> NDArray[np.uint8]:
    """Decode an image file to a channels-first (C, H, W) uint8 array."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)


def read_shape(path: str) -> tuple[int, ...]:
    """Read (3, H, W) from the image header without decoding pixels."""
    with Image.open(path) as im:
        width, height = im.size
    return (3, height, width)


# %% [markdown]
# ## Wire it into a dataset
#
# This is the same DataFrame adapter as the previous guide, with one addition: a
# `lazy` flag. When `lazy=True`, `__getitem__` returns a `LazyArray` instead of a
# fully decoded array. Everything else - target and metadata - is unchanged.


# %%
class DataFrameDataset:
    """Image classification dataset backed by a DataFrame, with optional lazy loading."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        index2label: dict[int, str],
        lazy: bool = False,
        dataset_id: str = "dataframe-catalog",
    ) -> None:
        self._df = dataframe.reset_index(drop=True)
        self.index2label = index2label
        self.lazy = lazy
        self.metadata: DatasetMetadata = DatasetMetadata(id=dataset_id, index2label=index2label)

    def __len__(self) -> int:
        return len(self._df)

    def _one_hot(self, label: int) -> np.ndarray:
        target = np.zeros(len(self.index2label), dtype=np.float32)
        target[label] = 1.0
        return target

    def __getitem__(self, index: int) -> tuple[Any, np.ndarray, DatumMetadata]:
        row = self._df.iloc[index]
        filepath = row["filepath"]

        # The only difference: hand back a deferred proxy instead of decoding now
        image = LazyArray(filepath, load_chw, read_shape) if self.lazy else load_chw(filepath)

        target = self._one_hot(int(row["label"]))
        datum_metadata = DatumMetadata(id=index, **{"weather": str(row["weather"])})
        return image, target, datum_metadata


# %% [markdown]
# ## Measure the difference
#
# {class}`.Metadata` iterates the entire dataset but only reads the target and
# datum metadata for each item - it never looks at the image. To see the difference,
# time a metadata bias pass with eager loading versus lazy loading.


# %%
def time_metadata_pass(lazy: bool) -> float:
    dataset = DataFrameDataset(catalog, index2label, lazy=lazy)
    start = time.perf_counter()
    metadata = Metadata(dataset)
    metadata.exclude = ["id"]
    Balance().evaluate(metadata)
    return time.perf_counter() - start


eager_seconds = time_metadata_pass(lazy=False)
lazy_seconds = time_metadata_pass(lazy=True)

print(f"Eager (decodes every image): {eager_seconds * 1000:7.1f} ms")
print(f"Lazy  (decodes none):        {lazy_seconds * 1000:7.1f} ms")
print(f"Speedup:                     {eager_seconds / lazy_seconds:7.1f}x")

# %% [markdown]
# The lazy pass is dramatically faster because no pixels are ever decoded - the
# images are only ever referenced as the discarded first element of each datum
# tuple.

# %% [markdown]
# ## Pixels are still there when you need them
#
# Lazy loading is transparent: any analysis that actually reads the array triggers
# the decode automatically. Inspect a single lazy item to see the proxy before and
# after materialization.

# %%
dataset = DataFrameDataset(catalog, index2label, lazy=True)
image, _, _ = dataset[0]

# shape comes from the header - still not decoded
print(f"Before access: {image!r}")

# any numpy access materializes and caches the pixels
decoded = np.asarray(image)
print(f"After access:  {image!r}")
print(f"Decoded array: {decoded.shape} ({decoded.dtype})")

# %% [markdown]
# Because of this, image-based analyses such as {func}`.compute_stats` work on the
# lazy dataset without any changes - they simply pay the decode cost they actually
# need.

# %%
stats_result = compute_stats(dataset, stats=ImageStats.PIXEL, normalize_pixel_values=False)
print(f"Computed stats over {stats_result['image_count']} images")

# %% [markdown]
# ## A note on the built-in option
#
# The [`maite-datasets`](https://github.com/aria-ml/maite-datasets) package ships
# this pattern out of the box. Its downloadable datasets and readers accept a
# `lazy=True` argument, so for those datasets you get deferred decoding without
# writing any of the proxy code above:
#
# ```python
# from maite_datasets.image_classification import CIFAR10
#
# dataset = CIFAR10("data", image_set="base", download=True, lazy=True)
# ```
#
# Build the proxy yourself, as shown here, when you have a custom dataset that is
# not one of those built-ins.

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
# - [How to add intrinsic factors to Metadata](./h2_add_intrinsic_factors.py)
#
# ### Tutorials
#
# - [Identify bias and correlations](./tt_identify_bias.py)
