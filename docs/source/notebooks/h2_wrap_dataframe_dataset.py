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
# # How to wrap a DataFrame as a dataset
#
# Datasets are often cataloged as a table, such as a CSV or parquet file, with a row per image or per box and columns
# for the file location, the label and any metadata. DataEval's evaluators accept any object that satisfies the
# {class}`.AnnotatedDataset` protocol (`__len__`, `__getitem__` and a `metadata` property), so a thin adapter turns a
# pandas `DataFrame` into a dataset without exporting it to a directory layout first.
#
# Use it when a tabular catalog describes your images and you want to analyze them in place, for classification or
# detection.
#
# You need `dataeval`, `pandas` and `pillow`. This guide writes small images to a temporary directory and describes them
# in two DataFrames.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image

from dataeval import Metadata
from dataeval.bias import Balance
from dataeval.protocols import AnnotatedDataset, DatasetMetadata, DatumMetadata

# %% tags=["hide-input"]
data_dir = Path(tempfile.mkdtemp())
rng = np.random.default_rng(0)
weather_options = ["clear", "rainy", "foggy"]

# Classification catalog: one row per image. The images are noise standing in for files on disk.
index2label = {0: "cat", 1: "dog", 2: "bird"}
rows = []
for i in range(300):
    filepath = data_dir / f"img_{i:03d}.png"
    Image.fromarray(rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)).save(filepath)
    rows.append({
        "filepath": str(filepath),
        "label": i % 3,
        "weather": weather_options[(i // 3) % 3],
        "altitude_m": float(50 + i),
    })
catalog = pd.DataFrame(rows)

# Detection catalog: one row per box, so an image with three boxes has three rows.
od_index2label = {0: "person", 1: "car", 2: "bicycle"}
occlusion_levels = ["none", "partial", "heavy"]
box_rows = []
for i in range(30):
    filepath = data_dir / f"box_{i:03d}.png"
    Image.fromarray(rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)).save(filepath)
    for _ in range(int(rng.integers(1, 5))):
        x, y = rng.integers(0, 80, size=2)
        w, h = rng.integers(16, 40, size=2)
        box_rows.append({
            "filepath": str(filepath),
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "label": int(rng.integers(0, 3)),
            "weather": weather_options[i % 3],  # image-level: the same for every box
            "occlusion": occlusion_levels[int(rng.integers(0, 3))],  # box-level: one value per box
        })
od_catalog = pd.DataFrame(box_rows)

# %% [markdown]
# ## Wrap an image classification catalog
#
# The catalog has one row per image: its `filepath`, an integer `label` and two factors, `weather` and `altitude_m`.

# %%
catalog.head()


# %% [markdown]
# The adapter returns an `(image, target, datum_metadata)` tuple for each item. The image is a channels-first
# `(C, H, W)` array, the target is the one-hot label, and the datum metadata is a dict that must contain an `id`. The
# `metadata` property describes the dataset as a whole, including the `index2label` mapping.


# %%
def load_chw(path: str) -> NDArray[np.uint8]:
    """Decode an image file to a channels-first (C, H, W) array."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)


class DataFrameICDataset:
    """An image classification dataset with one DataFrame row per image."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        index2label: dict[int, str],
        metadata_cols: list[str] | None = None,
        dataset_id: str = "dataframe-catalog",
    ) -> None:
        self._df = dataframe.reset_index(drop=True)  # keeps positions aligned with __getitem__
        self.index2label = index2label
        self._metadata_cols = metadata_cols or []
        self.metadata: DatasetMetadata = DatasetMetadata(id=dataset_id, index2label=index2label)

    def __len__(self) -> int:
        return len(self._df)

    def _image(self, path: str) -> Any:
        return load_chw(path)

    def __getitem__(self, index: int) -> tuple[Any, np.ndarray, DatumMetadata]:
        row = self._df.iloc[index]
        target = np.zeros(len(self.index2label), dtype=np.float32)
        target[int(row["label"])] = 1.0
        datum_metadata = DatumMetadata(id=index, **{col: row[col] for col in self._metadata_cols})
        return self._image(row["filepath"]), target, datum_metadata


# %%
dataset = DataFrameICDataset(catalog, index2label, metadata_cols=["weather", "altitude_m"])
metadata = Metadata(dataset)

print(f"{len(dataset)} images, factors: {metadata.factor_names}")

# %% [markdown]
# The `id` names the datum rather than describing it, so DataEval keeps it out of the factors on its own. The factors
# are exactly the metadata columns the adapter exposed.

# %% [markdown]
# ## Wrap an object detection catalog
#
# The catalog has one row per box, and rows that share a `filepath` belong to one image. The `x`, `y`, `w` and `h`
# columns are the box in top-left corner and size, `weather` holds for the whole image, and `occlusion` varies from box
# to box.

# %%
od_catalog.head()


# %% [markdown]
# `__len__` counts images, not rows, and the target is an {obj}`.ObjectDetectionTarget`: `boxes` of shape `(N, 4)` in
# `(x0, y0, x1, y1)` pixels, `labels` and `scores`. Any object with those attributes qualifies, so a small dataclass is
# enough. The adapter reuses the classification one, groups the rows by image once, and converts each image's boxes.
# Image-level factors go in as scalars and box-level factors as lists with one value per box.


# %%
@dataclass
class BoxTarget:
    boxes: np.ndarray  # (N, 4) boxes in x0, y0, x1, y1 pixels
    labels: np.ndarray  # (N,) class index per box
    scores: np.ndarray  # (N,) confidence per box; 1.0 for ground truth


class DataFrameODDataset(DataFrameICDataset):
    """An object detection dataset with one DataFrame row per box."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        index2label: dict[int, str],
        metadata_cols: list[str] | None = None,
        box_metadata_cols: list[str] | None = None,
        dataset_id: str = "dataframe-od-catalog",
    ) -> None:
        super().__init__(dataframe, index2label, metadata_cols, dataset_id)
        self._box_metadata_cols = box_metadata_cols or []
        # Group the rows by image in first-seen order, so __getitem__(i) always names the same image
        self._groups = [rows for _, rows in self._df.groupby("filepath", sort=False)]

    def __len__(self) -> int:
        return len(self._groups)

    def __getitem__(self, index: int) -> tuple[Any, BoxTarget, DatumMetadata]:  # pyright: ignore[reportIncompatibleMethodOverride]
        rows = self._groups[index]
        first = rows.iloc[0]
        x, y, w, h = (rows[col].to_numpy(dtype=np.float32) for col in ("x", "y", "w", "h"))
        labels = rows["label"].to_numpy(dtype=np.intp)
        target = BoxTarget(np.stack([x, y, x + w, y + h], axis=1), labels, np.ones(len(labels), dtype=np.float32))
        datum_metadata = DatumMetadata(
            id=index,
            **{col: first[col] for col in self._metadata_cols},
            **{col: rows[col].tolist() for col in self._box_metadata_cols},
        )
        return self._image(first["filepath"]), target, datum_metadata


# %%
od_dataset = DataFrameODDataset(
    od_catalog,
    od_index2label,
    metadata_cols=["weather"],
    box_metadata_cols=["occlusion"],
)
od_metadata = Metadata(od_dataset)

print(f"{len(od_catalog)} boxes in {len(od_dataset)} images")
print(
    od_metadata
    .rows_at(od_metadata.label_level)
    .select(["item_index", "target_index", "class_label", "weather", "occlusion"])
    .head(8)
)


# %% [markdown]
# DataEval models every detection as its own row, so within an image `weather` repeats and `occlusion` changes from box
# to box. For a catalog that does not store `(x0, y0, x1, y1)`, {class}`.BoundingBox` converts `xywh`, `cxcywh` and
# YOLO-normalized boxes.

# %% [markdown]
# ## Defer image decoding
#
# Decoding an image costs far more than reading its label, and {class}`.Metadata` and the bias evaluators never read the
# pixels. Return a `LazyArray` from `_image` instead. It satisfies the {class}`.Array` protocol, reads its shape from
# the file header, and decodes only on the first numpy access.


# %%
def read_shape(path: str) -> tuple[int, ...]:
    """Read (3, H, W) from the image header without decoding the pixels."""
    with Image.open(path) as im:
        width, height = im.size
    return (3, height, width)


class LazyArray:
    """An image file that is decoded on first numpy access and cached."""

    def __init__(
        self,
        path: str,
        loader: Callable[[str], NDArray[Any]],
        shape_loader: Callable[[str], tuple[int, ...]],
    ) -> None:
        self._path, self._loader, self._shape_loader = path, loader, shape_loader
        self._array: NDArray[Any] | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape if self._array is not None else self._shape_loader(self._path)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> NDArray[Any]:
        if self._array is None:
            self._array = self._loader(self._path)
        return self._array if dtype is None else self._array.astype(dtype)

    def __getitem__(self, key: Any) -> Any:
        return np.asarray(self)[key]

    def __iter__(self):
        return iter(np.asarray(self))

    def __len__(self) -> int:
        return self.shape[0]


class LazyICDataset(DataFrameICDataset):
    def _image(self, path: str) -> LazyArray:
        return LazyArray(path, load_chw, read_shape)


# %% [markdown]
# Time a metadata bias pass over the 300 images with eager and with lazy loading. The same override works on the
# detection adapter.


# %%
def time_metadata_pass(dataset: DataFrameICDataset) -> float:
    start = time.perf_counter()
    Balance().evaluate(Metadata(dataset))
    return time.perf_counter() - start


eager_seconds = time_metadata_pass(DataFrameICDataset(catalog, index2label, ["weather"]))
lazy_seconds = time_metadata_pass(LazyICDataset(catalog, index2label, ["weather"]))

print(f"Eager (decodes every image): {eager_seconds * 1000:7.1f} ms")
print(f"Lazy  (decodes none):        {lazy_seconds * 1000:7.1f} ms")

# %% [markdown]
# The lazy pass decodes nothing, because the images are only the discarded first element of each datum tuple. An
# analysis that does read the array, such as {func}`.image_stats`, triggers the decode and works unchanged. The
# downloadable datasets in [`maite-datasets`](https://github.com/aria-ml/maite-datasets) accept `lazy=True`, so build
# the proxy yourself only for a custom dataset.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
from dataeval.core import image_stats
from dataeval.flags import ImageStats
from dataeval.protocols import Array, ObjectDetectionTarget

image, target, datum_metadata = dataset[0]
assert isinstance(dataset, AnnotatedDataset)
assert image.shape == (3, 256, 256)
assert target.tolist() == [1.0, 0.0, 0.0]
assert set(datum_metadata) >= {"id", "weather", "altitude_m"}
assert set(metadata.factor_names) == {"weather", "altitude_m"}

od_image, od_target, od_datum = od_dataset[0]
assert isinstance(od_dataset, AnnotatedDataset)
assert isinstance(od_target, ObjectDetectionTarget)
assert od_image.shape == (3, 128, 128)
assert od_target.boxes.shape == (len(od_target.labels), 4)
assert len(od_dataset) == od_catalog["filepath"].nunique() < len(od_catalog)
assert len(od_datum["occlusion"]) == len(od_target.labels)  # pyright: ignore[reportGeneralTypeIssues]
assert set(od_metadata.factor_names) == {"weather", "occlusion"}

lazy_dataset = LazyICDataset(catalog, index2label, ["weather"])
lazy_image = lazy_dataset[0][0]
assert isinstance(lazy_image, Array)
assert lazy_image.shape == (3, 256, 256)
assert np.array_equal(np.asarray(lazy_image), image)
assert image_stats(lazy_dataset, stats=ImageStats.PIXEL, normalize_pixel_values=False)["image_count"] == len(catalog)
assert lazy_seconds < eager_seconds

# %% [markdown]
# ## Next steps
#
# - [Dataset Bias and Coverage](../concepts/DatasetBias.md) — Understand how class imbalance and missing data impact model reliability.
# - [Identify bias and correlations](./tt_identify_bias.py) — Analyze class and metadata correlations to spot potential dataset bias.
# - [How to build a MetadataLike object from a DataFrame](./h2_metadata_from_dataframe.py) — Build a MetadataLike object from a DataFrame for bias analysis without loading images.
