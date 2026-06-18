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
# # How to wrap a DataFrame-backed object detection dataset

# %% [markdown]
# ## Problem statement
#
# Object detection catalogues are commonly stored as tabular data, but unlike
# image classification an image holds *many* bounding boxes. The natural way to
# express this in a table is the *long* (tidy) format: **one row per box**, where
# rows that share an image identifier belong to the same image. This is what most
# annotation-tool CSV exports and flattened COCO/Pascal-VOC catalogues look like.
#
# DataEval does not require any particular dataset class. Its evaluators consume
# any object that satisfies the {class}`.AnnotatedDataset` protocol - a minimal
# interface of `__len__`, `__getitem__`, and a `metadata` property. For object
# detection, `__getitem__` returns an {class}`.ObjectDetectionTarget` (bounding
# boxes, labels, and scores) instead of a single class label.
#
# This guide shows you how to wrap a long-format DataFrame in an object detection
# dataset that DataEval can analyze, grouping the per-box rows into one target per
# image.

# %% [markdown]
# ### When to use
#
# Wrap a DataFrame when your images and their bounding-box annotations are
# described by a tabular catalog and you want to run DataEval analyses without
# first exporting to a directory layout or a built-in dataset format.

# %% [markdown]
# ### What you will need
#
# 1. A tabular catalog of your annotations, one row per box (here, a pandas
#    `DataFrame`)
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from dataeval import Metadata
from dataeval.protocols import AnnotatedDataset, DatasetMetadata, DatumMetadata, ObjectDetectionTarget

# %% [markdown]
# ## Build an example catalog
#
# In a real project your catalog would already exist and point at images on disk.
# To keep this guide self-contained, you will write a small handful of images to a
# temporary directory and describe their boxes in a `DataFrame`.
#
# The DataFrame has **one row per bounding box** and these columns:
#
# - `filepath` - where the image lives on disk (repeated for each box in the image)
# - `x`, `y`, `w`, `h` - the box in COCO-style top-left + width/height pixels
# - `label` - the integer class index for the box
# - `weather`, `altitude_m` - **image-level** factors (constant across an image's boxes)
# - `occlusion` - a **box-level** factor that varies from one box to the next

# %%
data_dir = Path(tempfile.mkdtemp())
rng = np.random.default_rng(0)

index2label = {0: "person", 1: "car", 2: "bicycle"}
weather_options = ["clear", "rainy", "foggy"]
occlusion_levels = ["none", "partial", "heavy"]

rows = []
# Creating 30 128x128 images
for i in range(30):
    # Stand-in for a real image file - replace with your own images on disk
    pixels = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    filepath = data_dir / f"img_{i:03d}.png"
    Image.fromarray(pixels).save(filepath)

    # A variable number of boxes per image is exactly why OD catalogs use one row
    # per box rather than one row per image.
    n_boxes = int(rng.integers(1, 5))
    for _ in range(n_boxes):
        x, y = rng.integers(0, 80, size=2)
        w, h = rng.integers(16, 40, size=2)
        rows.append({
            "filepath": str(filepath),
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "label": int(rng.integers(0, 3)),
            "weather": weather_options[i % 3],  # image-level factor (same for every box)
            "altitude_m": float(50 + i),  # image-level factor (same for every box)
            "occlusion": occlusion_levels[int(rng.integers(0, 3))],  # box-level factor (per box)
        })

catalog = pd.DataFrame(rows)
catalog.head()


# %% [markdown]
# ## Write the adapter
#
# DataEval's evaluators only need three things from a dataset, which together make
# up the {class}`.AnnotatedDataset` protocol:
#
# - `__len__()` - the number of items, which for object detection is the number of
#   **unique images**, not the number of rows
# - `__getitem__(index)` - an `(image, target, datum_metadata)` tuple for one image
#    - `image` - an array of shape `(C, H, W)`
#    - `target` - an {class}`.ObjectDetectionTarget`: `boxes` of shape `(N, 4)` in
#      `(x0, y0, x1, y1)` format, `labels` of shape `(N,)`, and `scores` of shape
#      `(N,)`
#    - `datum_metadata` - a `dict` of per-item metadata, which must contain an `id`
# - a `metadata` property - a {class}`.DatasetMetadata` describing the dataset as a
#   whole, including the `index2label` class-name mapping
#
# `ObjectDetectionTarget` is a runtime-checkable protocol, so any object that
# exposes `boxes`, `labels`, and `scores` qualifies. A small dataclass is enough.


# %%
@dataclass
class BoxTarget:
    """A minimal object detection target.

    Implements the ObjectDetectionTarget protocol by exposing the three
    attributes DataEval reads.
    """

    boxes: np.ndarray  # (N, 4) bounding boxes in x0, y0, x1, y1 pixel format
    labels: np.ndarray  # (N,) integer class index per box
    scores: np.ndarray  # (N,) confidence per box; 1.0 for ground truth


# %% [markdown]
# The adapter groups the catalog by image once up front so that positional
# indexing in `__getitem__` maps to a stable list of images. Each lookup decodes
# the image, converts that image's boxes to the format DataEval expects, and
# assembles the target.


# %%
class DataFrameODDataset:
    """An object detection dataset backed by a long-format pandas DataFrame.

    Each row describes one bounding box. Rows that share the image-path column
    belong to the same image and are grouped into a single target. Metadata is
    surfaced at two levels: ``metadata_cols`` are image-level (one value per image)
    and ``box_metadata_cols`` are box-level (one value per box).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        index2label: dict[int, str],
        image_col: str = "filepath",
        box_cols: tuple[str, str, str, str] = ("x", "y", "w", "h"),
        label_col: str = "label",
        metadata_cols: list[str] | None = None,
        box_metadata_cols: list[str] | None = None,
        dataset_id: str = "dataframe-od-catalog",
    ) -> None:
        self._df = dataframe.reset_index(drop=True)
        self.index2label = index2label
        self._box_cols = box_cols
        self._label_col = label_col
        self._metadata_cols = metadata_cols or []
        self._box_metadata_cols = box_metadata_cols or []
        # Group rows by image, preserving first-seen order, so __getitem__(i)
        # always refers to the same image. Each group is a (key, sub-frame) pair.
        self._groups = [rows for _, rows in self._df.groupby(image_col, sort=False)]
        self._image_col = image_col
        # DatasetMetadata advertises the class-name mapping to DataEval
        self.metadata: DatasetMetadata = DatasetMetadata(id=dataset_id, index2label=index2label)

    def __len__(self) -> int:
        return len(self._groups)

    def __getitem__(self, index: int) -> tuple[np.ndarray, BoxTarget, DatumMetadata]:
        rows = self._groups[index]
        first = rows.iloc[0]

        # Decode the image and convert to channels-first (C, H, W)
        image = np.asarray(Image.open(first[self._image_col]).convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)

        # Convert COCO-style (x, y, w, h) boxes to the (x0, y0, x1, y1) format
        # ObjectDetectionTarget expects.
        x, y, w, h = (rows[col].to_numpy(dtype=np.float32) for col in self._box_cols)
        boxes = np.stack([x, y, x + w, y + h], axis=1)

        labels = rows[self._label_col].to_numpy(dtype=np.intp)
        # Ground-truth boxes are certain, so every score is 1.0
        scores = np.ones(len(labels), dtype=np.float32)
        target = BoxTarget(boxes=boxes, labels=labels, scores=scores)

        # Image-level metadata is the same for every box, so take it from the first
        # row as a scalar. Box-level metadata is one value per box, passed as a list.
        # DataEval broadcasts the scalars across the image's detections and expands
        # the lists to one value per detection. Per-item metadata must include an "id".
        datum_metadata = DatumMetadata(
            id=index,
            **{col: first[col] for col in self._metadata_cols},
            **{col: rows[col].tolist() for col in self._box_metadata_cols},
        )

        return image, target, datum_metadata


# %% [markdown]
# Instantiate the adapter over your catalog, pointing it at the metadata columns you
# want DataEval to see.

# %%
dataset = DataFrameODDataset(
    catalog,
    index2label,
    metadata_cols=["weather", "altitude_m"],  # image-level
    box_metadata_cols=["occlusion"],  # box-level
)
print(f"Catalog rows (boxes): {len(catalog)}")
print(f"Dataset length (images): {len(dataset)}")

# %% [markdown]
# Inspect a single item to confirm the shapes and types match what DataEval
# expects. Verify that images have channels first, that targets have
# boxes (N, 4), labels (N,), and scores (N,), and that datum metadata holds an
# id, the image-level factors as scalars, and the box-level `occlusion` as a list
# with one entry per box.

# %%
image, target, datum_metadata = dataset[0]
print(f"image shape:    {image.shape} ({image.dtype})")
print(f"boxes shape:    {target.boxes.shape} (x0, y0, x1, y1)")
print(f"labels:         {target.labels}")
print(f"scores:         {target.scores}")
print(f"datum metadata: {datum_metadata}")

# %% [markdown]
# Because the protocols are runtime-checkable, you can verify structurally that
# both the dataset and its targets are valid for DataEval.

# %%
print(f"Is an AnnotatedDataset:          {isinstance(dataset, AnnotatedDataset)}")
print(f"Target is ObjectDetectionTarget: {isinstance(target, ObjectDetectionTarget)}")

# %% [markdown]
# ## Analyze it with DataEval
#
# The adapter now works anywhere a DataEval object detection dataset is expected.
# Build a {class}`.Metadata` object from it - DataEval reads the per-box labels and
# both the image-level and box-level metadata you exposed.

# %%
metadata = Metadata(dataset)

# "id" is a per-item identifier, not a meaningful factor for bias analysis
metadata.exclude = ["id"]
print(f"Factor names: {metadata.factor_names}")

# %% [markdown]
# ## Image-level and box-level factors
#
# Object detection metadata lives at two levels, and DataEval models **every
# detection as its own row**. Each factor is one of:
#
# - **Image-level** - one value per image (e.g. `weather`, `altitude_m`). You pass it
#   as a scalar and DataEval *broadcasts* it to every box in the image.
# - **Box-level** - one value per box (e.g. `occlusion`). You pass it as a list the
#   length of the image's boxes and DataEval keeps one value *per detection*.
#
# That split is why you give the adapter `metadata_cols` and `box_metadata_cols`
# separately. {class}`.Metadata` lays the dataset out one row per detection, so you
# can see the difference directly: within a single image `weather` repeats while
# `occlusion` changes from box to box.

# %%
print(metadata.target_data.select(["item_index", "target_index", "class_label", "weather", "occlusion"]).head(8))

# %% [markdown]
# That is all it takes: a small adapter that groups per-box rows - and tells
# DataEval which factors are image-level and which are box-level - turns any
# DataFrame-described object detection catalog into something DataEval can analyze,
# with no need to restructure your files on disk.

# %% [markdown]
# ## Adapting to your data
#
# - **Box format.** `ObjectDetectionTarget` uses pixel `(x0, y0, x1, y1)`. If your
#   catalog already stores that, drop the conversion and read the four columns
#   straight through. For other layouts - `xywh`, `cxcywh`, or YOLO-normalized -
#   {class}`.BoundingBox` converts to `xyxy` for you and handles the YOLO image-size
#   scaling, e.g. `BoundingBox(*row, bbox_format=BoundingBoxFormat.YOLO,
#   image_shape=image.shape).xyxy`.
# - **Nested boxes.** If each row is one image with a list or JSON column of boxes
#   instead of one row per box, only the row handling changes: parse that cell in
#   `__getitem__` rather than grouping the DataFrame.

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
# - [How to build a MetadataLike object from a DataFrame](./h2_metadata_from_dataframe.py)
# - [How to delay image loading until needed](./h2_lazy_load_images.py)
# - [How to specify custom statistics on object detection datasets](./h2_custom_image_stats_object_detection.py)
#
# ### Tutorials
#
# - [Identify bias and correlations](./tt_identify_bias.py)
