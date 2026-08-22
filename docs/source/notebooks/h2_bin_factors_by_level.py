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
# (binning-levels)=
# # How to reason about factor binning across levels

# %% [markdown]
# ## Problem statement
#
# {class}`.Metadata` discretizes every factor before analysis: continuous factors
# are cut into bins, categorical ones into integer codes. Bias evaluators read
# those integers, so where the bin boundaries fall decides what every downstream
# result means.
#
# On an object detection dataset the factors do not all live in the same place.
# `weather` and `altitude_m` describe an **image**; `box_area` describes a single
# **detection**. One image holds many detections, so the two kinds of factor have
# different numbers of values - and a dataset where some images are crowded and
# others are sparse has *unevenly* different numbers.
#
# The level one media item sits at is called the **unit** level, because that item
# is not always an image: on a video dataset a unit is a frame, and
# {attr}`.Metadata.unit_type` reports which it is. Everything below is an image
# dataset, so read "unit-level" as "one value per image".
#
# That raises a question with a wrong answer that is easy to reach: which rows
# should the binner read? If a unit-level factor were discretized over detection
# rows, its value would count once per detection, so crowded images would pull the
# bin boundaries toward themselves and an image with no detections at all would not
# be counted whatsoever.
#
# DataEval bins each factor at **its own level**. This guide shows what that means
# in practice, how to confirm it on your own data, and the one place where the
# distinction still needs your attention.

# %% [markdown]
# ### When to use
#
# Read this when you are interpreting bias or diversity results on a dataset that
# carries factors at more than one level - any object detection dataset with
# per-image metadata - or when you are choosing an `auto_bin_method`.

# %% [markdown]
# ### What you will need
#
# 1. A Python environment with `dataeval` installed
# 1. No data of your own; this guide builds a small synthetic dataset in memory

# %% [markdown]
# ## Getting started
#
# First import the required libraries needed to set up the example.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval
except Exception:
    pass

# This guide is about binning; silence the unrelated level-rename deprecation so
# the output stays on topic.
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# %%
from dataclasses import dataclass

import numpy as np

from dataeval import Metadata
from dataeval.protocols import DatasetMetadata, DatumMetadata

# %% [markdown]
# ## Build a dataset with uneven crowding
#
# The dataset below has 40 images. The first 20 are *sparse* - one detection each
# - and the last 20 are *crowded*, with eight detections apiece. `altitude_m`
# increases across the images and `weather` leans rainy in the crowded half, so
# both unit-level factors are correlated with detection density: exactly the
# situation where the choice of binning level changes the answer.


# %%
@dataclass
class BoxTarget:
    """A minimal object detection target: boxes, labels, and scores."""

    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


class CrowdingDataset:
    """A synthetic detection dataset whose crowding correlates with altitude."""

    def __init__(self, detections_per_image: list[int]) -> None:
        rng = np.random.default_rng(0)
        self._counts = detections_per_image
        # Precomputed so that every read of an item returns the same values.
        # Sorted, so altitude rises with the image index and therefore with crowding.
        self._altitudes = np.sort(rng.uniform(50.0, 400.0, len(detections_per_image)))
        self._areas = [rng.uniform(20.0, 200.0, count) for count in detections_per_image]
        self.metadata = DatasetMetadata(id="crowding-demo", index2label={0: "person", 1: "car"})

    def __len__(self) -> int:
        return len(self._counts)

    def __getitem__(self, index: int) -> tuple[np.ndarray, BoxTarget, DatumMetadata]:
        count = self._counts[index]
        target = BoxTarget(
            boxes=np.tile(np.array([[0.0, 0.0, 10.0, 10.0]]), (count, 1)),
            labels=np.arange(count) % 2,
            scores=np.ones(count),
        )
        # Weather is balanced across images - half clear, half rainy - but correlated
        # with crowding: three quarters of the sparse images are clear and three
        # quarters of the crowded ones are rainy.
        if index < len(self._counts) // 2:
            weather = "rainy" if index % 4 == 0 else "clear"
        else:
            weather = "clear" if index % 4 == 0 else "rainy"
        # altitude_m and weather describe the image; box_area describes each box.
        datum_metadata: DatumMetadata = DatumMetadata(
            id=index,
            **{
                "altitude_m": float(self._altitudes[index]),
                "weather": weather,
                "box_area": self._areas[index].tolist(),
            },
        )
        return np.zeros((3, 32, 32), dtype=np.uint8), target, datum_metadata


counts = [1] * 20 + [8] * 20
dataset = CrowdingDataset(counts)
metadata = Metadata(dataset, auto_bin_method="uniform_count", exclude=["id"])

print(f"images:     {metadata.level_counts['unit']}")
print(f"detections: {metadata.level_counts['instance']}")

# %% [markdown]
# ## Check where each factor lives
#
# {attr}`.Metadata.factor_info` reports the level each factor was defined at. That
# is also the level it was binned at.

# %%
for name, info in metadata.factor_info.items():
    print(f"{name:12s} level={info.level:9s} type={info.factor_type}")

# %% [markdown]
# `altitude_m` and `weather` came back as `unit`, `box_area` as `instance`.
# DataEval derived this from the shape of the metadata each image supplied: a
# scalar per image is unit-level, a list whose length matches that image's
# detection count is instance-level.
#
# A factor's discretized values live in a companion column, named for how the
# factor was processed. `FactorInfo` tells you which one to read.


# %%
def companion(md: Metadata, name: str) -> str:
    """Name of the column holding a factor's discretized values."""
    info = md.factor_info[name]
    if info.is_binned:
        return f"{name}↕"
    return f"{name}#" if info.is_digitized else name


altitude_bins = companion(metadata, "altitude_m")
print(f"altitude_m is stored discretized in {altitude_bins!r}")

# %% [markdown]
# ## Confirm the bin edges come from the images
#
# `altitude_m` is continuous and unit-level, so its bins are cut over the 40 image
# values rather than over the 180 detection rows those values propagate to. With
# `auto_bin_method="uniform_count"` the boundaries sit at quantiles, so you can see
# the difference directly.

# %%
altitudes = metadata.rows_at("unit")["altitude_m"].to_numpy()
replicated = np.repeat(altitudes, counts)

print(f"quartile edges over the 40 images:     {np.round(np.percentile(altitudes, [0, 25, 50, 75, 100]), 1)}")
print(f"quartile edges over the 180 detections: {np.round(np.percentile(replicated, [0, 25, 50, 75, 100]), 1)}")

# %% [markdown]
# The cut points differ, and they differ in a predictable direction: the crowded
# high-altitude images contribute eight values apiece, dragging every interior
# boundary upward. DataEval uses the first row - the images - so the bins describe
# your capture altitudes rather than your detection density.
#
# `uniform_width` is less exposed, since it depends only on the range rather than
# on the shape of the distribution. It is not immune, though, as the section on
# images without detections shows.

# %% [markdown]
# ## The invariant: a bin means the same thing at every level
#
# Because a factor is binned once and the result propagated downwards, an entity's
# bin is a single number no matter which level you read it from. This is what lets
# you compare a result computed on image rows with one computed on detection rows.

# %%
at_unit = metadata.rows_at("unit")[altitude_bins].to_list()
detections = metadata.rows_at("instance")
gathered = [detections.filter(detections["item_index"] == i)[altitude_bins][0] for i in range(len(counts))]

print(f"identical read from either level: {at_unit == gathered}")

# %% [markdown]
# ## Images with no detections still count
#
# An image carrying no detections contributes no detection row at all. Were binning
# done over detection rows, such an image would be invisible to the binner - its
# values absent from the edges, and the image itself left without a bin, whatever
# the binning method. Binning at the unit level includes it like any other.

# %%
sparse = CrowdingDataset([2, 1, 2, 0])
sparse_metadata = Metadata(sparse, exclude=["id"])
sparse_bins = companion(sparse_metadata, "altitude_m")

print(f"images:     {sparse_metadata.level_counts['unit']}")
print(f"detections: {sparse_metadata.level_counts['instance']}")
print(f"altitude bins at unit level: {sparse_metadata.rows_at('unit')[sparse_bins].to_list()}")

# %% [markdown]
# The fourth image has no detections, yet it holds a bin of its own.

# %% [markdown]
# ## The part that still needs your attention
#
# Binning is settled; *row counts* are not. {attr}`.Metadata.factor_data` - what
# every bias evaluator consumes - returns rows at {attr}`.Metadata.view`, which
# defaults to {attr}`.Metadata.label_level` so that they align with
# {attr}`.Metadata.class_labels`. A unit-level factor is replicated onto those
# rows, once per detection. {meth}`.Metadata.at` reads it once per image instead.
#
# The bin values are correct. The *marginal distribution* over those rows is not
# the distribution over your images: it weights each image by how many detections
# it contains.

# %%
unit_rows = metadata.rows_at("unit")["weather"].to_list()
detection_rows = metadata.rows_at("instance")["weather"].to_list()

print(f"weather over images:     clear={unit_rows.count('clear')}, rainy={unit_rows.count('rainy')}")
print(f"weather over detections: clear={detection_rows.count('clear')}, rainy={detection_rows.count('rainy')}")

# %% [markdown]
# Weather is split evenly across the images, yet reads better than two-to-one rainy
# across the detections - entirely because the rainy images are the crowded ones. An
# evaluator consuming `factor_data` sees the second distribution. When the question
# is about *images* - "is my capture schedule biased toward clear weather?" - read
# the factor at the unit level with {meth}`.Metadata.rows_at` rather than through
# `factor_data`.
#
# Both readings are legitimate; they answer different questions. What matters is
# knowing which one an evaluator handed you.

# %% [markdown]
# ## What you learned
#
# - Every factor is binned at its own level, so bin edges describe the entities the
#   factor actually belongs to
# - A bin assignment is the same number read from any level, which keeps results
#   comparable across levels
# - Entities with no children are binned like any other
# - `factor_data` returns instance-level rows, so a unit-level factor's *marginal*
#   there is weighted by detection count - read it with `rows_at` when the question
#   is about images

# %% [markdown]
# ## Next steps
#
# - [Dataset Bias and Coverage](../concepts/DatasetBias.md) — Learn about bias measurement and factor binning strategies.
# - [Acting on Results](../concepts/ActingOnResults.md) — Understand strategies for handling biased or imbalanced data factors.
# - [Identify bias and correlations](./tt_identify_bias.py) — Identify bias and correlations across dataset factor levels.
# - [How to apply DataEval's statistical outputs to Metadata](./h2_add_intrinsic_factors.py) — Augment dataset metadata with calculated statistical factors.
# - [How to wrap a DataFrame-backed object detection dataset](./h2_wrap_dataframe_od_dataset.py) — Wrap custom DataFrame object detection datasets for DataEval.
