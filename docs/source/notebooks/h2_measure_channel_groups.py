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
# # How to measure named groups of channels

# %% [markdown]
# ## Problem statement
#
# When analyzing multispectral, hyperspectral, or augmented imagery (such as RGB with
# near-infrared or LIDAR), you should measure statistics on specific band groups rather
# than averaging all channels together. By default, `{func}.compute_stats` averages all
# bands, which can obscure critical channel-specific characteristics (e.g., fusing
# near-infrared and visible ranges).
#
# In this guide, you will learn how to use the `channels` argument to define and name
# groups of bands. You will measure each group separately and output them as distinct
# columns that you can register with `{class}.Metadata`.

# %% [markdown]
# ### When to use
#
# You should use this approach when your images contain channels representing distinct
# physical measurements rather than standard colors, such as:
#
# - RGB augmented with near-infrared, thermal, elevation, or a LIDAR return.
# - Multispectral or hyperspectral imagery, where bands group into meaningful ranges.
# - Any sensor whose bands have different dynamic ranges, where a single scale is
#   insufficient.
#
# You do not need this for standard 1-channel or 3-channel imagery, where
# `{func}.compute_stats` measures the image correctly without additional configuration.

# %% [markdown]
# ### What you will need
#
# 1. A Python environment with `dataeval` installed.
# 1. Imagery with more than three bands (this guide uses a synthetic dataset so you can
#    run it anywhere).

# %% [markdown]
# ## Getting started
#
# First, you will import the required libraries to configure the environment and prepare
# the data.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval
except Exception:
    pass

# %%
import logging

import numpy as np
import polars as pl

from dataeval import Metadata
from dataeval.config import set_max_processes
from dataeval.core import compute_stats
from dataeval.flags import ImageStats
from dataeval.protocols import DatasetMetadata, DatumMetadata
from dataeval.utils.preprocessing import ChannelGroup

# You can limit the process pool to keep memory usage predictable, or leave it unset to
# use all available cores.
set_max_processes(4)

# You should configure logging to see warning messages that are logged rather than
# raised. Refer to [how to configure logging](./h2_configure_logging.py).
logging.basicConfig(format="%(levelname)s: %(message)s", force=True)
logging.getLogger("dataeval").setLevel(logging.WARNING)

# %% [markdown]
# ## Build a four-band dataset
#
# In this section, you will build a small synthetic dataset representing three visible
# bands and one near-infrared band stacked into a single `uint16` array.
#
# You will configure the dataset with two distinct channel groups to demonstrate how
# to handle differing dynamic ranges:
# - The visible bands hold standard 8-bit values (ranging from 0 to 255).
# - The near-infrared band represents a 12-bit sensor (ranging into the thousands).
#
# To simulate a target signal, half the scenes will represent vegetated areas (high
# near-infrared reflectance) and half will represent bare ground. Because vegetation
# appears identical to bare ground in visible light, you will search for the signal
# in the near-infrared band.

# %%
rng = np.random.default_rng(0)


class BandCubes:
    """A small image-classification dataset of four-band cubes: R, G, B, NIR."""

    index2label = {0: "bare", 1: "vegetated"}

    def __init__(self, count: int = 24) -> None:
        self._labels = [i % 2 for i in range(count)]
        self._cubes = []
        for label in self._labels:
            cube = np.empty((4, 32, 32), np.uint16)
            # Visible bands are the same scene either way — 8-bit values.
            cube[:3] = rng.integers(40, 200, (3, 32, 32))
            # Near-infrared separates them, on a 12-bit sensor's scale.
            level = 3000 if label else 500
            cube[3] = rng.integers(level, level + 800, (32, 32))
            self._cubes.append(cube)
        self.metadata = DatasetMetadata(id="band-cubes", index2label=self.index2label)

    def __len__(self) -> int:
        return len(self._cubes)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray, DatumMetadata]:
        return self._cubes[i], np.eye(2)[self._labels[i]], DatumMetadata(id=i)


dataset = BandCubes()
print(f"{len(dataset)} scenes, each {dataset[0][0].shape} of dtype {dataset[0][0].dtype}")

# %% [markdown]
# ## What the unnamed measurement tells you
#
# First, you will execute `{func}.compute_stats` without the `channels` argument to see the
# default behavior, which averages all bands together into a single measurement.

# %%
stats = ImageStats.PIXEL_MEAN | ImageStats.VISUAL_BRIGHTNESS | ImageStats.DIMENSION_DEPTH

whole = compute_stats(dataset, stats=stats, normalize_pixel_values=False)

for name, values in sorted(whole["stats"].items()):
    print(f"{name:12} first scene: {float(values[0]):10.2f}")

# %% [markdown]
# Running this computation leads to three incorrect measurements, though no errors are
# raised:
#
# - **Incorrect depth**: You will see `depth` reports 12 because the container is sized for
#   the largest band. The 8-bit visible bands are measured against an incorrect, larger
#   scale.
# - **Meaningless mean**: You will see `mean` averages both the 8-bit visible bands and
#   the 12-bit near-infrared band, resulting in a value that does not represent any single
#   physical unit.
# - **Inaccurate brightness**: You will see `brightness` is near zero (black). Because the
#   scale is anchored on the maximum value (4095), the visible pixels are mapped
#   incorrectly.
#
# You will receive a logger warning that identifies the affected statistics and suggests
# specifying `channels` to resolve the scaling discrepancy.

# %% [markdown]
# ## Name the band groups
#
# To measure bands individually or in specific subsets, you should pass a mapping of group
# names to band indices using the `channels` argument. Each group is measured jointly,
# and the resulting columns will be prefixed with the group's name.

# %%
grouped = compute_stats(
    dataset,
    stats=stats,
    normalize_pixel_values=False,
    channels={"rgb": [0, 1, 2], "nir": 3},
)

for name, values in sorted(grouped["stats"].items()):
    print(f"{name:16} first scene: {float(values[0]):10.2f}")

# %% [markdown]
# By grouping the channels, you will obtain correct measurements for each band range:
#
# - **Correct depth**: You will see `rgb_depth` is 8 and `nir_depth` is 12.
# - **Comparable mean**: You will see `rgb_mean` is in standard visible-light units and
#   comparable with other 8-bit imagery.
# - **Accurate brightness**: You will see `rgb_brightness` is correctly scaled based on
#   the visible bands.
#
# Note that the unprefixed columns remain in the output to represent the overall
# image-level or band-invariant statistics.

# %% [markdown]
# ## Does the band carry the signal?
#
# To verify that the target signal is properly isolated, you will compare the mean values
# of the visible and near-infrared bands across the two classes.

# %%
labels = np.array([dataset[i][1].argmax() for i in range(len(dataset))])
comparison = pl.DataFrame({
    "class": [dataset.index2label[int(label)] for label in labels],
    "nir_mean": np.asarray(grouped["stats"]["nir_mean"], dtype=float),
    "rgb_mean": np.asarray(grouped["stats"]["rgb_mean"], dtype=float),
    "whole_mean": np.asarray(grouped["stats"]["mean"], dtype=float),
})

print(comparison.group_by("class").agg(pl.all().mean().round(1)).sort("class"))

# %% [markdown]
# In the printed summary, you will observe the following:
#
# - `nir_mean` successfully separates the vegetated and bare classes.
# - `rgb_mean` shows no difference between classes, as expected.
# - `whole_mean` separates the classes but produces an uninterpretable average value due
#   to the combined ranges.
#
# You should use band-wise measurements to pinpoint where differences and signals exist
# across your channels.

# %% [markdown]
# ## Hand the groups to Metadata
#
# You should register the computed statistics with `{class}.Metadata` to use them in
# downstream tasks. You will pass the `{class}.StatsResult` to
# `{meth}.Metadata.add_factors` to automatically align the band-specific statistics with
# the dataset rows.

# %%
metadata = Metadata(dataset)
metadata.add_factors(grouped)

print("factors:", sorted(metadata.factor_names))

# %%
rows = metadata.rows_at("instance")
print(
    rows
    .group_by("class_label")
    .agg(
        pl.col("nir_mean").mean().round(1).alias("nir"),
        pl.col("rgb_mean").mean().round(1).alias("rgb"),
        pl.len().alias("scenes"),
    )
    .sort("class_label")
)

# %% [markdown]
# Once registered, you will be able to analyze the band groups using other DataEval
# tools like balance, diversity, and parity.
#
# Note: In DataEval v1.2, the `per_channel` argument was removed in favor of `channels`.
# If you are migrating standard RGB imagery, you should use
# `channels={"r": 0, "g": 1, "b": 2}`.

# %% [markdown]
# ## Where a scene cannot supply a group
#
# To handle datasets where some scenes are missing specific bands, you should use the
# same group definitions. In this example, you will add a three-band scene that lacks the
# near-infrared channel and compute statistics again.

# %%
ragged = [dataset[0][0], dataset[1][0][:3]]

patchy = compute_stats(
    ragged,
    stats=ImageStats.PIXEL_MEAN | ImageStats.PIXEL_MISSING,
    normalize_pixel_values=False,
    channels={"nir": 3},
)["stats"]

print(pl.DataFrame({name: np.asarray(values, dtype=float) for name, values in sorted(patchy.items())}))

# %% [markdown]
# You will observe that for the scene lacking the channel, `nir_mean` is reported as `NaN`
# and `nir_missing` is reported as `1.0`.
#
# Keep in mind the following behaviors:
# - **Column Alignment**: Columns are never omitted; they remain aligned with all other
#   records to ensure consistent indexing.
# - **All-or-Nothing Measurement**: If a group is defined with multiple bands and any
#   required band is missing, the entire group is marked as absent rather than partially
#   measured.
# - **Using Missing Indicators**: You should use `nir_missing` (which outputs `1.0` when
#   a group is absent) to programmatically detect missing channels in your dataset.

# %% [markdown]
# ## Combine band groups with the background
#
# You should combine `channels` with spatial masking arguments like `per_background` to
# analyze specific bands within localized regions. In this section, you will define
# bounding boxes and calculate near-infrared statistics for the background area of the
# image.

# %%
boxed = [(cube, [(4, 4, 16, 16)]) for cube in (dataset[0][0], dataset[1][0])]

composed = compute_stats(
    [image for image, _ in boxed],
    boxes=[boxes for _, boxes in boxed],
    stats=ImageStats.PIXEL_MEAN,
    per_target=False,
    per_background=True,
    normalize_pixel_values=False,
    channels={"nir": 3},
)["stats"]

print(sorted(composed))

# %% [markdown]
# You will find the prefixed statistic `background_nir_mean` representing the
# near-infrared average for pixels outside any bounding box.
#
# You should check `background_fraction` to ensure the background region covers a
# sufficient proportion of the image before trusting background statistics.

# %% [markdown]
# ## Declare a range for a physical band
#
# For datasets with bands representing physical values (such as elevation, reflectance,
# or temperature) rather than standard integer encodings, you should explicitly declare
# their expected ranges. If a range is undefined, statistics requiring intervals will
# output `NaN`.
#
# You will use the `{class}.ChannelGroup` wrapper to define the active channels and
# specify their value ranges.

# %%
physical = np.stack([
    rng.normal(0.0, 500.0, (32, 32)),  # elevation, metres relative to sea level
    rng.normal(0.0, 50.0, (32, 32)),  # a tighter instrument on the same scene
])

declared = compute_stats(
    [physical],
    stats=ImageStats.PIXEL_ENTROPY,
    normalize_pixel_values=False,
    channels={
        "elevation": ChannelGroup(0, value_range=(-2000.0, 2000.0)),
        "instrument": ChannelGroup(1, value_range=(-200.0, 200.0)),
    },
)["stats"]

for name, values in sorted(declared.items()):
    print(f"{name:20} {float(values[0]):.3f}")

# %% [markdown]
# You will observe that each band is binned over its declared interval, making the
# entropy calculations comparable across runs. The unnamed `entropy` will return `NaN`
# because the overall image lacks a single, coherent value range.

# %% [markdown]
# ## What is not measured per group
#
# Geometric statistics (such as width, height, and coordinates) are not calculated per
# channel group because channel subsetting does not modify the image dimensions.

# %%
geometry = compute_stats(
    dataset,
    stats=ImageStats.DIMENSION_WIDTH | ImageStats.DIMENSION_CHANNELS | ImageStats.DIMENSION_DEPTH,
    normalize_pixel_values=False,
    channels={"rgb": [0, 1, 2], "nir": 3},
)["stats"]

print(sorted(geometry))

# %% [markdown]
# You will see that `width` and `channels` appear as single, unprefixed columns, whereas
# `depth` appears per group (e.g., `rgb_depth`, `nir_depth`) because bands can use different
# encodings.
#
# Note: You should be aware that multi-dimensional or vector-valued statistics (e.g.,
# `rgb_histogram` and `nir_percentiles`) are dropped when you register results with
# `{class}.Metadata` because metadata factors must be one-dimensional.

# %% [markdown]
# ## Ask a different question of each group
#
# If you want to compute different metrics for different channel groups, you should pass
# `stats` as a mapping dictionary. In this mapping, keys represent the group names (or
# `None` for image-wide statistics) and values specify the desired metrics.

# %%
per_group = compute_stats(
    dataset,
    stats={
        None: ImageStats.DIMENSION_WIDTH | ImageStats.DIMENSION_CHANNELS,
        "rgb": ImageStats.VISUAL_BRIGHTNESS | ImageStats.HASH_XXHASH,
        "nir": ImageStats.PIXEL_MEAN,
    },
    normalize_pixel_values=False,
    channels={"rgb": [0, 1, 2], "nir": 3},
)["stats"]

print(sorted(per_group))

# %% [markdown]
# This mapping specifies the exact statistics computed for each view. You will observe that
# any omitted view is not measured.
#
# You should adhere to two requirements when configuring the `stats` mapping:
#
# - **Keys Match Channels**: Your keys must match the defined `channels` keys. Discrepancies
#   will result in configuration errors.
# - **Geometry Key Placement**: You should request geometric statistics under the `None`
#   key because they do not apply to specific channel groups. Asking for geometry in a group
#   key will trigger a warning.

# %% [markdown]
# ## Summary
#
# By naming band groups, you will ensure that DataEval measures multi-band imagery precisely:
#
# - **Range-specific evaluation**: You will measure each group against its own range,
#   preventing differing dynamic ranges from corrupting statistics.
# - **Direct Metadata integration**: You will output statistics as columns on each image's
#   row, allowing direct use in downstream analyses.
# - **Consistent schema**: You will receive consistent column formats, with missing bands
#   marked explicitly rather than omitted.
# - **Spatial composition**: You will combine channel groups with spatial constraints like
#   `per_background`.
# - **Custom metrics per group**: You will configure distinct statistics for each group by
#   passing `stats` as a mapping.
#
# You should use this feature whenever your imagery channels represent distinct physical
# measurements.

# %% [markdown]
# ## Next steps
#
# ### Concepts
#
# - [Image Statistics](../concepts/ImageStatistics.md) — Understand image statistics and metrics computed across channels.
# - [Data Integrity](../concepts/DataIntegrity.md) — Learn how data integrity checks identify quality issues in dataset channels.
# - [Metadata Levels](../concepts/MetadataLevels.md) — Explore dataset, datum, and factor levels in DataEval metadata.
#
# ### Tutorials
#
# - [Compare objects to their surroundings](./tt_compare_objects_to_surroundings.py) — Measure visual context and object contrast against surrounding pixels.
# - [Analyze a dataset across its levels](./tt_analyze_across_levels.py) — Evaluate datasets across instance, item, and dataset levels.
#
# ### How-to guides
#
# - [How to specify custom statistics on object detection datasets](./h2_custom_image_stats_object_detection.py) — Compute custom bounding box and image statistics for object detection tasks.
# - [How to build dataset views](./h2_build_dataset_views.py) — Filter and transform dataset subsets using View and operations.
# - [How to add intrinsic factors to Metadata](./h2_add_intrinsic_factors.py) — Add domain-specific intrinsic factors to Metadata for deeper analysis.
