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
# # How to measure named groups of channels

# %% [markdown]
# ## Problem statement
#
# Not every image is a picture. A scene may arrive as three visible bands augmented with near-infrared, as a LIDAR
# return stacked beside them, or as a hyperspectral cube of two hundred bands. The interesting questions about that
# data are band-wise. Is the near-infrared band saturated? Which scenes have a dead band? Is the shortwave range
# drifting between train and test? The unit of analysis in each of those is still the image. The band qualifies the
# measurement; it is not a separate thing being measured.
#
# {func}`.compute_stats` on its own gives you neither. Left alone it averages every band together, which fuses
# near-infrared into a number that means nothing. This guide shows how to use its `channels` argument to name groups
# of bands and measure each one separately, as ordinary columns you can hand to {class}`.Metadata`.

# %% [markdown]
# ### When to use
#
# Reach for this when your images carry bands that are different measurements rather than different colors:
#
# - RGB augmented with near-infrared, thermal, elevation or a LIDAR return
# - Multispectral or hyperspectral imagery, where bands group into meaningful ranges
# - Any sensor whose bands have different dynamic ranges, so one scale cannot describe them all
#
# You do not need this for ordinary 1-channel or 3-channel imagery. There the image *is* a picture, one band group,
# and {func}`.compute_stats` already measures it correctly with no extra argument.

# %% [markdown]
# ### What you will need
#
# 1. A Python environment with `dataeval` installed
# 1. Imagery with more than three bands. This guide builds a small synthetic set, so it runs anywhere

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

# Statistics are calculated across a process pool. Capping it keeps this example's memory
# footprint predictable; leave it unset to use every available core.
set_max_processes(4)

# Several points in this guide are reported through DataEval's logger rather than raised.
# Turning it on is what makes them visible here; see
# [how to configure logging](./h2_configure_logging.py) for the full picture.
logging.basicConfig(format="%(levelname)s: %(message)s", force=True)
logging.getLogger("dataeval").setLevel(logging.WARNING)

# %% [markdown]
# ## Build a four-band dataset
#
# A small synthetic stand-in for the real thing: three visible bands and one near-infrared band, stacked into a single
# `uint16` array the way a multispectral reader would hand it over.
#
# The two band groups deliberately disagree about their range. The visible bands hold ordinary 8-bit values, while
# the near-infrared band comes off a 12-bit sensor and runs into the thousands. That disagreement is the whole point.
# It is what a single whole-image measurement cannot represent, and it is what real sensors actually do.
#
# Half the scenes are vegetated and half are bare ground. Vegetation reflects strongly in near-infrared and is
# indistinguishable from bare ground in visible light, so the signal we are looking for lives in exactly one band.

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
# Start without `channels` at all. This is what {func}`.compute_stats` does by default: one measurement over every
# band at once.

# %%
stats = ImageStats.PIXEL_MEAN | ImageStats.VISUAL_BRIGHTNESS | ImageStats.DIMENSION_DEPTH

whole = compute_stats(dataset, stats=stats, normalize_pixel_values=False)

for name, values in sorted(whole["stats"].items()):
    print(f"{name:12} first scene: {float(values[0]):10.2f}")

# %% [markdown]
# Three things have gone wrong, and none of them announce themselves as errors.
#
# `depth` reports 12, because the cube is stored in a container sized for its largest band. The visible bands are
# 8-bit data and are now being measured against a scale four times too large.
#
# `mean` is in no unit at all. It is the average of three reflectance-like bands in the low hundreds and one sensor
# reading in the thousands, so it is a number about the *container*, not about the scene.
#
# `brightness` is nearly black. Brightness is a perceptual reading, a position between black and white, and the
# whole-cube reading anchors that scale on 4095, so genuinely mid-grey visible pixels land near the bottom of it. It
# is also averaging near-infrared into a claim about what a person would see, which is not something near-infrared
# can contribute to.
#
# DataEval says so rather than leaving you to notice. The warning above names the statistics affected and the
# argument that fixes them. It warns rather than refuses, because the dimension statistics beside them are perfectly
# well defined at any band count, and a hard stop would take those down too.

# %% [markdown]
# ## Name the band groups
#
# Pass a mapping from a name to the bands it covers. Each group is measured **jointly**, so `rgb` reduces over bands
# 0, 1 and 2 together, and lands as columns prefixed with the group's name.

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
# Each group is now measured against its own range, which is what the prefixed columns buy you:
#
# - `rgb_depth` is 8 and `nir_depth` is 12. Both ranges are correct and apply to different bands of one image.
# - `rgb_mean` is back in visible-light units, and is directly comparable against any other 8-bit imagery you have.
# - `rgb_brightness` is a real perceptual reading, because it is anchored on the visible bands' own scale rather than
#   on the container's.
#
# The unprefixed columns are still there. The unnamed measurement is always computed, so adding `channels` to a
# pipeline that already reads `brightness` never takes `brightness` away, and the band-invariant statistics need
# somewhere to be reported from.

# %% [markdown]
# ## Does the band carry the signal?
#
# The dataset was built so that vegetation is invisible in the visible bands and obvious in near-infrared. A band-wise
# measurement should show exactly that.

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
# `nir_mean` separates the two classes by roughly a factor of four. `rgb_mean` does not separate them at all, which is
# correct, since the visible bands are the same scene in both. And `whole_mean`, the unnamed measurement, does
# separate them, but only because the near-infrared band drags the average around; the number itself describes
# nothing you can name.
#
# That middle column is the one worth dwelling on. A band-wise measurement can tell you *where* a difference lives,
# and a whole-image measurement cannot.

# %% [markdown]
# ## Hand the groups to Metadata
#
# This is what band columns are for. A whole {class}`.StatsResult` can go straight to
# {meth}`.Metadata.add_factors`, which places every value on the row it describes. From there the band groups are
# ordinary factors, usable by every tool that reads metadata.

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
# `nir_mean` is now a factor like any other, so balance, diversity and parity can all see it.
#
# This is the reason band groups are columns rather than rows. The older `per_channel=True` returns one row per
# channel, on a third axis of the source index. A source index addresses an item and a target, with no level for a
# channel to land on, so per-channel statistics can not reach the factor layer at all. `per_channel` is
# deprecated for this reason; `channels` is its replacement, and for plain RGB the migration is
# `channels={"r": 0, "g": 1, "b": 2}`.

# %% [markdown]
# ## Where a scene cannot supply a group
#
# Real collections are ragged. A sensor is swapped, an older acquisition has fewer bands, a file is truncated. Add one
# three-band scene and ask for the near-infrared group anyway.

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
# The second scene reports `nir_mean` as NaN and `nir_missing` as 1.0.
#
# The column still exists, at full length, aligned with every other column. A group is never quietly skipped, because
# a skipped group would shorten its array and silently offset it against the rest of the result.
#
# The value is absent rather than approximated. A group is **all-or-nothing**: had the group named bands 2 through 5,
# a four-band scene would not be measured over the two it happens to have. One column name has to mean one thing
# across a dataset, and a scene missing bands it should have is a defect that ought to read as a defect.
#
# `nir_missing` is how you find them. Every other statistic reports absence as NaN; this one reports it as 1.0,
# because measuring the lack of *presence* of data is precisely what it is for.

# %% [markdown]
# ## Combine band groups with the background
#
# Band groups slice the channel axis; `per_background` masks the spatial axes. They are separate arguments because
# they compose, and the composition answers questions neither can alone. Whether the unannotated scene is hot in
# near-infrared is a real question about vegetation encroaching outside the labelled objects.

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
# Region first, then band: `background_nir_mean` is the near-infrared measurement of the pixels no box covers.
# `background_fraction` sits beside it and should be read first. A background measured over a few percent of a scene
# is noise wearing a measurement's clothes.

# %% [markdown]
# ## Declare a range for a physical band
#
# Integer imagery carries its range in its encoding, so DataEval reads it off. A band holding *physical* values does
# not: elevation below sea level, mean-centred reflectance, temperature in Celsius. There is nothing to decode,
# so nothing is guessed: statistics that need an interval report NaN, and say so through the logger.
#
# Give such a band its range with {class}`.ChannelGroup`, which is the same mapping value with room for per-group
# options.

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
# Each band is binned over the interval it actually occupies, so the two entropies are comparable to each other and to
# later runs on the same instruments. The unnamed `entropy` is NaN: the two bands genuinely disagree about their
# range, so there is no single interval that would describe the pair, and inventing one from the largest value present
# would produce a number that looks like a measurement and is not.

# %% [markdown]
# ## What is not measured per group
#
# Geometry. Dropping bands does not move a bounding box or change an image's width, so a per-group copy of a geometric
# statistic would restate the plain value under a new name.

# %%
geometry = compute_stats(
    dataset,
    stats=ImageStats.DIMENSION_WIDTH | ImageStats.DIMENSION_CHANNELS | ImageStats.DIMENSION_DEPTH,
    normalize_pixel_values=False,
    channels={"rgb": [0, 1, 2], "nir": 3},
)["stats"]

print(sorted(geometry))

# %% [markdown]
# `width` and `channels` appear once. `depth` appears per group, because bands of one cube can carry different
# encodings and each group's depth is genuinely its own. That is the disagreement this whole guide started from.
#
# Vector-valued statistics are a separate limitation worth knowing about: `rgb_histogram` and `nir_percentiles` are
# computed, but {class}`.Metadata` drops multi-dimensional factors, so they do not reach the factor layer. That is
# true of their unprefixed forms as well, and is not specific to band groups.

# %% [markdown]
# ## Summary
#
# Naming band groups turns a multi-band image from something DataEval measures badly into something it measures the
# way you would want to by hand:
#
# - each group against its own range, so bands with different dynamic ranges stop corrupting each other
# - as columns on the image's row, so they reach {class}`.Metadata` and everything downstream of it
# - all-or-nothing where a scene cannot supply a group, so one column name means one thing
# - composable with `per_background`, so region and band cross rather than compete
#
# Reach for it whenever the bands of your imagery are different measurements. Leave it alone when they are not.

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
