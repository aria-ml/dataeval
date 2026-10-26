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
#
# By default {func}`.image_stats` averages every band of an image together, which hides what each band says. In
# multispectral and hyperspectral imagery, or RGB with near-infrared or LIDAR added, the bands carry different physical
# measurements and often different value ranges. The `channels` argument names groups of bands, and each group is
# measured on its own into prefixed columns that {class}`.Metadata` can take.
#
# Use it when an image has more than three bands, or bands with different dynamic ranges.
#
# You need `dataeval`. This guide builds 24 synthetic scenes of red, green, blue (8-bit) and near-infrared (12-bit)
# bands. Half the scenes are vegetated and differ only in the near-infrared band.

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
from dataeval.core import image_stats
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

# %% tags=["hide-input"]
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
# ## Measure the bands as one
#
# Without `channels`, every band is averaged into a single measurement.

# %%
stats = ImageStats.PIXEL_MEAN | ImageStats.VISUAL_BRIGHTNESS | ImageStats.DIMENSION_DEPTH

whole = image_stats(dataset, stats=stats, normalize_pixel_values=False)

for name, values in sorted(whole["stats"].items()):
    print(f"{name:12} first scene: {float(values[0]):10.2f}")

# %% [markdown]
# Nothing raises, but three values are wrong: `depth` reports 12 because the container is sized for the largest band,
# `mean` mixes 8-bit and 12-bit values into a number with no unit, and `brightness` is near zero because the scale is
# anchored on 4095. DataEval logs a warning naming the affected statistics and suggesting `channels`.

# %% [markdown]
# ## Name the band groups
#
# Pass a mapping of group names to band indices. Each group is measured jointly and its columns are prefixed with the
# group's name.

# %%
grouped = image_stats(
    dataset,
    stats=stats,
    normalize_pixel_values=False,
    channels={"rgb": [0, 1, 2], "nir": 3},
)

for name, values in sorted(grouped["stats"].items()):
    print(f"{name:16} first scene: {float(values[0]):10.2f}")

# %% [markdown]
# `rgb_depth` is 8 and `nir_depth` is 12, and `rgb_mean` is in the units of any 8-bit image. The unprefixed columns stay
# for the whole image. Geometry such as `width` is not produced per group, because choosing bands does not change the
# dimensions, and `channels` also composes with `per_background`, which adds `background_nir_mean`.

# %% [markdown]
# ## Hand the groups to Metadata
#
# Pass the whole result to {meth}`.Metadata.add_factors` and each group's statistics land on the row they describe.
# Vector-valued statistics such as `rgb_histogram` are skipped, because a factor must be one-dimensional.

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
# `nir_mean` separates the vegetated scenes from the bare ones, and `rgb_mean` does not. The whole-image `mean`
# separates them too, but cannot say which band carries the signal.

# %% [markdown]
# ## Handle scenes that lack a band
#
# A scene does not have to carry every band. The second scene below has three bands, so it cannot supply the `nir`
# group.

# %%
ragged = [dataset[0][0], dataset[1][0][:3]]

patchy = image_stats(
    ragged,
    stats=ImageStats.PIXEL_MEAN | ImageStats.PIXEL_MISSING,
    normalize_pixel_values=False,
    channels={"nir": 3},
)["stats"]

print(pl.DataFrame({name: np.asarray(values, dtype=float) for name, values in sorted(patchy.items())}))

# %% [markdown]
# The scene without the band gets `nan` for `nir_mean` and 1.0 for `nir_missing`. Columns are never dropped, so they
# stay aligned with every other record, and a group missing any one of its bands is absent as a whole rather than partly
# measured. Filter on `nir_missing` to find those scenes.

# %% [markdown]
# ## Declare ranges and choose statistics per group
#
# A band with physical units needs a declared range, or statistics that need an interval, such as entropy, return `nan`.
# Wrap the band in {class}`.ChannelGroup` and give it a `value_range`. The unprefixed `entropy` is `nan` because the
# whole image has no single range.

# %%
physical = np.stack([
    rng.normal(0.0, 500.0, (32, 32)),  # elevation, meters relative to sea level
    rng.normal(0.0, 50.0, (32, 32)),  # a tighter instrument on the same scene
])

declared = image_stats(
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
# To ask a different question of each group, pass `stats` as a mapping. Its keys must match the `channels` keys, with
# `None` for statistics of the whole image, and geometry belongs under `None`. A group left out is not measured.

# %%
per_group = image_stats(
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

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
boxed = [(cube, [(4, 4, 16, 16)]) for cube in (dataset[0][0], dataset[1][0])]

composed = image_stats(
    [image for image, _ in boxed],
    boxes=[boxes for _, boxes in boxed],
    stats=ImageStats.PIXEL_MEAN,
    per_target=False,
    per_background=True,
    normalize_pixel_values=False,
    channels={"nir": 3},
)["stats"]

print(sorted(composed))
geometry = image_stats(
    dataset,
    stats=ImageStats.DIMENSION_WIDTH | ImageStats.DIMENSION_CHANNELS | ImageStats.DIMENSION_DEPTH,
    normalize_pixel_values=False,
    channels={"rgb": [0, 1, 2], "nir": 3},
)["stats"]

print(sorted(geometry))

assert whole["stats"]["depth"][0] == 12
assert grouped["stats"]["rgb_depth"][0] == 8
assert grouped["stats"]["nir_depth"][0] == 12
assert "mean" in grouped["stats"]
assert "background_nir_mean" in composed
assert {"width", "channels", "rgb_depth", "nir_depth"} <= set(geometry)
assert not {"rgb_width", "nir_width"} & set(geometry)

by_class = (
    rows
    .group_by("class_label")
    .agg(pl.col("nir_mean").mean(), pl.col("rgb_mean").mean(), pl.col("mean").mean())
    .sort("class_label")
)
assert by_class["mean"][1] > 2 * by_class["mean"][0]
assert by_class["nir_mean"][1] > 3 * by_class["nir_mean"][0]
assert abs(by_class["rgb_mean"][1] - by_class["rgb_mean"][0]) < 5
assert {"nir_mean", "rgb_mean"} <= set(metadata.factor_names)

assert np.isnan(patchy["nir_mean"][1])
assert not np.isnan(patchy["nir_mean"][0])
assert patchy["nir_missing"][0] == 0.0
assert patchy["nir_missing"][1] == 1.0

assert np.isnan(declared["entropy"][0])
assert np.isfinite(declared["elevation_entropy"][0])
assert np.isfinite(declared["instrument_entropy"][0])
assert set(per_group) == {"width", "channels", "rgb_brightness", "rgb_xxhash", "nir_mean"}

# %% [markdown]
# ## Next steps
#
# - [Image Statistics](../concepts/ImageStatistics.md) — Understand image statistics and metrics computed across
#   channels.
# - [How to specify custom statistics on object detection datasets](./h2_custom_image_stats_object_detection.py) —
#   Compute custom bounding box and image statistics for object detection tasks.
# - [How to add intrinsic factors to Metadata](./h2_add_intrinsic_factors.py) — Add domain-specific intrinsic factors to
#   Metadata for deeper analysis.
