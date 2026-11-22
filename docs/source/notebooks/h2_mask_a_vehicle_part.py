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
# # Mask a vehicle part out of image statistics
#
# A hood or a burned-in HUD stays in every frame of its platform, so the statistics of {func}`.image_stats` partly
# describe the vehicle and not the scene. Pass `mask=` and the pixel statistics leave those pixels out. See
# [Image statistics](../concepts/ImageStatistics.md) for what a mask changes and what it leaves whole.
#
# Use it when you compare footage from platforms that carry a part with footage from platforms that do not, as
# {class}`.Outliers` does, or when you set an absolute bound. A mask the size of the frame needs only `dataeval`. A mask
# of another size is scaled with OpenCV, from `dataeval[opencv]`.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval
except Exception:
    pass

# %%
import numpy as np
from numpy.typing import NDArray

from dataeval.config import set_seed
from dataeval.core import image_stats
from dataeval.data import Filter, View
from dataeval.flags import ImageStats
from dataeval.protocols import DatasetMetadata, DatumMetadata
from dataeval.quality import Outliers

set_seed(0)  # For reproducibility

# %% tags=["hide-input"]
HEIGHT, WIDTH = 64, 96
rng = np.random.default_rng(0)


class Frames:
    """Constructed frames stand in for yours: 40 from a clear sensor and 10 from a sensor with a painted hood."""

    def __init__(self) -> None:
        self.metadata = DatasetMetadata(id="frames", index2label={0: "road", 1: "field"})
        self.sensors = ["hood" if i % 5 == 4 else "clear" for i in range(50)]
        self.images: list[NDArray[np.uint8]] = []
        for sensor in self.sensors:
            scene = rng.integers(60, 180, (HEIGHT // 4, WIDTH // 4)).repeat(4, 0).repeat(4, 1)
            frame = np.clip(scene + rng.normal(0, 6, (HEIGHT, WIDTH)), 0, 255).astype(np.uint8)
            if sensor == "hood":
                frame[-20:] = 235
            self.images.append(np.stack([frame] * 3))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[NDArray[np.uint8], NDArray[np.float32], DatumMetadata]:
        target = np.eye(2, dtype=np.float32)[index % 2]  # Filter needs targets, and the guide does not use them
        return self.images[index], target, DatumMetadata(id=index, **{"sensor": self.sensors[index]})


dataset = Frames()

# %% [markdown]
# ## Split the dataset by sensor
#
# Each frame's metadata names its `sensor`, and ten of the 50 frames come from the sensor with the hood.
# {class}`.Filter` takes one {class}`.View` for each sensor. A mask is a 2D image of the frame's size in which nonzero
# marks the pixels to leave out. Cover every pixel the part reaches in any frame, and bind each mask to the name of its
# sensor so that no sensor is measured with another's.

# %%
hood_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
hood_mask[-20:] = True

masks = {"clear": None, "hood": hood_mask}
splits = {sensor: View(dataset, Filter(where=f"sensor = '{sensor}'")) for sensor in masks}
print({sensor: len(split) for sensor, split in splits.items()})

# %% [markdown]
# ## Measure each sensor with its own mask
#
# Give {func}`.image_stats` the mask of the sensor it measures, and none for the sensor with no part. Each split is then
# measured over the pixels its own part leaves. The detail `masked_fraction` says how much of a frame the mask removed,
# and no mask is recorded in the result, so keep the name of each mask with its result.

# %%
stats = ImageStats.DIMENSION | ImageStats.PIXEL | ImageStats.VISUAL
results = [image_stats(split, stats=stats, mask=masks[sensor]) for sensor, split in splits.items()]

print("share of a hood frame removed:", results[1].get("details", {})["masked_fraction"][0])

# %% [markdown]
# ## Find the outliers across the sensors
#
# {meth}`.Outliers.from_stats` takes the list of results. Its `item_index` counts within each split, so each split's
# {meth}`.View.resolve_indices` maps it back to the dataset. An outlier is relative to the rest of the data, so a part
# shows up this way only when the platform carrying it is the minority.


# %%
def flagged(measured):
    """Return the dataset indices of the frames that Outliers flags, given the results of the splits in order."""
    found = Outliers().from_stats(measured)
    return sorted(
        index
        for n, split in enumerate(splits.values())
        for index in split.resolve_indices(found.filter(dataset_index=n)["item_index"].unique().to_list())
    )


unmasked = [image_stats(split, stats=stats) for split in splits.values()]
without_mask, with_mask = flagged(unmasked), flagged(results)
print("flagged without the mask:", without_mask)
print("flagged with the mask:", with_mask)

# %% [markdown]
# Without the mask, every frame with the hood is flagged. With it, none is.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
hood_frames = {i for i, sensor in enumerate(dataset.sensors) if sensor == "hood"}
assert len(dataset) == 50
assert len(hood_frames) == 10
assert {sensor: len(split) for sensor, split in splits.items()} == {"clear": 40, "hood": 10}
assert results[1].get("details", {})["masked_fraction"][0] == 20 / 64
assert hood_frames <= set(without_mask)
assert not hood_frames & set(with_mask)

# %% [markdown]
# ## Next steps
#
# - [Masking a part of the vehicle or a HUD](../concepts/ImageStatistics.md#masking-a-part-of-the-vehicle-or-a-hud) —
#   Draw a mask that covers the part, and read what a masked statistic leaves whole.
# - [Filter a dataset by metadata factors](./h2_filter_by_metadata.py) — Write the predicate that takes each split, as
#   SQL or a polars expression.
# - [Add intrinsic factors to Metadata](./h2_add_intrinsic_factors.py) — Add measured statistics to
#   {class}`.Metadata` for bias analysis.
