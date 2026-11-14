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
# # Find crowded, truncated and conflicting annotations
#
# Boxes that crowd each other, run off the image or overlap a box of another class are the ones to review before you
# train or score a detector. The `SPATIAL` statistics of {func}`~dataeval.core.image_stats` measure each box against its
# neighbors and the image edge, and {class}`.Balance` shows whether they depend on the class. See
# [Dataset bias](../concepts/DatasetBias.md#five-questions-about-factors-and-classes) and {class}`.ImageStats`.
#
# Use this guide to see which classes the spatial statistics tie to, and to list the boxes to review. It builds on
# [Add intrinsic factors to Metadata](./h2_add_intrinsic_factors.py), which shows `add_factors` and `Balance`. The data
# is a stand-in with planted defects.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval
except Exception:
    pass

# %%
import warnings

import numpy as np
import polars as pl
from IPython.display import display

from dataeval import Metadata
from dataeval.bias import Balance
from dataeval.core import image_stats
from dataeval.flags import ImageStats

# Balance warns that a declared cut leaves a bin empty, which any small dataset does.
warnings.filterwarnings("ignore", message=".*Declared cuts left bins unused")
pl.Config.set_tbl_rows(20)

# %% [markdown]
# ## Build a detection dataset with known defects
#
# The 80 scenes have four classes, and each planted defect must show in the column that the table names. Only an
# image's shape is read, so the images are flat.
#
# | Defect | Scenes | Planted as | Shows in |
# | --- | --- | --- | --- |
# | crowded | 4 | six packed `person` boxes | `covered_fraction` |
# | rider | 4 | a `person` on a `bicycle` | `max_iou_other_class` |
# | conflicting | 6 | a `car` also annotated as a `van` | `max_iou_other_class` |
# | truncated | 4 | an unclipped `car` past the right edge | `box_in_image_fraction` |
# | placeholder | 20 | a `bicycle` saved with zero area | `spatial_status` |

# %% tags=["hide-input"]
from dataclasses import dataclass

from dataeval.protocols import DatasetMetadata, DatumMetadata

HEIGHT, WIDTH = 120, 160
CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "van"}
SIZES = {0: ((10, 20), (16, 28)), 1: ((12, 24), (10, 18)), 2: ((14, 30), (10, 20)), 3: ((16, 32), (12, 22))}
GROUPS = ["crowded"] * 4 + ["rider"] * 4 + ["conflicting"] * 6 + ["truncated"] * 4


@dataclass
class BoxTarget:
    """The boxes, labels and scores of one image."""

    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


class Scenes:
    """A stand-in object detection dataset of flat images."""

    def __init__(self, boxes: list[list[list[int]]], labels: list[list[int]]) -> None:
        self._boxes, self._labels = boxes, labels
        self.metadata = DatasetMetadata(id="scenes", index2label=CLASSES)

    def __len__(self) -> int:
        return len(self._boxes)

    def __getitem__(self, index: int) -> tuple[np.ndarray, BoxTarget, DatumMetadata]:
        labels = np.asarray(self._labels[index], dtype=np.intp)
        target = BoxTarget(
            np.asarray(self._boxes[index], dtype=np.float32), labels, np.ones(len(labels), dtype=np.float32)
        )
        return np.full((3, HEIGHT, WIDTH), 40 + index, dtype=np.uint8), target, DatumMetadata(id=index)


def is_clear(box: list[int], others: list[list[int]]) -> bool:
    """Whether the box keeps two pixels from every other box."""
    return all(box[0] >= o[2] + 2 or o[0] >= box[2] + 2 or box[1] >= o[3] + 2 or o[1] >= box[3] + 2 for o in others)


def iou(a: list[int], b: list[int]) -> float:
    """The intersection over union of two boxes."""
    width, height = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    inner = max(width, 0) * max(height, 0)
    return inner / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inner)


def build_scenes(seed: int = 0) -> tuple[Scenes, list[str]]:
    """Return the dataset and, for each box in order, the defect planted in it ("" for none)."""
    rng = np.random.default_rng(seed)
    boxes, labels, defects = [], [], []
    scene_boxes: list[list[int]] = []
    scene_labels: list[int] = []
    scene_defects: list[str] = []

    def add(box: list[int], label: int, defect: str = "") -> None:
        scene_boxes.append(box)
        scene_labels.append(label)
        scene_defects.append(defect)

    def place(taken: list[list[int]], label: int, margin: int = 0) -> list[int]:
        for _ in range(500):
            (w0, w1), (h0, h1) = SIZES[label]
            width, height = int(rng.integers(w0, w1 + 1)), int(rng.integers(h0, h1 + 1))
            x = int(rng.integers(margin, WIDTH - width - margin + 1))
            y = int(rng.integers(margin, HEIGHT - height - margin + 1))
            if is_clear(box := [x, y, x + width, y + height], taken):
                return box
        raise RuntimeError("no room for a box")

    for scene in range(80):
        kind = GROUPS[scene] if scene < len(GROUPS) else ""
        scene_boxes.clear()
        scene_labels.clear()
        scene_defects.clear()
        if kind == "crowded":
            left, top = int(rng.integers(10, WIDTH - 80)), int(rng.integers(10, HEIGHT - 50))
            for j in range(6):
                width, height = int(rng.integers(10, 17)), int(rng.integers(18, 27))
                x, y = left + 6 * j + int(rng.integers(0, 2)), top + 4 * (j % 3)
                add([x, y, x + width, y + height], 0, "crowded")
        elif kind == "rider":
            bicycle = place([], 1, margin=8)
            for _ in range(500):  # a person on the bicycle: the same box, higher and a little narrower
                dx0, dx1, up, down = (int(rng.integers(lo, hi)) for lo, hi in ((0, 3), (0, 3), (2, 7), (0, 4)))
                person = [bicycle[0] + dx0, bicycle[1] - up, bicycle[2] - dx1, bicycle[3] - down]
                if 0.55 <= iou(person, bicycle) <= 0.85:
                    break
            else:
                raise RuntimeError("no rider")
            add(bicycle, 1, "rider")
            add(person, 0, "rider")
        elif kind == "truncated":
            width, height = int(rng.integers(20, 33)), int(rng.integers(12, 21))
            x, y = WIDTH - int(width * rng.uniform(0.5, 0.6)), int(rng.integers(0, HEIGHT - height + 1))
            add([x, y, x + width, y + height], 2, "truncated")  # runs off the right edge and is not clipped
        for n in range(int(rng.integers(2, 5))):
            car = place(scene_boxes, 2)
            add(car, 2, "conflicting" if kind == "conflicting" and n == 0 else "")
            if kind == "conflicting" and n == 0:
                add([car[0], car[1], car[2] + 1, car[3] + 1], 3, "conflicting")  # the same car again, as a van
        for _ in range(0 if kind == "crowded" else int(rng.integers(1, 4))):
            add(place(scene_boxes, 0), 0)
        if 18 <= scene < 38:
            add([0, 0, 0, 0], 1, "placeholder")  # a bicycle the annotator could not place, saved as a zero box
        elif 38 <= scene < 48:
            add(place(scene_boxes, 1), 1)
        boxes.append(list(scene_boxes))
        labels.append(list(scene_labels))
        defects.extend(scene_defects)
    return Scenes(boxes, labels), defects


dataset, defects = build_scenes()

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
planted = np.asarray(defects)
assert len(dataset) == 80
assert len(defects) == 456
counts = {
    name: int((planted == name).sum()) for name in ("crowded", "rider", "conflicting", "truncated", "placeholder")
}
assert counts == {"crowded": 24, "rider": 8, "conflicting": 12, "truncated": 4, "placeholder": 20}
again_dataset, again_defects = build_scenes()
assert again_defects == defects
assert all(dataset[i][1].boxes.tolist() == again_dataset[i][1].boxes.tolist() for i in range(80))

# %% [markdown]
# ## Measure the spatial statistics and attach them
#
# Measure `SPATIAL` and each box's size relative to its image. `per_image=False` returns one row per box, since the
# statistics are NaN on an image's own row. {meth}`.Metadata.add_factors` ignores `details`, so promote `spatial_status`
# yourself, with the `source_index`. Declare the bins as in
# [Control and reuse a factor's binning](./h2_control_factor_binning.py).

# %%
stats = image_stats(
    dataset,
    stats=ImageStats.SPATIAL | ImageStats.DIMENSION_REL_WIDTH | ImageStats.DIMENSION_REL_HEIGHT,
    per_image=False,
)
details = stats.get("details", {})

fractions = [0, 0.25, 0.5, 0.75, 1]
fraction_factors = ("box_in_image_fraction", "covered_fraction", "max_iou_same_class", "max_iou_other_class")
bins = dict.fromkeys(fraction_factors, fractions)
metadata = Metadata(dataset, continuous_factor_bins={**bins, "nn_distance_norm": 4, "rel_width": 4, "rel_height": 4})
metadata.add_factors(stats)
metadata.add_factors({"spatial_status": details["spatial_status"]}, source_index=stats["source_index"])

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert dict(metadata.level_counts) == {"unit": 80, "instance": 456}
status = np.asarray(details["spatial_status"])
assert ((status == "invalid_box") == (planted == "placeholder")).all()
for column, values in stats["stats"].items():
    nan = np.isnan(np.asarray(values, dtype=float))
    # Every spatial statistic is NaN on exactly the placeholder boxes. The two size columns stay finite.
    assert (nan == (planted == "placeholder")).all() if column not in ("rel_width", "rel_height") else not nan.any()
assert "spatial_status" in metadata.factor_names

# %% [markdown]
# ## Test whether the statistics track the class
#
# {class}`.Balance` reads each factor against the class. Keep the rows it flags.

# %%
classwise = Balance().evaluate(metadata).classwise
classwise = classwise.with_columns(pl.col("factor_name", "class_name").cast(pl.Utf8))
flagged = classwise.filter(pl.col("is_imbalanced")).sort("mi_value", descending=True)
display(flagged)

# %% [markdown]
# `van` reads highest, against `max_iou_other_class` at 0.73, because every van here is a copy of a car. `bicycle` reads
# against every spatial factor, between 0.48 and 0.56, `spatial_status` included. `rel_width` and `rel_height` are the
# size baseline. Each box counts as one observation, the boxes of an image are not independent, and each defect sits in
# one class, so these associations exist by construction. See [Limitations](../concepts/DatasetBias.md#limitations).

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert flagged.height == 14
top = flagged.row(0, named=True)
assert (top["class_name"], top["factor_name"]) == ("van", "max_iou_other_class")
assert abs(top["mi_value"] - 0.73) < 0.01
labels = np.concatenate([dataset[i][1].labels for i in range(80)])
assert (labels == 3).sum() == 6
assert (planted[labels == 3] == "conflicting").all()  # every van is a copy of a car
spatial = ["nn_distance_norm", "box_in_image_fraction", "covered_fraction", "max_iou_same_class", "max_iou_other_class"]
bicycle = flagged.filter(
    (pl.col("class_name") == "bicycle") & pl.col("factor_name").is_in([*spatial, "spatial_status"])
)
assert bicycle.height == 6
assert bicycle["mi_value"].is_between(0.48, 0.56).all()

# %% [markdown]
# ## Check the missing values by class before you filter
#
# A box with zero area has the status `invalid_box` and NaN spatial statistics. `Balance` counts a missing value as a
# value of its own, so a class with many of them reads against every factor. See
# [Missing values get a bin
# with a position but no meaning](../concepts/Binning.md#missing-values-get-a-bin-with-a-position-but-no-meaning).
# Read each class's share of invalid boxes before you drop the NaN rows.

# %%
boxes = metadata.rows_at("instance").with_columns(
    pl.col("class_label").replace_strict(metadata.index2label).alias("class")
)
invalid = boxes.group_by("class", maintain_order=True).agg(
    pl.len().alias("boxes"),
    (pl.col("spatial_status") == "invalid_box").sum().alias("invalid"),
    (pl.col("spatial_status") == "invalid_box").mean().round(2).alias("share"),
)
display(invalid)

# %% [markdown]
# Only `bicycle` has invalid boxes, 20 of 34. They are NaN in all five spatial statistics, so on their own they can tie
# `bicycle` to every one. Dropping the NaN rows first would remove 59% of the class and hide that.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
counts = {row["class"]: (row["boxes"], row["invalid"]) for row in invalid.iter_rows(named=True)}
assert counts == {"person": (171, 0), "car": (245, 0), "bicycle": (34, 20), "van": (6, 0)}
assert invalid.filter(pl.col("class") == "bicycle")["share"].item() == 0.59
dropped = boxes.filter(pl.col("covered_fraction").is_nan())
assert dropped.height == 20
assert set(dropped["class"]) == {"bicycle"}

# %% [markdown]
# ## Review the candidates
#
# Keep boxes with status `ok`, because polars sorts NaN above every number and a NaN row passes `>= 0.5`. Give each box
# its first reason: it overlaps another class by 0.5 or more, less than 0.9 of it is inside the image, or other boxes
# cover half of it. The cut-offs are this guide's choices, not calibrated.

# %%
pairs = pl.DataFrame({
    "item_index": [index.item for index in stats["source_index"]],
    "target_index": [index.key for index in stats["source_index"]],
    "counterpart": [metadata.index2label.get(label, "") for label in details["max_iou_other_class_label"]],
    "counterpart_box": details["max_iou_other_class_target_index"],
})
reason = (
    pl
    .when(pl.col("max_iou_other_class") >= 0.5)
    .then(pl.lit("overlap"))
    .when(pl.col("box_in_image_fraction") < 0.9)
    .then(pl.lit("truncated"))
    .when(pl.col("covered_fraction") >= 0.5)
    .then(pl.lit("crowded"))
)
candidates = (
    boxes
    .filter(pl.col("spatial_status") == "ok")
    .join(pairs, on=["item_index", "target_index"])
    .with_columns(reason.alias("reason"))
    .filter(pl.col("reason").is_not_null())
)
summary = candidates.group_by("reason", "class", "counterpart", maintain_order=True).agg(
    pl.len().alias("boxes"), pl.first("item_index"), pl.first("target_index"), pl.first("counterpart_box")
)
display(summary)

# %% [markdown]
# A person on a bicycle is two objects, and a car and a van on one spot are one object annotated twice, a conflicting
# annotation. The counterpart is the box of another class that a box overlaps most, at `counterpart_box` in the same
# item, and its label tells the two cases apart. The covered fraction is box coverage by other annotations, not
# occlusion. `box_in_image_fraction` reads 1.0 when the annotation tool clips boxes to the image, so only an unclipped
# box can show truncation.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert candidates.height == 40
assert dict(candidates.group_by("reason").len().iter_rows()) == {"overlap": 20, "crowded": 16, "truncated": 4}

# The rows are in the order of the result, so the planted defects line up with them.
assert boxes["item_index"].to_list() == [index.item for index in stats["source_index"]]
assert boxes["target_index"].to_list() == [index.key for index in stats["source_index"]]
listed = candidates.select("item_index", "target_index").with_columns(pl.lit(1).alias("listed"))
marked = boxes.with_columns(pl.Series("defect", defects)).join(listed, on=["item_index", "target_index"], how="left")
by_defect = marked.group_by("defect").agg(pl.len().alias("boxes"), pl.col("listed").count().alias("listed"))
assert {row["defect"]: (row["boxes"], row["listed"]) for row in by_defect.iter_rows(named=True)} == {
    "": (388, 0),  # no undamaged box is a candidate
    "placeholder": (20, 0),  # the invalid boxes are filtered out
    "rider": (8, 8),
    "conflicting": (12, 12),
    "truncated": (4, 4),
    "crowded": (24, 16),  # the other eight are covered by less than half
}

# Each pair names its counterpart, a car for every van and a person or a bicycle for a rider.
pairs_found = candidates.filter(pl.col("reason") == "overlap").select("class", "counterpart")
assert set(pairs_found.iter_rows()) == {("bicycle", "person"), ("person", "bicycle"), ("car", "van"), ("van", "car")}
assert (candidates.filter(pl.col("class") == "van")["counterpart"] == "car").all()

# Unfiltered, the invalid boxes pass the comparison.
unfiltered = boxes.filter((pl.col("covered_fraction") >= 0.5) & (pl.col("spatial_status") == "invalid_box"))
assert unfiltered.height == 20

# %% [markdown]
# ## Next steps
#
# - [Dataset bias](../concepts/DatasetBias.md) — Learn how Balance reads factors against classes and where it stops.
# - [Identify bias and correlations](./tt_identify_bias.py) — Walk through balance and diversity analysis on a realistic
#   dataset.
# - [Trace a finding back to its source](./h2_trace_findings_to_source.py) — Open the image behind a flagged box.
