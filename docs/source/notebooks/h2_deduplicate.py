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
# # Identify duplicates
#
# Duplicates add no new information and can distort model training by over-emphasizing the features that appear in them.
# {class}`.Duplicates` finds exact and near duplicate images so you can remove them before you train. It also looks past
# pixels: when a dataset carries object detection targets, it checks whether two items share one annotation (a synthetic
# copy) or share pixels but disagree on annotation (one collect labeled twice).
#
# Use it when you need to find duplicate images in a dataset, as one of the first steps of exploratory data analysis.
#
# You need `dataeval`, `maite-datasets` and a dataset to analyze.

# %% tags=["remove_cell"]
# Google Colab Only
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval maite-datasets
except Exception:
    pass

# %%
from dataclasses import dataclass

import numpy as np
from IPython.display import display
from maite_datasets.image_classification import MilitaryVehicles

from dataeval import Metadata
from dataeval.config import set_max_processes
from dataeval.data import Indices, View
from dataeval.protocols import DatasetMetadata, DatumMetadata
from dataeval.quality import Duplicates

set_max_processes(4)

# %% [markdown]
# ## Load the data
#
# Load the MilitaryVehicles train split, 7,823 photographs of 24 types of ground vehicle, and read the class of each
# image.

# %%
train_ds = MilitaryVehicles("./data", image_set="train", download=True, lazy=True)
index2label = train_ds.metadata["index2label"]

# Get the labels
labels = Metadata(train_ds).class_labels

# %% [markdown]
# ## Find the duplicates
#
# Ask the `Duplicates` class to find the duplicates. You do not need to plant any: the dataset holds its own. See
# [Clean a dataset](./tt_clean_dataset.py) for how to inspect the groups before you remove anything.

# %%
# Initialize the Duplicates class to begin to identify duplicate images. A hash radius of 0 counts two images as
# duplicates only when their hashes are identical.
identifyDuplicates = Duplicates(hash_radius=0)

# Evaluate the data
results = identifyDuplicates.evaluate(train_ds)

# %% [markdown]
# The results can be viewed as a DataFrame with exact and near groups enumerated.

# %%
display(results)

# %% [markdown]
# Each exact group is one photograph that appears more than once with identical pixels, and each near group holds
# photographs whose perceptual hashes match. Check what labels the images in each exact group carry.

# %%
different = sum(len({labels[i] for i in group}) > 1 for group in results.exact)
first_group = results.exact[0]
print(f"{len(results.exact)} exact groups and {len(results.near)} near groups in {len(train_ds):,} images")
print(f"Exact groups whose images carry different labels: {different} of {len(results.exact)}")
print(f"The first exact group, images {first_group}, is filed under {[index2label[labels[i]] for i in first_group]}")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(train_ds) == 7823
assert (len(results.exact), len(results.near)) == (67, 38)
assert different == len(results.exact)  # the text says every exact group disagrees on the label
assert first_group == [478, 6118]
assert [index2label[labels[i]] for i in first_group] == ["BM-30", "T-64"]
assert all(
    np.array_equal(np.asarray(train_ds[group[0]][0]), np.asarray(train_ds[i][0]))
    for group in results.exact
    for i in group
)  # identical pixels

# %% [markdown]
# Every exact group holds identical pixels filed under different vehicles, so the same photograph teaches the model two
# answers. `Duplicates` finds the copies. It cannot tell which label is right, and that decision is yours.

# %% [markdown]
# ## Remove the duplicates
#
# `deduplicate()` turns the result into a plan of what to remove. Its arguments state the policy: which kinds of
# duplicate to collapse, and which member of each set to keep. By default it collapses exact duplicates and keeps
# the first member of each set. Near duplicates are a judgment about how similar is too similar, so you add them with
# `dup_types=["exact", "near"]` once you have looked at them.
#
# Pass the plan to `Indices` with `exclude=True`. Build the view on `train_ds`, the dataset the result was
# computed on, because the plan names positions in that dataset.

# %%
plan = results.deduplicate()
deduplicated_ds = View(train_ds, Indices(plan, exclude=True))
print(plan)
print(f"{len(train_ds)} items before, {len(deduplicated_ds)} after")

# %% [markdown]
# The plan removes 72 images from 67 groups because five groups hold three copies. Here the choice of survivor decides
# the label. Compare `keep="first"`, the default, with `keep="last"` on the first group.

# %%
kept = {
    keep: set(View(train_ds, Indices(results.deduplicate(keep=keep), exclude=True)).resolve_indices())
    for keep in ("first", "last")
}
for keep, survivors in kept.items():
    print(f'keep="{keep}" keeps', [index2label[labels[i]] for i in first_group if i in survivors])

# %% [markdown]
# The two policies keep different labels for the same photograph, and neither knows which is correct. Pass
# `exclude_groups` the `group_id` of any group you want to leave alone and settle by hand, or correct the labels first
# and then apply the plan.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert sum(len(group) for group in results.exact) - len(results.exact) == len(plan) == 72
assert len(deduplicated_ds) == len(train_ds) - len(plan) == 7751
assert [index2label[labels[i]] for i in first_group if i in kept["first"]] == ["BM-30"]
assert [index2label[labels[i]] for i in first_group if i in kept["last"]] == ["T-64"]

# %% [markdown]
# ## Find annotation duplicates
#
# MilitaryVehicles has no object detection targets, so everything above compared pixels only. When a dataset does carry
# targets (boxes and labels), `Duplicates` also digests the annotation itself, which is on by default and needs no
# argument. Two accessors narrow the result to the annotation-driven relations:
#
# - `divergent()`: items whose annotation was checked and found to disagree, meaning one collect labeled two different
#   ways.
# - `augmented()`: items sharing one annotation over different pixels, the signature of a synthetic copy.
#
# Real detection datasets rarely hold a clean pair of each kind, so this guide builds a tiny object detection dataset
# with four items to see both. Items 0 and 1 have identical pixels and two
# different boxes, a divergent pair. Items 2 and 3 have the same box over two independently sampled images, an augmented
# pair.


# %% tags=["hide-input"]
@dataclass
class BoxTarget:
    """A minimal object detection target: boxes, labels, and scores."""

    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


class TinyObjectDetectionDataset:
    """Four items as two duplicate pairs: one relabeled, one augmented."""

    def __init__(self, images: list[np.ndarray], boxes: list[list[list[float]]], labels: list[list[int]]) -> None:
        self._images = images
        self._boxes = boxes
        self._labels = labels
        self.metadata: DatasetMetadata = DatasetMetadata(id="tiny-od", index2label={0: "object"})

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[np.ndarray, BoxTarget, DatumMetadata]:
        item_labels = np.asarray(self._labels[index], dtype=np.intp)
        target = BoxTarget(
            boxes=np.asarray(self._boxes[index], dtype=np.float32),
            labels=item_labels,
            scores=np.ones(len(item_labels), dtype=np.float32),
        )
        return self._images[index], target, DatumMetadata(id=index)


rng = np.random.default_rng(0)
same_image = rng.random((3, 16, 16))

annotation_images = [same_image, same_image.copy(), rng.random((3, 16, 16)), rng.random((3, 16, 16))]
annotation_boxes = [[[1.0, 1.0, 6.0, 6.0]], [[9.0, 9.0, 14.0, 14.0]], [[0.0, 0.0, 5.0, 5.0]], [[0.0, 0.0, 5.0, 5.0]]]
annotation_labels = [[0], [0], [0], [0]]

annotation_ds = TinyObjectDetectionDataset(annotation_images, annotation_boxes, annotation_labels)

# %%
annotation_results = Duplicates(hash_radius=0).evaluate(annotation_ds)

# %% [markdown]
# `divergent()` finds the pair whose pixels matched but whose annotation did not. That is a label-quality issue, not a
# redundancy one.

# %%
display(annotation_results.divergent())

# %% [markdown]
# `augmented()` finds the pair whose annotation matched but whose pixels did not. That is a synthetic copy's signature.

# %%
display(annotation_results.augmented())

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(annotation_results.divergent()) == 1
assert annotation_results.divergent().data()["item_indices"].item().to_list() == [0, 1]
assert len(annotation_results.augmented()) == 1
assert annotation_results.augmented().data()["item_indices"].item().to_list() == [2, 3]

# %% [markdown]
# ## Next steps
#
# - [Data integrity](../concepts/DataIntegrity.md) — Analyze image-level and target-level statistics to identify data
#   quality issues.
# - [Identify duplicate video](./tt_identify_duplicate_video.py) — Find duplicate footage, train/test leakage, and
#   redundant frames in video and MOT datasets.
# - [Detect common augmentations as duplicates](./tt_augmentation_duplicates.py) — Find near-identical images created
#   through synthetic transformations and augmentations.
