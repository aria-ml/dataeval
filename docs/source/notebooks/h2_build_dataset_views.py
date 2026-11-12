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
# # Build dataset views with View and Operation
#
# Analysis rarely runs on a dataset as it ships. {class}`.View` builds a curated version of it without copying: it
# applies an ordered pipeline of {class}`.Operation` steps lazily and works anywhere a dataset does.
#
# Use a view when you need:
#
# - only certain classes, a balanced distribution or a capped size
# - a reproducible shuffle or labels rewritten into a shared vocabulary
#
# This guide uses MilitaryVehicles; any {class}`~dataeval.protocols.AnnotatedDataset` works.

# %% tags=["remove_cell"]
# Google Colab Only
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval maite-datasets
except Exception:
    pass

# %%
import numpy as np
from maite_datasets.image_classification import MilitaryVehicles

from dataeval import Metadata
from dataeval.config import set_seed
from dataeval.data import Filter, Limit, Operation, Relabel, Shuffle, View

set_seed(0)  # For reproducibility

# %% [markdown]
# ## Create a view
#
# You use the MilitaryVehicles test split, 1,621 photographs of 24 vehicle types. `lazy=True` reads each image only when
# you access it, and `MilitaryVehicles.hierarchy` groups the types into seven families such as `Tank` and `BMP`.

# %%
vehicles = MilitaryVehicles("./data", image_set="test", download=True, lazy=True)
index2label = vehicles.metadata["index2label"]
class_index = {name: index for index, name in index2label.items()}
families = MilitaryVehicles.hierarchy["vehicle"]["military"]["land vehicle"]  # family -> its vehicle types
labels = Metadata(vehicles).class_labels  # the class of each image, to see what a view kept

print("source dataset size:", len(vehicles))
print("source vocabulary:", index2label)
print("families:", {family: len(types) for family, types in families.items()})

# %% [markdown]
# A {class}`.View` takes a dataset and a list of operations. It behaves like a dataset:
# `len()`, indexing, and iteration all work. Nothing is copied, and the source is
# untouched.

# %%
tank_classes = [class_index[name] for name in families["Tank"]]
view = View(vehicles, operations=[Filter(classes=tank_classes), Limit(200)])

print("view size:", len(view))
print("source is unchanged:", len(vehicles))
print("first 10 source indices behind the view:", view.resolve_indices()[:10])

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(view) == 200
assert len(vehicles) == 1621
assert len(index2label) == 24
assert {family: len(types) for family, types in families.items()}["Tank"] == 6

# %% [markdown]
# The built-in operations:
#
# | Operation | What it does |
# | --- | --- |
# | {class}`.Filter` | keep datums matching classes and/or a metadata-factor predicate, and drop the detections or tracks that do not match |
# | {class}`.ClassBalance` | resample to a balanced class distribution, optionally to a target size |
# | {class}`.Limit` | cap to the first n of the current selection |
# | {class}`.Shuffle` | randomize order, optionally seeded |
# | {class}`.Reverse` | reverse the current order |
# | {class}`.Indices` | keep the given source indices |
# | {class}`.Relabel` | rewrite labels into a target vocabulary, dropping out-of-vocabulary datums |

# %% [markdown]
# ## Order the operations
#
# Operations run in the order you list them, and each one sees the result of the previous.
# The same two operations in opposite orders answer different questions.

# %%
limit_then_filter = View(vehicles, operations=[Limit(200), Filter(classes=tank_classes)])
filter_then_limit = View(vehicles, operations=[Filter(classes=tank_classes), Limit(200)])

print("Limit(200) -> Filter(Tank):", len(limit_then_filter), "(tanks *within* the first 200 images)")
print("Filter(Tank) -> Limit(200):", len(filter_then_limit), "(the first 200 tanks)")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert np.all(np.diff(labels) >= 0)  # the split is stored class by class
assert len(limit_then_filter) == 0
assert len(filter_then_limit) == 200

# %% [markdown]
# The first view is empty, and it raised no error. MilitaryVehicles stores each split class by class, so the first 200
# images belong to the first few classes and none is a tank. `Limit` is a sample only when the order carries no
# meaning, so put {class}`.Shuffle` before it, and a seed makes the sample reproducible.

# %%
first_500 = View(vehicles, operations=[Limit(500)])
sample_500 = View(vehicles, operations=[Shuffle(seed=0), Limit(500)])

print("classes in Limit(500):", len(set(labels[first_500.resolve_indices()])), "of", len(index2label))
print("classes in Shuffle -> Limit(500):", len(set(labels[sample_500.resolve_indices()])), "of", len(index2label))

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(set(labels[first_500.resolve_indices()])) == 8
assert len(set(labels[sample_500.resolve_indices()])) == 24
assert len(sample_500) == 500

# %% [markdown]
# ### Operations see the result of earlier operations
#
# Operations that inspect the data read it through the operations before them. That means a
# filter placed after a relabeling step matches against the new vocabulary, not the
# original one.
#
# Here you collapse the 24 vehicle types into their seven families with {class}`.Relabel`, then filter on class `1`.

# %%
family_names = tuple(families)
family_of = {vehicle_type: family for family, types in families.items() for vehicle_type in types}


def source_types(v: View) -> list[str]:
    """The original vehicle types sitting behind a view's items."""
    return sorted({index2label[int(labels[i])] for i in v.resolve_indices()})


relabel_first = View(vehicles, operations=[Relabel(family_of, family_names), Filter(classes=[1])])
filter_first = View(vehicles, operations=[Filter(classes=[1]), Relabel(family_of, family_names)])

print("vocabulary after Relabel:", relabel_first.metadata.get("index2label"))
print()
print("Relabel -> Filter(classes=[1]):", len(relabel_first), "images, source types", source_types(relabel_first))
print("Filter(classes=[1]) -> Relabel:", len(filter_first), "images, source types", source_types(filter_first))

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert relabel_first.metadata.get("index2label") == dict(enumerate(family_names))
assert family_names[1] == "BMP"
assert index2label[1] == "30N6E"
assert source_types(relabel_first) == ["BMP-1", "BMP-2", "BMP-T15"]
assert source_types(filter_first) == ["30N6E"]
assert (len(relabel_first), len(filter_first)) == (176, 21)
assert int(np.argmax(filter_first[0][1])) == family_names.index("Air Defense")  # relabeled to its family

# %% [markdown]
# With `Relabel` first, class `1` means the new vocabulary's `BMP`, so all three BMP types are kept. With `Filter`
# first, class `1` still means the source type `30N6E`, so only its 21 images survive, and `Relabel` then files them
# under their family, `Air Defense`.

# %% [markdown]
# ## Write your own Operation
#
# Subclass {class}`.Operation` and implement `apply(view)`. Inside `apply` you have three
# levers, and you may use any combination of them:
#
# | Lever | How | Use it for |
# | --- | --- | --- |
# | cardinality / order | set `view.selection` | filtering, reordering, limiting, resampling |
# | content | `view.map(fn)` | rewriting each datum |
# | metadata | override `apply_metadata()` | rewriting dataset-level metadata |
#
# `view.selection` is the list of surviving source indices. `view.map(fn)` registers a
# per-datum transform that is applied lazily on access; pass `where={...}` to target only
# specific source indices. If your operation reads each datum's target, set the class
# attribute `requires` so the view can validate the dataset shape up front.


# %%
class KeepEvery(Operation):
    """Cardinality: keep every nth item of the current selection."""

    def __init__(self, n: int) -> None:
        self.n = n

    def apply(self, view: View) -> None:
        view.selection = view.selection[:: self.n]


class Dim(Operation):
    """Content: darken every image by a factor, as in dusk imagery. Registers a transform, reads no data here."""

    def __init__(self, factor: float) -> None:
        self.factor = factor

    def apply(self, view: View) -> None:
        view.map(lambda datum: ((np.asarray(datum[0]) * self.factor).astype(np.uint8), datum[1], datum[2]))


custom = View(vehicles, operations=[Shuffle(seed=0), Limit(100), KeepEvery(10), Dim(0.5)])
first = custom.resolve_indices()[0]

print("custom view size:", len(custom))
print("source indices:", custom.resolve_indices())
print(
    "mean pixel value of the first item, source:",
    round(float(np.asarray(vehicles[first][0]).mean()), 1),
    "view:",
    round(float(np.asarray(custom[0][0]).mean()), 1),
)

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert custom.resolve_indices() == View(vehicles, [Shuffle(seed=0), Limit(100)]).resolve_indices()[::10]
assert len(custom) == 10
assert np.array_equal(np.asarray(custom[0][0]), (np.asarray(vehicles[first][0]) * 0.5).astype(np.uint8))

# %% [markdown]
# `Dim` never touched the data while building the view. It only registered a
# transform, which runs when an item is read. Operations that filter must read the data to
# decide what to keep, so they scan once at construction. Pure transforms stay free until
# access.

# %% [markdown]
# ## Track what a view did
#
# A view records its own lineage, which is useful for reproducibility and for writing sidecar
# metadata alongside derived datasets.

# %%
nested = View(View(vehicles, [Filter(classes=tank_classes)]), [Limit(25)])

print("operations, innermost first:", nested.operation_groups)
print("original dataset:", type(nested.root).__name__)
print("source indices behind the first 5 items:", nested.resolve_indices()[:5])

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(nested.operation_groups) == 2
assert nested.root is vehicles

# %% [markdown]
# ## Next steps
#
# - [Ontology](../concepts/Ontology.md) — Learn about taxonomic ontologies and label space mapping.
# - [Dataset bias](../concepts/DatasetBias.md) — Understand balance in dataset views.
# - [Conform and merge datasets with different label vocabularies](./h2_conform_and_merge_datasets.py) — Conform
#   and merge datasets with disparate label vocabularies into a unified view.
