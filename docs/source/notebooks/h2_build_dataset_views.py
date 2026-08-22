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
# # How to build dataset views with View and Operation

# %% [markdown]
# ## Problem statement
#
# Analysis rarely runs on a dataset exactly as it ships. You need a *curated* version of it:
# only certain classes, a balanced class distribution, a capped size, a reproducible shuffle,
# or labels rewritten into a shared vocabulary — often several of these at once.
#
# Copying the data to do this is wasteful and loses provenance. {class}`.View` instead wraps a
# dataset and applies an ordered pipeline of {class}`.Operation` steps **lazily**, producing a
# new dataset-shaped object without modifying or duplicating the original.

# %% [markdown]
# ### When to use
#
# Use a {class}`.View` whenever you need a subset or a rewritten version of a dataset:
# building train/test splits, focusing an experiment on a few classes, balancing an
# imbalanced dataset, capping a dataset for a quick run, or conforming labels before merging
# datasets. A view is accepted anywhere a dataset is — pass it straight to
# {class}`.Embeddings`, {class}`.Metadata`, or any evaluator.

# %% [markdown]
# ### What you will need
#
# 1. Any dataset implementing {class}`~dataeval.protocols.AnnotatedDataset`.
# 1. A Python environment with the following packages installed:
#    - `dataeval`
#    - `maite-datasets`

# %% [markdown]
# ## Getting started
#
# Let's import the required libraries needed to set up a minimal working example

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
from maite_datasets.image_classification import MNIST

from dataeval.config import set_seed
from dataeval.data import ClassFilter, Limit, Operation, Relabel, Shuffle, View

set_seed(0)  # For reproducibility

# %% [markdown]
# ## A dataset to work with
#
# We use the MNIST test split (10,000 handwritten digits). Any
# {class}`~dataeval.protocols.AnnotatedDataset` works the same way.

# %%
mnist = MNIST("./data", image_set="test", download=True)
print("source dataset size:", len(mnist))
print("source vocabulary:", mnist.metadata.get("index2label"))

# %% [markdown]
# ## The basics
#
# A {class}`.View` takes a dataset and a list of operations. It behaves like a dataset —
# `len()`, indexing, and iteration all work — but nothing is copied, and the source is
# untouched.

# %%
view = View(mnist, operations=[ClassFilter([0, 1]), Limit(500)])

print("view size:", len(view))
print("source is unchanged:", len(mnist))
print("first 10 source indices behind the view:", view.resolve_indices()[:10])

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(view) == 500
assert len(mnist) == 10000

# %% [markdown]
# ## Order matters
#
# Operations run **in the order you list them**, and each one sees the result of the previous.
# This makes a pipeline read top-to-bottom, but it also means order is meaningful — the same
# two operations in opposite orders answer different questions.

# %%
limit_then_filter = View(mnist, operations=[Limit(500), ClassFilter([0, 1])])
filter_then_limit = View(mnist, operations=[ClassFilter([0, 1]), Limit(500)])

print("Limit(500) -> ClassFilter([0, 1]):", len(limit_then_filter), "(0s and 1s *within* the first 500)")
print("ClassFilter([0, 1]) -> Limit(500):", len(filter_then_limit), "(the first 500 0s and 1s)")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(limit_then_filter) < len(filter_then_limit) == 500

# %% [markdown]
# Because each operation applies to the current selection, an operation can appear more than
# once. This pipeline takes a reproducible random sample *from a specific window* of the
# dataset — something a single pass could not express.

# %%
windowed = View(mnist, operations=[Limit(2000), Shuffle(seed=0), Limit(10)])
print("10 random items drawn from the first 2,000:", windowed.resolve_indices())

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(windowed) == 10
assert all(i < 2000 for i in windowed.resolve_indices())

# %% [markdown]
# ## Operations see the result of earlier operations
#
# Operations that inspect the data read it *through* the operations before them. That means a
# filter placed after a relabeling step matches against the **new** vocabulary, not the
# original one.
#
# Here we collapse the ten digits into `even`/`odd` with {class}`.Relabel`, then filter on the
# relabeled classes.

# %%
digit_names = mnist.metadata.get("index2label", {})
EVEN_ODD = ("even", "odd")
parity = {name: (EVEN_ODD[digit % 2]) for digit, name in digit_names.items()}


def source_digits(v: View) -> list[int]:
    """The original MNIST digits sitting behind a view's items."""
    return sorted({int(np.argmax(mnist[i][1])) for i in v.resolve_indices()})


relabel_first = View(mnist, operations=[Relabel(parity, EVEN_ODD), ClassFilter([0])])
filter_first = View(mnist, operations=[ClassFilter([0]), Relabel(parity, EVEN_ODD)])

print("vocabulary after Relabel:", relabel_first.metadata.get("index2label"))
print()
print("Relabel -> ClassFilter([0]):", len(relabel_first), "images, source digits", source_digits(relabel_first))
print("ClassFilter([0]) -> Relabel:", len(filter_first), "images, source digits", source_digits(filter_first))

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert "index2label" in relabel_first.metadata
assert relabel_first.metadata["index2label"] == {0: "even", 1: "odd"}
assert source_digits(relabel_first) == [0, 2, 4, 6, 8]
assert source_digits(filter_first) == [0]
assert len(relabel_first) > len(filter_first)

# %% [markdown]
# The same two operations in the opposite order give very different results. With `Relabel`
# first, `ClassFilter([0])` reads each datum *through* it, so class `0` means the **new**
# vocabulary's `even` — every even digit is kept. With `ClassFilter` first, class `0` still
# means the source digit `zero`, so only zeros survive.

# %% [markdown]
# ## Writing your own Operation
#
# Subclass {class}`.Operation` and implement `apply(view)`. Inside `apply` you have three
# levers, and you may use any combination of them:
#
# | lever | how | use it for |
# | --- | --- | --- |
# | cardinality / order | set `view.selection` | filtering, reordering, limiting, resampling |
# | content | `view.map(fn)` | rewriting each datum |
# | metadata | override `apply_metadata()` | rewriting dataset-level metadata |
#
# `view.selection` is the list of surviving *source* indices. `view.map(fn)` registers a
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


class Invert(Operation):
    """Content: invert pixel intensities. Registers a transform, reads no data here."""

    def apply(self, view: View) -> None:
        view.map(lambda datum: (255 - np.asarray(datum[0]), datum[1], datum[2]))


custom = View(mnist, operations=[Limit(100), KeepEvery(10), Invert()])

print("custom view size:", len(custom))
print("source indices:", custom.resolve_indices())
print(
    "a background pixel — source:",
    int(np.asarray(mnist[0][0])[0, 0, 0]),
    "view:",
    int(np.asarray(custom[0][0])[0, 0, 0]),
)

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert custom.resolve_indices() == list(range(0, 100, 10))
assert np.array_equal(np.asarray(custom[0][0]), 255 - np.asarray(mnist[0][0]))

# %% [markdown]
# Note that `Invert` never touched the data while building the view — it only *registered* a
# transform, which runs when an item is read. Operations that filter must read the data to
# decide what to keep, so they scan once at construction; pure transforms stay free until
# access.

# %% [markdown]
# ## Built-in operations
#
# | operation | what it does |
# | --- | --- |
# | {class}`.ClassFilter` | keep datums containing one of the given classes (and mask other detections for object detection) |
# | {class}`.ClassBalance` | resample to a balanced class distribution, optionally to a target size |
# | {class}`.Limit` | cap to the first *n* of the current selection |
# | {class}`.Shuffle` | randomize order, optionally seeded |
# | {class}`.Reverse` | reverse the current order |
# | {class}`.Indices` | keep the given source indices |
# | {class}`.Relabel` | rewrite labels into a target vocabulary, dropping out-of-vocabulary datums |

# %% [markdown]
# ## Tracking what a view did
#
# A view records its own lineage, which is useful for reproducibility and for writing sidecar
# metadata alongside derived datasets.

# %%
nested = View(View(mnist, [ClassFilter([0, 1])]), [Limit(25)])

print("operations, innermost first:", nested.operation_groups)
print("original dataset:", type(nested.root).__name__)
print("source indices behind the first 5 items:", nested.resolve_indices()[:5])

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(nested.operation_groups) == 2
assert nested.root is mnist

# %% [markdown]
# ## Summary
#
# - {class}`.View` wraps a dataset with an ordered pipeline of {class}`.Operation` steps,
#   lazily and without copying.
# - Operations run in the order given, each seeing the result of the previous — so order is
#   meaningful, and an operation may appear more than once.
# - Operations that inspect data read it *through* earlier operations.
# - Write your own by subclassing {class}`.Operation` and implementing `apply(view)`.

# %% [markdown]
# ## Next steps
#
# - [Ontology](../concepts/Ontology.md) — Learn about taxonomic ontologies and label space mapping.
# - [Dataset Bias and Coverage](../concepts/DatasetBias.md) — Understand coverage and balance in dataset views.
# - [How to conform and merge datasets with different label vocabularies](./h2_conform_and_merge_datasets.py) — Conform and merge datasets with disparate label vocabularies into a unified view.
# - [How to align label spaces across datasets](./h2_align_label_spaces.py) — Establish typed correspondences between source and reference label spaces.
# - [How to detect undersampled data subsets](./h2_detect_undersampling.py) — Identify undersampled or low-density regions in dataset views.
