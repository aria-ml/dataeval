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
# # How to build dataset views with View and Operation
#
# Analysis rarely runs on a dataset as it ships. {class}`.View` builds a curated version of it without copying: it
# applies an ordered pipeline of {class}`.Operation` steps lazily and works anywhere a dataset does.
#
# Use a view when you need:
#
# - only certain classes, a balanced distribution or a capped size
# - a reproducible shuffle or labels rewritten into a shared vocabulary
#
# This guide uses MNIST; any {class}`~dataeval.protocols.AnnotatedDataset` works.

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
from dataeval.data import Filter, Limit, Operation, Relabel, Shuffle, View

set_seed(0)  # For reproducibility

# %% [markdown]
# ## Create a view
#
# You use the MNIST test split (10,000 handwritten digits).

# %%
mnist = MNIST("./data", image_set="test", download=True)
print("source dataset size:", len(mnist))
print("source vocabulary:", mnist.metadata.get("index2label"))

# %% [markdown]
# A {class}`.View` takes a dataset and a list of operations. It behaves like a dataset:
# `len()`, indexing, and iteration all work. Nothing is copied, and the source is
# untouched.

# %%
view = View(mnist, operations=[Filter(classes=[0, 1]), Limit(500)])

print("view size:", len(view))
print("source is unchanged:", len(mnist))
print("first 10 source indices behind the view:", view.resolve_indices()[:10])

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(view) == 500
assert len(mnist) == 10000

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
limit_then_filter = View(mnist, operations=[Limit(500), Filter(classes=[0, 1])])
filter_then_limit = View(mnist, operations=[Filter(classes=[0, 1]), Limit(500)])

print("Limit(500) -> Filter(classes=[0, 1]):", len(limit_then_filter), "(0s and 1s *within* the first 500)")
print("Filter(classes=[0, 1]) -> Limit(500):", len(filter_then_limit), "(the first 500 0s and 1s)")

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(limit_then_filter) < len(filter_then_limit) == 500

# %% [markdown]
# Because each operation applies to the current selection, an operation can appear more than
# once. This pipeline takes a reproducible random sample from a specific window of the
# dataset, which a single pass could not express.

# %%
windowed = View(mnist, operations=[Limit(2000), Shuffle(seed=0), Limit(10)])
print("10 random items drawn from the first 2,000:", windowed.resolve_indices())

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(windowed) == 10
assert all(i < 2000 for i in windowed.resolve_indices())

# %% [markdown]
# ### Operations see the result of earlier operations
#
# Operations that inspect the data read it through the operations before them. That means a
# filter placed after a relabeling step matches against the new vocabulary, not the
# original one.
#
# Here you collapse the ten digits into `even`/`odd` with {class}`.Relabel`, then filter on the
# relabeled classes.

# %%
digit_names = mnist.metadata.get("index2label", {})
EVEN_ODD = ("even", "odd")
parity = {name: (EVEN_ODD[digit % 2]) for digit, name in digit_names.items()}


def source_digits(v: View) -> list[int]:
    """The original MNIST digits sitting behind a view's items."""
    return sorted({int(np.argmax(mnist[i][1])) for i in v.resolve_indices()})


relabel_first = View(mnist, operations=[Relabel(parity, EVEN_ODD), Filter(classes=[0])])
filter_first = View(mnist, operations=[Filter(classes=[0]), Relabel(parity, EVEN_ODD)])

print("vocabulary after Relabel:", relabel_first.metadata.get("index2label"))
print()
print("Relabel -> Filter(classes=[0]):", len(relabel_first), "images, source digits", source_digits(relabel_first))
print("Filter(classes=[0]) -> Relabel:", len(filter_first), "images, source digits", source_digits(filter_first))

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert "index2label" in relabel_first.metadata
assert relabel_first.metadata["index2label"] == {0: "even", 1: "odd"}
assert source_digits(relabel_first) == [0, 2, 4, 6, 8]
assert source_digits(filter_first) == [0]
assert len(relabel_first) > len(filter_first)

# %% [markdown]
# With `Relabel` first, class `0` means the new vocabulary's `even`, so every even digit
# is kept. With `Filter` first, class `0` still means the source digit `zero`, so only zeros
# survive.

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
# `Invert` never touched the data while building the view. It only registered a
# transform, which runs when an item is read. Operations that filter must read the data to
# decide what to keep, so they scan once at construction. Pure transforms stay free until
# access.

# %% [markdown]
# ## Track what a view did
#
# A view records its own lineage, which is useful for reproducibility and for writing sidecar
# metadata alongside derived datasets.

# %%
nested = View(View(mnist, [Filter(classes=[0, 1])]), [Limit(25)])

print("operations, innermost first:", nested.operation_groups)
print("original dataset:", type(nested.root).__name__)
print("source indices behind the first 5 items:", nested.resolve_indices()[:5])

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert len(nested.operation_groups) == 2
assert nested.root is mnist

# %% [markdown]
# ## Next steps
#
# - [Ontology](../concepts/Ontology.md) — Learn about taxonomic ontologies and label space mapping.
# - [Dataset Bias and Coverage](../concepts/DatasetBias.md) — Understand coverage and balance in dataset views.
# - [How to conform and merge datasets with different label vocabularies](./h2_conform_and_merge_datasets.py) — Conform
#   and merge datasets with disparate label vocabularies into a unified view.
