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
# # How to conform and merge datasets with different label vocabularies
#
# A second, incoming dataset can arrive labeled under a different scheme: different class names, reordered indices and
# an overlapping but not identical set of classes. You cannot concatenate it with your reference dataset, because index
# `0` might mean `"sedan"` in one and `"submarine"` in the other. This guide conforms the incoming dataset to the
# reference vocabulary, an {class}`.Ontology`, with {func}`.label_alignment` and {class}`.Relabel`, then merges the two
# with {func}`.merge_datasets`. See [ontology alignment](./h2_align_label_spaces.py).
#
# Use it to combine annotation sources that name or index their classes differently, or to bring an incoming dataset
# into a fixed reference taxonomy before analysis.
#
# You need a reference ontology and two datasets.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval
except Exception:
    pass

# %%
from collections import Counter
from collections.abc import Iterable, Mapping

import numpy as np

from dataeval import Ontology
from dataeval.core import label_alignment
from dataeval.data import Relabel, View, merge_datasets
from dataeval.protocols import DatasetMetadata

# %% [markdown]
# Conforming and merging act only on labels and `index2label`, so the images
# are irrelevant here. The example uses a small in-memory dataset with blank images. A real
# {class}`~dataeval.protocols.AnnotatedDataset` (e.g. a `maite-datasets` loader or
# your own) works identically.


# %% tags=["hide-input"]
class ToyDataset:
    """A minimal image-classification dataset: one-hot targets + an index2label."""

    def __init__(self, dataset_id: str, labels: Iterable[int], index2label: Mapping[int, str]) -> None:
        self._labels = list(labels)
        self._index2label = dict(index2label)
        self.metadata = DatasetMetadata(id=dataset_id, index2label=self._index2label)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int):
        onehot = np.zeros(len(self._index2label), dtype=np.float32)
        onehot[self._labels[index]] = 1.0
        return np.zeros((3, 8, 8), dtype=np.float32), onehot, {"id": index}


def labels_from_counts(counts: dict[str, int], index2label: dict[int, str]) -> list[int]:
    """Expand ``{class_name: count}`` into a per-image list of label indices."""
    name_to_index = {name: index for index, name in index2label.items()}
    return [name_to_index[name] for name, count in counts.items() for _ in range(count)]


# %% [markdown]
# ## Define the reference vocabulary
#
# The reference vocabulary is an {class}`.Ontology`, the label space everything will
# be expressed in. Here is a taxonomy of vessels grouped by domain (air / land /
# water). Note that it has no undersea branch: `"submarine"` is therefore
# out-of-vocabulary and will be dropped when conforming.

# %%
reference_ontology = Ontology.from_hierarchy({
    "aircraft": {"airliner": None, "fighter jet": None},
    "land vehicle": {"sedan": None, "pickup truck": None},
    "watercraft": {"frigate": None, "cargo ship": None},
})
print(reference_ontology)
print("reference vocabulary:", {i: reference_ontology.concept(c).label for i, c in enumerate(reference_ontology.ids)})

# %% [markdown]
# ## Conform the reference dataset
#
# The reference data uses its own label ordering. You align its class names to the
# ontology and conform it. Every class is in the vocabulary, so this is lossless: it
# re-expresses the labels in the reference's index space.

# %%
REFERENCE_VOCAB = {0: "sedan", 1: "pickup truck", 2: "frigate", 3: "cargo ship"}
reference_raw = ToyDataset(
    "reference",
    labels_from_counts({"sedan": 6, "pickup truck": 4, "frigate": 5, "cargo ship": 3}, REFERENCE_VOCAB),
    REFERENCE_VOCAB,
)

reference_alignment = label_alignment(reference_raw.metadata.get("index2label", {}).values(), reference_ontology)
reference = View(reference_raw, [Relabel(reference_alignment["class_remap"], reference_ontology)])

print("reference images:", len(reference))
print("reference now uses the vocabulary:", reference.metadata.get("index2label"))

# %% [markdown]
# ## Align the incoming labels to the reference vocabulary
#
# The incoming data uses a different, reordered `index2label` (note `submarine`
# is index `0` here), and an overlapping-but-different class set: `sedan` and
# `frigate` overlap the reference, `fighter jet` is new (but in the vocabulary), and
# `submarine` is not in the reference vocabulary.

# %%
INCOMING_VOCAB = {0: "submarine", 1: "frigate", 2: "sedan", 3: "fighter jet"}
incoming = ToyDataset(
    "incoming",
    labels_from_counts({"submarine": 4, "frigate": 3, "sedan": 5, "fighter jet": 2}, INCOMING_VOCAB),
    INCOMING_VOCAB,
)
print("incoming vocabulary:", incoming.metadata.get("index2label"))

# %% [markdown]
# {func}`.label_alignment` relates the incoming class names to the reference
# ontology. Matching is by name, so the reordered indices are irrelevant: `sedan`,
# `frigate`, and `fighter jet` map by equivalence, while `submarine` is
# out-of-vocabulary.

# %%
incoming_alignment = label_alignment(incoming.metadata.get("index2label", {}).values(), reference_ontology)
for c in incoming_alignment["correspondences"]:
    print(f"  {c.source:>12}  {c.relation:<11} -> {c.target}")
print("out-of-vocabulary:", incoming_alignment["unaligned_source"])

# %% [markdown]
# ## Conform the incoming dataset
#
# {class}`.Relabel` applies the alignment: it drops the out-of-vocabulary `submarine`
# images, rewrites the remaining labels into the reference index space, and replaces
# the dataset's `index2label` with the reference vocabulary.

# %%
relabel = Relabel(incoming_alignment["class_remap"], reference_ontology)
incoming_conformed = View(incoming, [relabel])

print("kept", len(incoming_conformed), "of", len(incoming), "images")
print("dropped (out-of-vocabulary):", dict(relabel.dropped))
print(
    "incoming now uses the reference vocabulary:",
    incoming_conformed.metadata.get("index2label") == reference.metadata.get("index2label"),
)

# %% [markdown]
# ## Merge the datasets
#
# Both datasets now share one `index2label`, so {func}`.merge_datasets` can combine
# them into a single dataset view. The per-class counts show the union: `sedan` and
# `frigate` come from both datasets, `pickup truck`/`cargo ship` from the reference,
# and `fighter jet` from the incoming data.

# %%
merged = merge_datasets(reference, incoming_conformed)
print("merged images:", len(merged))

index2label = merged.metadata.get("index2label", {})
counts = Counter(index2label[int(np.argmax(datum[1]))] for datum in merged)
print("per-class counts:", dict(counts))

# %% [markdown]
# Conforming first is what makes the merge sound. Merging the datasets before
# conforming raises a `ValueError`, because their label vocabularies do not line up.

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
try:
    merge_datasets(reference, incoming)
    raise AssertionError("merging before conforming should raise")
except ValueError:
    pass

# %% [markdown]
# ## Next steps
#
# - [Ontology](../concepts/Ontology.md) — Explore the taxonomic model the reference is built on, correspondences,
#   relations, and mergeability.
# - [How to align two label spaces](./h2_align_label_spaces.py) — Relate class names and categories between different
#   dataset label spaces.
