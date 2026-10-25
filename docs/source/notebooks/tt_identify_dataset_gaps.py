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
# # Identify gaps in a labeled dataset
#
# In this tutorial you will combine an ontology, class labels, and image embeddings to find gaps in a dataset that no
# single measurement shows on its own. You will work with real CIFAR-10 imagery and real embeddings from a pretrained
# ResNet18.
#
# Estimated time to complete: 15 minutes
#
# Relevant ML stages: [Data Engineering](../getting-started/roles/ML_Lifecycle.md#data-engineering)
#
# Relevant personas: [Data Engineer](../getting-started/roles/data_engineer.md), [Data Scientist](../getting-started/roles/data_scientist.md), [T&E Engineer](../getting-started/roles/te_engineer.md)

# %% [markdown]
# ## What you will do
#
# - Define an {class}`.Ontology` of the classes a model is meant to recognize
# - Read flat per-class counts with {func}`.label_stats`
# - Measure label-space coverage with {class}`.Representation`. This shows which classes and whole branches are missing.
# - Extract image embeddings from CIFAR-10 with a pretrained ResNet18
# - Measure embedding-space coverage with {class}`.Coverage`. This shows which present classes lack variety: clustered,
#   one-dimensional, or padded with duplicate frames.
# - Combine all three into a single per-class gap report
#
# ## What you will learn
#
# - You will learn that "is my dataset complete?" has two orthogonal axes: which categories you have (labels) and how
#   varied each one is (embeddings). You need both.
# - You will learn that a class can look healthy by count yet still be a gap on either axis.

# %% [markdown]
# ## What you will need
#
# - Basic familiarity with Python
# - A Python environment with `dataeval`, `maite-datasets`, and `torchvision` installed

# %% [markdown]
# ## Introduction
#
# A dataset is complete relative to what your model must recognize. It can fall short in two independent ways:
#
# - **Label-space gaps** occur when a sanctioned class has no examples, or a whole branch of the taxonomy is empty. Flat
#   counts cannot show this: a class with zero examples contributes zero rows, so it does not appear at all. An
#   **ontology** supplies the list of what should exist, and {class}`.Representation` measures coverage against it.
# - **Embedding-space gaps** occur when a class is present and well counted, but its images lack variety.
#   {class}`.Coverage` measures this from image embeddings along three independent axes: **dispersion** (is the class
#   clustered into a tiny region?), **isotropy** (does it vary in many directions, or just one?), and **near-duplicate
#   fraction** (is a share of the class the same frame repeated?).
#
# A dataset can pass one axis and fail the other. You will build a small CIFAR-10 collection where each axis catches a
# different gap, then combine them into one report.

# %% [markdown]
# ## Setup
#
# Begin by importing the pieces you need. You will load real CIFAR-10 images and extract embeddings from them with a
# pretrained ResNet18. See [Assess an unlabeled data space](./tt_assess_data_space.py) for a deeper look at the
# embedding-only gap workflow with clustering and outliers.

# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval maite-datasets
except Exception:
    pass

# %%
from collections.abc import Iterable, Mapping
from typing import Any

import dataeval_plots as dep
import numpy as np
import plotly.io as pio
import polars as pl
import torch
from maite_datasets.image_classification import CIFAR10
from sklearn.decomposition import PCA
from torchvision.models import ResNet18_Weights, resnet18

from dataeval import Embeddings, Metadata, Ontology
from dataeval.config import set_device
from dataeval.core import label_stats
from dataeval.extractors import TorchExtractor
from dataeval.protocols import DatasetMetadata, DatumMetadata
from dataeval.scope import Coverage, Representation

pl.Config.set_tbl_rows(20)  # show every class in the tables below, not a truncated view

# Use the GPU if one is available, otherwise the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_device(device)

# Render the embedding plots below as interactive plotly figures.
dep.set_default_backend("plotly")
pio.renderers.default = "notebook"  # embed the plotly JS in the notebook output

# %% [markdown]
# ## The ontology
#
# Start with the sanctioned classes your model must recognize. For an everyday object
# recognizer trained on CIFAR-10, the ten categories split into vehicles and animals, each
# resolved to specific types. You build the {class}`.Ontology` from a nested dictionary with
# {meth}`.Ontology.from_hierarchy`.

# %%
ontology = Ontology.from_hierarchy({
    "subject": {
        "vehicle": {
            "wheeled": ["automobile", "truck"],
            "airborne": ["airplane"],
            "waterborne": ["ship"],
        },
        "animal": {
            "mammal": ["cat", "dog", "deer", "horse"],
            "avian": ["bird"],
            "amphibian": ["frog"],
        },
    },
})
print(ontology)
print("sanctioned leaf classes:", len(ontology.leaves))

# %% [markdown]
# ## The dataset
#
# Suppose a collection was scraped together for this recognizer. You start from real CIFAR-10 imagery and assemble a
# subset that looks healthy by count. Almost every class has plenty of images, but several were collected in ways that
# each hide a different kind of gap:
#
# - Four classes (`airplane`, `cat`, `dog`, `deer`) are plentiful and naturally varied (300 random images each).
# - `ship` is plentiful but clustered: its 300 images are all copies of a single frame with tiny pixel jitter, the same
#   stock photo returned over and over.
# - `truck` is plentiful but padded with duplicates: 200 varied real trucks plus 100 copies of one frame. A scrape
#   duplicated a single image into an otherwise varied class.
# - `automobile` is plentiful but one-dimensional: 300 frames interpolating between two cars, so the class spreads out
#   yet varies along a single axis (for example, one scene under a continuous pan).
# - `bird` is thin: only 30 images were collected.
# - `frog` and `horse` were never collected. The ontology sanctions them, but they are absent.
#
# `ship`, `truck`, and `automobile` each look healthy by count, and each hides a different embedding-space gap that a
# different one of `Coverage`'s signals will catch. First, load CIFAR-10 and group the image indices by class.

# %%
base = CIFAR10(root="./data", image_set="base", download=True)
cifar_labels = base.index2label  # {0: "airplane", 1: "automobile", ...}

cifar_targets = np.array([int(np.asarray(base[i][1]).argmax()) for i in range(len(base))])
by_class = {name: np.where(cifar_targets == index)[0] for index, name in cifar_labels.items()}
print("CIFAR-10 classes available:", list(by_class))

# %% [markdown]
# Now assemble the subset, encoding one gap per class as described above.

# %%
rng = np.random.default_rng(0)

healthy = ["airplane", "cat", "dog", "deer"]
present = healthy + ["automobile", "truck", "bird", "ship"]  # `frog`/`horse` never collected.
index2label = dict(enumerate(present))
name_to_index = {name: index for index, name in index2label.items()}

images: list[np.ndarray] = []
labels: list[int] = []


def add(name: str, made: Iterable[np.ndarray]) -> None:
    """Append a class's images (real or constructed) to the growing subset."""
    for image in made:
        images.append(np.asarray(image))
        labels.append(name_to_index[name])


def real(name: str, size: int) -> list[np.ndarray]:
    """``size`` random real CIFAR-10 images for ``name``."""
    return [np.asarray(base[int(i)][0]) for i in rng.choice(by_class[name], size=size, replace=False)]


def jittered(frame: np.ndarray, count: int) -> list[np.ndarray]:
    """``count`` near-duplicate copies of one frame with tiny pixel jitter."""
    f = frame.astype(np.int16)
    return [np.clip(f + rng.integers(-3, 4, size=f.shape), 0, 255).astype(np.uint8) for _ in range(count)]


# Four well-populated, naturally varied classes.
for name in healthy:
    add(name, real(name, 300))

# `bird`: under-collected — a handful of real images (a label-space gap).
add("bird", real("bird", 30))

# `ship`: clustered — 300 near-duplicate copies of a single frame.
add("ship", jittered(np.asarray(base[int(rng.choice(by_class["ship"]))][0]), 300))

# `truck`: padded with duplicates — 200 varied real trucks plus 100 copies of one frame.
add("truck", real("truck", 200))
add("truck", jittered(np.asarray(base[int(rng.choice(by_class["truck"]))][0]), 100))

# `automobile`: one-dimensional — 300 frames interpolating between two real cars, so the class
# spreads out but varies along a single axis.
car_a, car_b = (
    np.asarray(base[int(i)][0]).astype(np.float32) for i in rng.choice(by_class["automobile"], 2, replace=False)
)
add("automobile", [np.clip((1 - t) * car_a + t * car_b, 0, 255).astype(np.uint8) for t in np.linspace(0, 1, 300)])


class Cifar10Subset:
    """A minimal image-classification dataset over real CIFAR-10 images: one-hot targets + an index2label."""

    def __init__(
        self, dataset_id: str, images: Iterable[np.ndarray], labels: Iterable[int], index2label: Mapping[int, str]
    ) -> None:
        self._images = list(images)
        self._labels = list(labels)
        self._index2label = dict(index2label)
        self.metadata = DatasetMetadata(id=dataset_id, index2label=self._index2label)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> tuple[Any, Any, DatumMetadata]:
        onehot = np.zeros(len(self._index2label), dtype=np.float32)
        onehot[self._labels[index]] = 1.0
        return self._images[index], onehot, {"id": index}


dataset = Cifar10Subset("cifar10-gaps", images, labels, index2label)
metadata = Metadata(dataset)
print("images:", len(dataset), "| classes present:", len(present))

# %% [markdown]
# ## Flat counts
#
# Start with {func}`.label_stats`, which gives the per-class counts. The counts look mostly healthy: seven
# well-populated classes and one thin one.

# %%
stats = label_stats(metadata.class_labels, index2label=metadata.index2label)
counts = {metadata.index2label[i]: c for i, c in stats["label_counts_per_class"].items()}

print("classes present:", stats["class_count"])
for name in sorted(counts, key=lambda n: counts[n], reverse=True):
    print(f"  {name:>12}  {counts[name]}")

# %% [markdown]
# Counts cannot show two things: `frog` and `horse` are missing entirely (no rows to count), and nothing here tells you
# whether `ship`'s 300 images are varied or all the same. The next two sections address both.

# %% [markdown]
# ## Label-space coverage
#
# Now run {class}`.Representation`, which compares the labels to the ontology and returns a collection worklist: which
# classes to acquire (none collected) or augment (under-represented), relative to an even spread across the sanctioned
# classes.

# %%
representation = Representation(ontology).evaluate(metadata)

print(f"leaf coverage: {representation.leaf_coverage:.0%} of sanctioned classes have any examples")
print(representation.data().select(["concept", "parent", "action", "count", "target", "deficit"]))

# %% [markdown]
# The ontology shows what the counts did not: two classes have zero examples, and `dark_branches` rolls `frog` up to a
# whole branch of the taxonomy (`amphibian`) being empty.

# %%
print(representation.dark_branches)

# %% [markdown]
# On the label axis, the gaps are clear: the entire `amphibian` branch is missing, `horse` is missing, and `bird` is
# under-collected. The next question is whether the classes you do have are varied.

# %% [markdown]
# ## Embedding extraction
#
# To measure variety you need a numeric representation of each image. Extract a feature vector per image with a
# [pretrained ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) from
# Torchvision. Point a {class}`.TorchExtractor` at its penultimate `avgpool` layer (its learned features, not the
# 1000-class logits), give it the model's standard resize-and-normalize preprocessing as `transforms`, and run the
# dataset through {class}`.Embeddings`. That yields a 512-dimensional feature per image.
#
# For other ways to build embeddings, see [How to create embeddings](./h2_create_embeddings.py).

# %%
resnet = resnet18(weights=ResNet18_Weights.DEFAULT, progress=False)
extractor = TorchExtractor(resnet, transforms=ResNet18_Weights.DEFAULT.transforms(), layer_name="avgpool")
features = np.asarray(Embeddings(dataset, extractor=extractor, batch_size=64))
print("feature shape:", features.shape)

# %% [markdown]
# ### Reduce to 128 dimensions with PCA
#
# {class}`.Coverage` estimates isotropy (in how many independent directions a class varies) from an effective-dimension
# calculation that is only well defined when a class has more samples than embedding dimensions. The largest classes
# here have 300 images, so at 512-d every class falls short and isotropy comes back null. Reducing to 128 dimensions
# with PCA, which keeps the highest-variance axes, lowers the dimension count below the 300-sample classes, so their
# isotropy becomes measurable. The thin `bird` class (30 images) is still reported as null. Fit PCA once across the
# whole dataset, never per class.

# %%
embeddings = PCA(n_components=128, random_state=0).fit_transform(features)
print("embedding shape:", embeddings.shape)

# %% [markdown]
# :::{note}
#
# ResNet18 keeps this tutorial fast. If you need sharper separation between varied and near-duplicate classes on your
# own data, swap in a larger backbone such as `torchvision.models.resnet50`. The rest of the workflow is unchanged.
#
# :::

# %% [markdown]
# ## Embedding-space coverage
#
# Finally, run {class}`.Coverage`, which describes each class with three independent variety signals. Each is a gap that
# raw counts cannot see:
#
# - `dispersion`: the magnitude of spread (mean distance to centroid, relative to a typical class). A value near 1 is
#   typical, and a value well below 1 means the class is clustered.
# - `isotropy`: the shape of that spread (in how many independent directions the class varies). A value well below the
#   typical class means it varies along one axis, even when dispersion is fine.
# - `near_duplicate_fraction`: redundancy (the share of the class in unusually tight nearest-neighbor pairs). A high
#   value means a share of the class is repeated frames, even when dispersion is fine.
#
# Classes below `min_class_samples` are unassessed. `isotropy` needs more samples still, so a thin class shows `null`.

# %%
coverage = Coverage(num_observations=20, min_class_samples=20).evaluate(metadata, embeddings=embeddings)

print(coverage.data().select(["class", "count", "dispersion", "isotropy", "near_duplicate_fraction", "assessable"]))

# %% [markdown]
# Three classes look healthy by count, yet each trips a different signal while the other two signals stay normal:
#
# - `ship` has dispersion an order of magnitude below the rest: its 300 images collapse into a tiny region of the space,
#   so the class is clustered.
# - `automobile` has normal dispersion but isotropy near zero: it spreads out, yet all along a single axis, because the
#   interpolation gave it one degree of freedom.
# - `truck` has normal dispersion and isotropy but a high near-duplicate fraction (about a third): that share of it sits
#   in near-identical pairs, the duplicated frame.
#
# None of these is visible in the counts, and only `ship` would have surfaced from dispersion alone.

# %% [markdown]
# ### See the shapes
#
# You can see all three gaps. Project the embeddings down to 2D with PCA and color by class. The varied classes spread
# into broad clouds, `ship` collapses to a single tight knot (clustered), `automobile` stretches into a thin line
# (one-dimensional), and `truck` is a normal cloud with a dense clump off to one side (the duplicated frame). Hover and
# zoom to explore.
#
# PCA is the right choice here because it preserves relative spread, so a tight class still looks tight.
# Neighbor-embedding methods like t-SNE or UMAP renormalize local density and would mask the effects you want to see.

# %%
dep.project(
    embeddings,
    method="pca",
    labels=metadata.class_labels,
    label_names=metadata.index2label,
    title="CIFAR-10 embeddings in 2D (PCA) — ship knots, automobile lines, truck clumps",
)

# %% [markdown]
# ## The combined gap report
#
# Now join the two axes into one per-class view over every sanctioned class. A class is a label gap if it is missing or
# under-collected. It is an embedding gap if one of the three signals trips: clustered (low dispersion), one-dimensional
# (low isotropy), or padded with duplicates (high near-duplicate fraction). Each maps to a different fix.

# %%
worklist = representation.data()
missing = set(worklist.filter(pl.col("action") == "acquire")["concept"].to_list())
under_collected = set(worklist.filter(pl.col("action") == "augment")["concept"].to_list())

cov = coverage.data()
dispersion = dict(zip(cov["class"].to_list(), cov["dispersion"].to_list(), strict=True))
isotropy = dict(zip(cov["class"].to_list(), cov["isotropy"].to_list(), strict=True))
near_duplicate = dict(zip(cov["class"].to_list(), cov["near_duplicate_fraction"].to_list(), strict=True))

# Peer-relative cutoffs: "well below a typical (median) class".
disp_cutoff = 0.5 * float(np.median([v for v in dispersion.values() if v is not None]))
iso_cutoff = 0.5 * float(np.median([v for v in isotropy.values() if v is not None]))
NEAR_DUPLICATE_CUTOFF = 0.1  # more than 10% of the class sitting in near-duplicate pairs

report = []
for leaf in ontology.leaves:
    disp, iso, ndup = dispersion.get(leaf), isotropy.get(leaf), near_duplicate.get(leaf)
    if leaf in missing:
        label_status, gap = "missing", "acquire data (label gap)"
    elif leaf in under_collected:
        label_status, gap = "under-collected", "collect more (label gap)"
    elif disp is not None and disp < disp_cutoff:
        label_status, gap = "present", "broaden (clustered)"
    elif ndup is not None and ndup > NEAR_DUPLICATE_CUTOFF:
        label_status, gap = "present", "deduplicate (repeated frames)"
    elif iso is not None and iso < iso_cutoff:
        label_status, gap = "present", "vary the axis (one-dimensional)"
    else:
        label_status, gap = "present", "ok"
    report.append({
        "class": leaf,
        "count": int(counts.get(leaf, 0)),
        "label_status": label_status,
        "dispersion": round(disp, 2) if disp is not None else None,
        "isotropy": round(iso, 2) if iso is not None else None,
        "near_dup": round(ndup, 2) if ndup is not None else None,
        "gap": gap,
    })

gap_order = {
    "acquire data (label gap)": 0,
    "collect more (label gap)": 1,
    "broaden (clustered)": 2,
    "deduplicate (repeated frames)": 3,
    "vary the axis (one-dimensional)": 4,
    "ok": 5,
}
report.sort(key=lambda row: (gap_order[str(row["gap"])], str(row["class"])))
print(pl.DataFrame(report))

# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
report_by_class = {row["class"]: row for row in report}
assert missing == {"frog", "horse"}  # never collected -> acquire
assert under_collected == {"bird"}  # thin -> augment
assert "amphibian" in representation.dark_branches["concept"].to_list()  # frog's branch is dark
assert report_by_class["ship"]["gap"] == "broaden (clustered)"  # low dispersion
assert report_by_class["truck"]["gap"] == "deduplicate (repeated frames)"  # high near-duplicate fraction
assert report_by_class["automobile"]["gap"] == "vary the axis (one-dimensional)"  # low isotropy
assert all(report_by_class[c]["gap"] == "ok" for c in ["airplane", "cat", "dog", "deer"])  # the varied classes

# %% [markdown]
# Each check caught a gap the others missed:
#
# - Counts alone would only have pointed you toward `bird`.
# - The ontology added the missing `horse` and the empty `amphibian` branch.
# - The embeddings added three of your largest classes: `ship` (clustered), `automobile` (one-dimensional), and `truck`
#   (duplicate-padded). None of these is visible in the counts, and only `ship` is visible from dispersion alone.
#
# Each gap maps to a different fix: acquire new classes, collect more of the thin ones, broaden the clustered one, vary
# the one-dimensional one, and deduplicate the padded one.

# %% [markdown]
# ## Conclusion
#
# In this tutorial you turned a set of counts that looked healthy into a concrete, prioritized collection plan. You saw
# that "is my dataset complete?" is two questions: {class}`.Representation` answered which categories you have against
# an {class}`.Ontology`, and {class}`.Coverage` answered how varied each one is along three independent axes
# (dispersion, isotropy, near-duplicate fraction). Each axis surfaces a gap that the counts, the ontology, and the other
# two signals could miss.

# %% [markdown]
# ## Next steps
#
# - [Ontology concepts](../concepts/Ontology.md) — Learn how hierarchies and taxons organize dataset class labels and target scopes.
# - [Dataset bias and coverage](../concepts/DatasetBias.md) — Read about dispersion, isotropy, and representation metrics for assessing dataset coverage.
# - [Assess data space tutorial](./tt_assess_data_space.py) — Discover coverage gaps, clusters, and outliers in unlabeled dataset embeddings.
# - [Reconcile labels against an ontology](./h2_reconcile_labels_ontology.py) — Map and validate dataset labels against a target domain ontology.

# %% [markdown]
# ## On your own
#
# Swap CIFAR-10 for your own dataset: build (or load) an {class}`.Ontology` of your sanctioned classes, extract
# embeddings with an extractor, and run the same three checks. The combined report is your collection plan.
